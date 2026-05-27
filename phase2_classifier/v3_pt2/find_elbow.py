"""Test for an empirical elbow on a random 500-polygon sample of the cohort.

The cohort itself is our negative-ish baseline (user noted: most US industrial
stock is pre-2008, so a random cohort sample is mostly negatives). If the
classifier discriminates, we should see:
  - Bimodal histogram on the key discriminator(s) (NIR/red, R/B)
  - A valley between built and natural modes -> that's the elbow

If everything is unimodal smooth, no metric is going to fix it without higher
spatial resolution.

Also compare against the 31 positives so we can see where they sit on the
distribution.
"""
import os
import time
import numpy as np
import pandas as pd
import requests
import rasterio
from concurrent.futures import ThreadPoolExecutor, as_completed
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
from rasterio.transform import from_bounds as transform_from_bounds
from rasterio.features import rasterize
from shapely.wkb import loads as wkb_loads

os.environ.setdefault("AWS_REQUEST_PAYER", "requester")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("GDAL_HTTP_MULTIPLEX", "YES")
os.environ.setdefault("GDAL_HTTP_VERSION", "2")
os.environ.setdefault("CPL_VSIL_CURL_USE_HEAD", "NO")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
POLYS = os.path.join(ROOT, "..", "data_us/phase2/v3/stage2_candidate_polygons.parquet")
CANDS = os.path.join(ROOT, "..", "data_us/phase2/v3/stage2_candidates.parquet")
POS_CALIB = os.path.join(ROOT, "..", "data_us/phase2/v3/cover_class_positives.parquet")
OUT = os.path.join(ROOT, "..", "data_us/phase2/v3/elbow_sample.parquet")

STAC = "https://landsatlook.usgs.gov/stac-server/search"
MAX_SCENES = 12
MARGIN_M = 30
N_SAMPLE = 200
BANDS = ("blue", "green", "red", "nir08", "swir16", "swir22")


def _stac_search(lat, lon):
    body = {
        "collections": ["landsat-c2l2-sr"],
        "intersects": {"type": "Point", "coordinates": [float(lon), float(lat)]},
        "datetime": "2008-06-01T00:00:00Z/2008-09-30T23:59:59Z",
        "query": {"eo:cloud_cover": {"lt": 40}, "platform": {"in": ["LANDSAT_7"]}},
        "limit": MAX_SCENES,
    }
    for attempt in range(5):
        try:
            r = requests.post(STAC, json=body, timeout=30)
            if r.status_code == 200:
                return r.json().get("features", [])
            time.sleep(2 ** attempt)
        except Exception:
            time.sleep(2 ** attempt)
    return []


def _href(feat, band_key):
    a = feat.get("assets", {}).get(band_key)
    if a is None:
        return None
    href = a.get("alternate", {}).get("s3", {}).get("href") or a.get("href", "")
    return href if href.startswith("s3://") else None


def _expand_bbox_ll(bbox, margin_m=MARGIN_M):
    cy = (bbox[1] + bbox[3]) / 2
    dlat = margin_m / 111_000
    dlon = margin_m / (111_000 * np.cos(np.radians(cy)))
    return (bbox[0] - dlon, bbox[1] - dlat, bbox[2] + dlon, bbox[3] + dlat)


def _fetch_median(hrefs, bbox_ll):
    stack = []
    with rasterio.Env():
        for href in hrefs:
            try:
                with rasterio.open(href) as src:
                    bbox_utm = transform_bounds("EPSG:4326", src.crs, *bbox_ll)
                    win = from_bounds(*bbox_utm, transform=src.transform).round_offsets().round_lengths()
                    if win.width < 2 or win.height < 2:
                        continue
                    arr = src.read(1, window=win, boundless=True, fill_value=0).astype(np.float32)
                refl = arr * 2.75e-5 - 0.2
                refl[arr == 0] = np.nan
                refl[(refl < 0) | (refl > 1)] = np.nan
                if np.isnan(refl).all():
                    continue
                stack.append(refl)
            except Exception:
                continue
    if not stack:
        return None
    h = min(a.shape[0] for a in stack)
    w = min(a.shape[1] for a in stack)
    return np.nanmedian(np.stack([a[:h, :w] for a in stack], axis=0), axis=0)


def _poly_mask(shape, bbox_ll, polygon_wkb):
    h, w = shape
    ex0, ey0, ex1, ey1 = bbox_ll
    try:
        poly = wkb_loads(polygon_wkb)
        transform = transform_from_bounds(ex0, ey0, ex1, ey1, w, h)
        mask = rasterize([(poly, 1)], out_shape=(h, w), transform=transform,
                         fill=0, dtype=np.uint8, all_touched=True).astype(bool)
        return mask if mask.any() else None
    except Exception:
        return None


def score(bbox_ll, polygon_wkb, hrefs_by_band):
    expanded = _expand_bbox_ll(bbox_ll)
    medians = {}
    for b in BANDS:
        med = _fetch_median(hrefs_by_band[b], expanded)
        if med is None:
            return {"error": f"no {b}"}
        medians[b] = med
    hs = min(m.shape[0] for m in medians.values())
    ws = min(m.shape[1] for m in medians.values())
    medians = {k: v[:hs, :ws] for k, v in medians.items()}
    mask = _poly_mask((hs, ws), expanded, polygon_wkb)
    if mask is None:
        return {"error": "no mask"}
    mask = mask[:hs, :ws]
    valid = mask.copy()
    for v in medians.values():
        valid &= ~np.isnan(v)
    if not valid.any():
        return {"error": "no valid pixels"}
    return {b: float(np.nanmean(medians[b][valid])) for b in BANDS} | {"pixels": int(valid.sum())}


def main():
    print("=== Sampling 500 polygons from full 344K cohort ===")
    cands = pd.read_parquet(CANDS, columns=["building_id", "ovt_id", "xmin", "ymin", "xmax", "ymax", "lat", "lon"])
    polys = pd.read_parquet(POLYS).set_index("ovt_id")
    rng = np.random.default_rng(2026)
    sample = cands.sample(n=N_SAMPLE, random_state=42).reset_index(drop=True)

    work = []
    for c in sample.itertuples():
        if c.ovt_id not in polys.index:
            continue
        work.append({
            "building_id": c.building_id,
            "lat": c.lat, "lon": c.lon,
            "wkb": polys.loc[c.ovt_id]["geometry_wkb"],
            "bbox": (float(c.xmin), float(c.ymin), float(c.xmax), float(c.ymax)),
        })
    print(f"  {len(work)} polygons to score")

    scene_cache = {}
    def _hrefs_for_band(lat, lon, band):
        key = (round(lat, 1), round(lon, 1))
        if key not in scene_cache:
            scene_cache[key] = _stac_search(lat, lon)
        return [h for h in (_href(f, band) for f in scene_cache[key]) if h]

    def _do(w):
        bbox = w["bbox"]
        cy, cx = (bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2
        hrefs_by_band = {b: _hrefs_for_band(cy, cx, b) for b in BANDS}
        res = score(bbox, w["wkb"], hrefs_by_band)
        res["building_id"] = w["building_id"]
        res["lat"] = w["lat"]
        res["lon"] = w["lon"]
        return res

    t0 = time.time()
    results = []
    done = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(_do, w) for w in work]
        for fut in as_completed(futs):
            results.append(fut.result())
            done += 1
            if done % 100 == 0 or done == len(futs):
                print(f"  {done}/{len(futs)}  elapsed={time.time()-t0:.0f}s", flush=True)
    print(f"done in {time.time()-t0:.0f}s")

    df = pd.DataFrame(results)
    df.to_parquet(OUT, index=False)
    print(f"wrote {OUT}")

    valid = df[df["red"].notna()].copy()
    valid["nir_over_red"] = valid["nir08"] / valid["red"]
    valid["r_over_b"] = valid["red"] / valid["blue"]
    valid["ndvi"] = (valid["nir08"] - valid["red"]) / (valid["nir08"] + valid["red"])
    valid["brightness"] = (valid["red"] + valid["green"] + valid["blue"]) / 3

    print(f"\n=== {len(valid)} valid polygons ===")

    # Load positives for comparison
    pos = pd.read_parquet(POS_CALIB)
    pos = pos[pos["red"].notna()].copy()
    pos["nir_over_red"] = pos["nir08"] / pos["red"]
    pos["r_over_b"] = pos["red"] / pos["blue"]

    print("\n=== Histogram of NIR/red (looking for bimodality) ===")
    print("Random cohort sample:")
    bins = [0.7, 0.85, 1.0, 1.05, 1.10, 1.20, 1.30, 1.50, 2.00, 3.00, 5.00, 10.0, 20.0]
    cohort_hist, _ = np.histogram(valid["nir_over_red"].clip(0, 20), bins=bins)
    pos_hist, _ = np.histogram(pos["nir_over_red"].clip(0, 20), bins=bins)
    total_c = cohort_hist.sum()
    total_p = pos_hist.sum()
    print(f"{'NIR/red range':<18} {'cohort %':>10} {'cohort n':>10} {'positives %':>13} {'pos n':>8}")
    print("-" * 70)
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        cp = 100 * cohort_hist[i] / total_c
        pp = 100 * pos_hist[i] / total_p
        bar = '█' * int(cp/2)
        print(f"  {lo:>4.2f}–{hi:<6.2f}     {cp:>8.1f}%  {cohort_hist[i]:>8}  {pp:>11.1f}%  {pos_hist[i]:>6}  {bar}")

    print("\n=== Cumulative distribution — find the elbow ===")
    print(f"{'NIR/red >= T':<15}  {'cohort kept':>15}  {'positives kept':>16}")
    for t in [0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 1.30, 1.40, 1.50, 1.70, 2.00, 2.50, 3.00]:
        c_keep = (valid["nir_over_red"] >= t).mean()
        p_keep = (pos["nir_over_red"] >= t).mean()
        print(f"  T={t:.2f}              {c_keep:>13.1%}  {p_keep:>14.1%}")

    print("\n=== Per-band cohort signature (mean across sample) ===")
    for c in ["blue", "green", "red", "nir08", "swir16", "swir22"]:
        v = valid[c]
        print(f"  {c:<8}  p25={v.quantile(.25):.3f}  p50={v.quantile(.5):.3f}  p75={v.quantile(.75):.3f}")

    print("\n=== Where do cohort polygons cluster? ===")
    # Coarse 2D bins of (NIR/red, R/B)
    cohort_clean = valid[(valid["nir_over_red"] > 0.5) & (valid["nir_over_red"] < 20) &
                        (valid["r_over_b"] > 0.5) & (valid["r_over_b"] < 5)]
    nir_bins = [0.7, 1.0, 1.1, 1.3, 1.7, 3.0, 20]
    rb_bins = [0.5, 1.0, 1.1, 1.2, 1.4, 1.8, 5]
    H, _, _ = np.histogram2d(cohort_clean["nir_over_red"], cohort_clean["r_over_b"], bins=[nir_bins, rb_bins])
    print(f"{'NIR/R\\R/B':<12}", end="")
    for j in range(len(rb_bins) - 1):
        print(f"{rb_bins[j]:.1f}-{rb_bins[j+1]:.1f}".center(10), end="")
    print()
    for i in range(len(nir_bins) - 1):
        print(f"{nir_bins[i]:.1f}-{nir_bins[i+1]:.1f}    ", end="")
        for j in range(len(rb_bins) - 1):
            print(f"{int(H[i,j]):>5}     ", end="")
        print()


if __name__ == "__main__":
    main()
