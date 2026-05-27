"""Calibrate an NDVI_2008 classifier against the 31 known greenfields.

Hypothesis: for every site in our cohort, 2026 looks like a roof. So we only need
to ask "was the polygon's 2008 patch vegetation/cropland?" — i.e., NDVI > some T.

Per matched polygon:
  1. Pull L7 ETM+ red (B3) + NIR (B4) median composite over polygon bbox+margin,
     Jun-Sep 2008, cloud<40, ≤12 scenes.
  2. NDVI = (NIR - red) / (NIR + red)  per pixel.
  3. Rasterize polygon at 30m + 1px dilation (all_touched=True).
  4. mean_ndvi = NDVI[mask].mean()
  5. Per site, take MAX of mean_ndvi across cluster polygons (the strongest "natural in 2008" signal).
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
from scipy.ndimage import binary_dilation

os.environ.setdefault("AWS_REQUEST_PAYER", "requester")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("GDAL_HTTP_MULTIPLEX", "YES")
os.environ.setdefault("GDAL_HTTP_VERSION", "2")
os.environ.setdefault("CPL_VSIL_CURL_USE_HEAD", "NO")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MATCHES = os.path.join(ROOT, "..", "data_us/phase2/v3/announcement_polygon_matches.parquet")
POLYS = os.path.join(ROOT, "..", "data_us/phase2/v3/stage2_candidate_polygons.parquet")
CANDS = os.path.join(ROOT, "..", "data_us/phase2/v3/stage2_candidates.parquet")
OUT = os.path.join(ROOT, "..", "data_us/phase2/v3/ndvi_calibration.parquet")

STAC = "https://landsatlook.usgs.gov/stac-server/search"
MAX_SCENES = 12
MARGIN_M = 30  # 1px buffer at 30m


def _stac_search(lat, lon):
    body = {
        "collections": ["landsat-c2l2-sr"],  # Level-2 surface reflectance — already atmospherically corrected
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


def _scene_to_href(feat, band_key):
    """band_key: 'red' or 'nir08' (Landsat STAC asset names for L7)."""
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
    """L2 SR scaling: reflectance = DN * 2.75e-5 - 0.2. Valid range ~[0, 1]."""
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
                # Scale to surface reflectance
                refl = arr * 2.75e-5 - 0.2
                refl[arr == 0] = np.nan  # fill values
                refl[(refl < 0) | (refl > 1)] = np.nan  # out-of-range / saturated
                if np.isnan(refl).all():
                    continue
                stack.append(refl)
            except Exception:
                continue
    if not stack:
        return None, 0
    h = min(a.shape[0] for a in stack)
    w = min(a.shape[1] for a in stack)
    return np.nanmedian(np.stack([a[:h, :w] for a in stack], axis=0), axis=0), len(stack)


def _poly_mask(shape, bbox_ll, polygon_wkb):
    h, w = shape
    ex0, ey0, ex1, ey1 = bbox_ll
    try:
        poly = wkb_loads(polygon_wkb)
        transform = transform_from_bounds(ex0, ey0, ex1, ey1, w, h)
        mask = rasterize([(poly, 1)], out_shape=(h, w), transform=transform,
                         fill=0, dtype=np.uint8, all_touched=True).astype(bool)
        if not mask.any():
            return None
        # 1px dilation gives some edge tolerance without swamping signal at 30m
        mask = binary_dilation(mask, iterations=1)
        return mask
    except Exception:
        return None


def score_polygon_ndvi(bbox_ll, polygon_wkb, hrefs_red, hrefs_nir):
    expanded = _expand_bbox_ll(bbox_ll)
    red, nr = _fetch_median(hrefs_red, expanded)
    if red is None:
        return {"error": "no 2008 red", "n_red": nr}
    nir, nn = _fetch_median(hrefs_nir, expanded)
    if nir is None:
        return {"error": "no 2008 nir", "n_nir": nn}
    h = min(red.shape[0], nir.shape[0])
    w = min(red.shape[1], nir.shape[1])
    red, nir = red[:h, :w], nir[:h, :w]
    mask = _poly_mask((h, w), expanded, polygon_wkb)
    if mask is None or not mask.any():
        return {"error": "no mask", "n_red": nr, "n_nir": nn}
    h2 = min(mask.shape[0], red.shape[0])
    w2 = min(mask.shape[1], red.shape[1])
    red, nir, mask = red[:h2, :w2], nir[:h2, :w2], mask[:h2, :w2]
    denom = nir + red
    denom[denom == 0] = np.nan
    ndvi = (nir - red) / denom
    inside = ndvi[mask & ~np.isnan(ndvi)]
    if len(inside) == 0:
        return {"error": "no valid pixels", "n_red": nr, "n_nir": nn}
    return {
        "mean_ndvi_2008": float(np.nanmean(inside)),
        "median_ndvi_2008": float(np.nanmedian(inside)),
        "pixels": int(len(inside)),
        "n_red": nr, "n_nir": nn,
    }


def main():
    matches = pd.read_parquet(MATCHES)
    usable = matches[matches["quality"].isin(["strong", "multi", "far"])].reset_index(drop=True)
    polys = pd.read_parquet(POLYS).set_index("ovt_id")
    cands = pd.read_parquet(CANDS, columns=["building_id", "ovt_id", "xmin", "ymin", "xmax", "ymax"]).set_index("building_id")

    work = []
    for site_idx, site in enumerate(usable.itertuples(index=False)):
        for bid in site.matched_building_ids:
            if bid not in cands.index:
                continue
            c = cands.loc[bid]
            if c.ovt_id not in polys.index:
                continue
            work.append({
                "site_idx": site_idx,
                "building_id": bid,
                "wkb": polys.loc[c.ovt_id]["geometry_wkb"],
                "bbox": (float(c.xmin), float(c.ymin), float(c.xmax), float(c.ymax)),
            })
    print(f"sites: {len(usable)}  polygons to score: {len(work)}")

    scene_cache = {}
    def _hrefs(lat, lon, band_key):
        key = (round(lat, 1), round(lon, 1))
        if key not in scene_cache:
            feats = _stac_search(lat, lon)
            scene_cache[key] = feats
        return [h for h in (_scene_to_href(f, band_key) for f in scene_cache[key]) if h]

    def _do(w):
        bbox = w["bbox"]
        cy = (bbox[1] + bbox[3]) / 2
        cx = (bbox[0] + bbox[2]) / 2
        hrefs_red = _hrefs(cy, cx, "red")
        hrefs_nir = _hrefs(cy, cx, "nir08")
        res = score_polygon_ndvi(bbox, w["wkb"], hrefs_red, hrefs_nir)
        res["site_idx"] = w["site_idx"]
        res["building_id"] = w["building_id"]
        return res

    t0 = time.time()
    results = []
    done = 0
    with ThreadPoolExecutor(max_workers=32) as ex:
        futs = [ex.submit(_do, w) for w in work]
        for fut in as_completed(futs):
            results.append(fut.result())
            done += 1
            if done % 20 == 0 or done == len(futs):
                print(f"  [{done}/{len(futs)}]  elapsed={time.time()-t0:.0f}s", flush=True)
    print(f"done in {time.time()-t0:.0f}s")

    by_site = {}
    for r in results:
        by_site.setdefault(r["site_idx"], []).append(r)

    rows = []
    for si, site in enumerate(usable.itertuples(index=False)):
        per = by_site.get(si, [])
        ndvis = [r["mean_ndvi_2008"] for r in per if "mean_ndvi_2008" in r]
        best = max(ndvis) if ndvis else None
        rows.append({
            "project": site.project, "state": site.state, "lat": site.lat, "lng": site.lng,
            "quality": site.quality, "n_500m": site.n_500m,
            "n_polys_scored": len(per),
            "max_ndvi_2008": best,
            "per_poly_ndvi": [r.get("mean_ndvi_2008") for r in per],
            "per_poly_pixels": [r.get("pixels") for r in per],
            "per_poly_errors": [r.get("error") for r in per],
        })

    out = pd.DataFrame(rows)
    # Dedup by (lat,lng) — same site recorded twice in the announcements CSV
    out["ll_key"] = out["lat"].round(5).astype(str) + ',' + out["lng"].round(5).astype(str)
    out_dedup = out.drop_duplicates("ll_key").drop(columns=["ll_key"]).reset_index(drop=True)
    out_dedup.to_parquet(OUT, index=False)
    print(f"\nwrote {OUT}: {len(out_dedup)} unique sites")

    valid = out_dedup[out_dedup["max_ndvi_2008"].notna()].copy()
    print(f"valid scores: {len(valid)}/{len(out_dedup)}")

    if len(valid):
        print(f"\nNDVI_2008 distribution (known greenfields — expected HIGH):")
        for q in [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]:
            print(f"  p{int(q*100):>2}: {valid['max_ndvi_2008'].quantile(q):+.3f}")
        print(f"  min: {valid['max_ndvi_2008'].min():+.3f}")
        print(f"  max: {valid['max_ndvi_2008'].max():+.3f}")

        print(f"\n=== Recall sweep (keep if NDVI_2008 >= T → 'natural in 2008 → built since') ===")
        for t in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
            n = (valid['max_ndvi_2008'] >= t).sum()
            print(f"  T={t:+.2f}  recall={n/len(valid):.2%}  ({n}/{len(valid)})")
        print(f"\nT at recall=95%: {valid['max_ndvi_2008'].quantile(0.05):+.3f}")
        print(f"T at recall=90%: {valid['max_ndvi_2008'].quantile(0.10):+.3f}")

        print(f"\n=== Bottom 8 sites (lowest NDVI — potential misses) ===")
        for r in valid.sort_values("max_ndvi_2008").head(8).itertuples():
            px = [p for p in r.per_poly_pixels if p]
            print(f"  ndvi={r.max_ndvi_2008:+.3f}  {r.state}  polys={r.n_polys_scored}  pix={px}  {r.project[:55]}")


if __name__ == "__main__":
    main()
