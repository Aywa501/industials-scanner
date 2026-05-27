"""Per-polygon color signature for the 31 known greenfields, 2008 L7 SR.

No dilation, no buffer beyond the polygon — every pixel touched by the polygon
counts equally. Pull blue/green/red/nir, compute mean per band, derive:
  - NDVI = (NIR - red) / (NIR + red)
  - brightness = mean(red, green, blue)
  - nir_over_red = NIR / red

Apply the rule from the discussion:
  natural if NDVI > 0.30                            (clear vegetation)
  natural if 0.10 < NDVI <= 0.30                    (sparse veg / dry cropland)
  natural if NDVI <= 0.10 AND brightness <= 0.30    (dark or moderate, includes desert via NIR check below)
                                AND NIR > 1.1*red   (desert / sparse natural)
  built   if NDVI <= 0.10 AND brightness > 0.30 AND NIR <= 1.1*red   (concrete/metal/bright roof)
  built   if NDVI <= 0.10 AND brightness <= 0.15 AND NIR <= 1.1*red  (asphalt-like)
  ambiguous otherwise -> treat as natural (recall-first)
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
MATCHES = os.path.join(ROOT, "..", "data_us/phase2/v3/announcement_polygon_matches.parquet")
POLYS = os.path.join(ROOT, "..", "data_us/phase2/v3/stage2_candidate_polygons.parquet")
CANDS = os.path.join(ROOT, "..", "data_us/phase2/v3/stage2_candidates.parquet")
OUT = os.path.join(ROOT, "..", "data_us/phase2/v3/color_calibration.parquet")

STAC = "https://landsatlook.usgs.gov/stac-server/search"
MAX_SCENES = 12
MARGIN_M = 30
BANDS = ("blue", "green", "red", "nir08")


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
    """L2 SR scaling: refl = DN * 2.75e-5 - 0.2. Valid [0,1]."""
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
    """No dilation, no buffer — every pixel touched by the polygon, exactly."""
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


def classify(ndvi, brightness, nir_over_red):
    """Return one of {natural, built, ambiguous}. See docstring at top."""
    if np.isnan(ndvi) or np.isnan(brightness) or np.isnan(nir_over_red):
        return "missing"
    if ndvi > 0.30:
        return "natural"  # vegetation
    if ndvi > 0.10:
        return "natural"  # sparse veg / dry cropland
    # ndvi <= 0.10
    if nir_over_red > 1.1:
        return "natural"  # desert / very sparse natural with some NIR boost
    # nir ≈ red
    if brightness > 0.30:
        return "built"    # concrete / metal / bright roof
    if brightness < 0.15:
        return "built"    # asphalt-like (treat dark+no-veg as built; recall risk acknowledged)
    return "ambiguous"    # moderate brightness, no NIR boost — keep (recall-first)


def score(bbox_ll, polygon_wkb, hrefs_by_band):
    expanded = _expand_bbox_ll(bbox_ll)
    medians = {}
    for b in BANDS:
        med = _fetch_median(hrefs_by_band[b], expanded)
        if med is None:
            return {"error": f"no {b}"}
        medians[b] = med
    # Align shapes
    hs = min(m.shape[0] for m in medians.values())
    ws = min(m.shape[1] for m in medians.values())
    medians = {k: v[:hs, :ws] for k, v in medians.items()}
    mask = _poly_mask((hs, ws), expanded, polygon_wkb)
    if mask is None:
        return {"error": "no mask"}
    mask = mask[:hs, :ws]
    valid = mask & ~np.isnan(medians["red"]) & ~np.isnan(medians["nir08"]) & ~np.isnan(medians["green"]) & ~np.isnan(medians["blue"])
    if not valid.any():
        return {"error": "no valid pixels"}
    means = {b: float(np.nanmean(medians[b][valid])) for b in BANDS}
    red, nir, green, blue = means["red"], means["nir08"], means["green"], means["blue"]
    ndvi = (nir - red) / (nir + red) if (nir + red) > 0 else np.nan
    brightness = (red + green + blue) / 3
    nir_over_red = nir / red if red > 0 else np.nan
    cls = classify(ndvi, brightness, nir_over_red)
    return {
        "red": red, "green": green, "blue": blue, "nir": nir,
        "ndvi": ndvi, "brightness": brightness, "nir_over_red": nir_over_red,
        "pixels": int(valid.sum()),
        "class": cls,
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
    print(f"sites: {len(usable)}  polygons: {len(work)}")

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

    # Per-site aggregate: take the polygon classified most favourably (recall-first).
    # natural > ambiguous > built > missing
    rank = {"natural": 0, "ambiguous": 1, "built": 2, "missing": 3, None: 3}
    by_site = {}
    for r in results:
        by_site.setdefault(r["site_idx"], []).append(r)

    rows = []
    for si, site in enumerate(usable.itertuples(index=False)):
        per = by_site.get(si, [])
        # Pick the polygon with the most natural-looking classification (lowest rank)
        per_sorted = sorted(per, key=lambda r: rank.get(r.get("class"), 99))
        chosen = per_sorted[0] if per_sorted else {}
        rows.append({
            "project": site.project, "state": site.state, "lat": site.lat, "lng": site.lng,
            "quality": site.quality, "n_polys": len(per),
            "best_class": chosen.get("class"),
            "best_ndvi": chosen.get("ndvi"),
            "best_brightness": chosen.get("brightness"),
            "best_nir_over_red": chosen.get("nir_over_red"),
            "best_red": chosen.get("red"),
            "best_green": chosen.get("green"),
            "best_blue": chosen.get("blue"),
            "best_nir": chosen.get("nir"),
            "best_pixels": chosen.get("pixels"),
            "per_poly_classes": [r.get("class") for r in per],
            "per_poly_ndvi": [r.get("ndvi") for r in per],
            "per_poly_brightness": [r.get("brightness") for r in per],
        })

    out = pd.DataFrame(rows)
    out["ll_key"] = out["lat"].round(5).astype(str) + ',' + out["lng"].round(5).astype(str)
    out = out.drop_duplicates("ll_key").drop(columns=["ll_key"]).reset_index(drop=True)
    out.to_parquet(OUT, index=False)
    print(f"\nwrote {OUT}: {len(out)} unique sites")

    print(f"\n=== Per-site classification (best-of-cluster) ===")
    print(out["best_class"].value_counts(dropna=False))
    recall = (out["best_class"].isin(["natural", "ambiguous"])).mean()
    print(f"\nRecall (natural + ambiguous): {recall:.2%}  ({(out['best_class'].isin(['natural','ambiguous'])).sum()}/{len(out)})")

    print(f"\n=== Misclassified (built) — these are sites we'd wrongly drop ===")
    miss = out[out["best_class"] == "built"]
    for r in miss.itertuples():
        print(f"  {r.state}  ndvi={r.best_ndvi:+.3f}  bright={r.best_brightness:.3f}  nir/red={r.best_nir_over_red:.2f}  "
              f"r={r.best_red:.2f} g={r.best_green:.2f} b={r.best_blue:.2f} nir={r.best_nir:.2f}  "
              f"{r.project[:50]}")

    print(f"\n=== Per-band ranges across all sites ===")
    for c in ["best_ndvi", "best_brightness", "best_nir_over_red", "best_red", "best_green", "best_blue", "best_nir"]:
        v = out[c].dropna()
        if len(v):
            print(f"  {c:<20}  min={v.min():+.3f}  p25={v.quantile(.25):+.3f}  p50={v.quantile(.5):+.3f}  p75={v.quantile(.75):+.3f}  max={v.max():+.3f}")


if __name__ == "__main__":
    main()
