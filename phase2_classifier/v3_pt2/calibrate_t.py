"""Calibrate Stage 2b's change threshold T against the 316 known-changed greenfields.

Per known-greenfield announcement (post-filter from match_announcements.py):
  1. Pull every candidate polygon within 500m of the site (the "cluster").
  2. For each polygon, score change_2008_vs_2026 with the polygon-mask method.
     - 2008 baseline: L7 ETM+ pan median composite, Jun-Sep, cloud<40, ≤12 scenes.
     - 2026 endpoint: L8 OLI + L9 OLI-2 pan median composite, Jun-Sep, cloud<40, ≤12 scenes.
  3. Take MAX change over the cluster -> the best signal Stage 2b could find for this site.
  4. The 95th-percentile recall point in the max-change distribution -> empirical T.

This is local-only (~40 sites × few scenes each, single process, no EC2 needed).
"""
import os
import sys
import time
import json
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
OUT = os.path.join(ROOT, "..", "data_us/phase2/v3/t_calibration.parquet")

STAC = "https://landsatlook.usgs.gov/stac-server/search"
MAX_SCENES = 12
MARGIN_M = 15

# 2008 ETM+; 2026 OLI/OLI-2
YEARS = {
    2008: {"collections": ["landsat-c2l1"], "datetime": "2008-06-01T00:00:00Z/2008-09-30T23:59:59Z",
           "platforms": ["LANDSAT_7"], "band_key": "pan"},  # ETM+ band 8
    2026: {"collections": ["landsat-c2l1"], "datetime": "2026-04-01T00:00:00Z/2026-10-31T23:59:59Z",
           "platforms": ["LANDSAT_8", "LANDSAT_9"], "band_key": "pan"},  # OLI/OLI-2 band 8
}


def _stac_search(lat, lon, year):
    cfg = YEARS[year]
    body = {
        "collections": cfg["collections"],
        "intersects": {"type": "Point", "coordinates": [float(lon), float(lat)]},
        "datetime": cfg["datetime"],
        "query": {
            "eo:cloud_cover": {"lt": 40},
            "platform": {"in": cfg["platforms"]},
        },
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


def _scene_to_href(feat):
    """Return the s3:// href for the pan band asset (band 8 for both ETM+ and OLI)."""
    assets = feat.get("assets", {})
    for key in ("pan", "B8", "B08", "B8.TIF"):
        if key in assets:
            href = assets[key].get("alternate", {}).get("s3", {}).get("href") or assets[key].get("href", "")
            if href.startswith("s3://"):
                return href
    return None


def _expand_bbox_ll(bbox, margin_m=MARGIN_M):
    cy = (bbox[1] + bbox[3]) / 2
    dlat = margin_m / 111_000
    dlon = margin_m / (111_000 * np.cos(np.radians(cy)))
    return (bbox[0] - dlon, bbox[1] - dlat, bbox[2] + dlon, bbox[3] + dlat)


def _fetch_pan_median(hrefs, bbox_ll):
    stack = []
    with rasterio.Env():
        for href in hrefs:
            try:
                with rasterio.open(href) as src:
                    bbox_utm = transform_bounds("EPSG:4326", src.crs, *bbox_ll)
                    win = from_bounds(*bbox_utm, transform=src.transform).round_offsets().round_lengths()
                    if win.width < 4 or win.height < 4:
                        continue
                    arr = src.read(1, window=win, boundless=True, fill_value=0).astype(np.float32)
                arr[arr == 0] = np.nan
                if np.isnan(arr).all():
                    continue
                stack.append(arr)
            except Exception:
                continue
    if not stack:
        return None, 0
    h = min(a.shape[0] for a in stack)
    w = min(a.shape[1] for a in stack)
    return np.nanmedian(np.stack([a[:h, :w] for a in stack], axis=0), axis=0), len(stack)


def _poly_mask(pan_shape, bbox_ll, polygon_wkb):
    h, w = pan_shape
    ex0, ey0, ex1, ey1 = bbox_ll
    try:
        poly = wkb_loads(polygon_wkb)
        transform = transform_from_bounds(ex0, ey0, ex1, ey1, w, h)
        mask = rasterize([(poly, 1)], out_shape=(h, w), transform=transform,
                         fill=0, dtype=np.uint8, all_touched=True).astype(bool)
        if not mask.any() or mask.all():
            return None
        mask = binary_dilation(mask, iterations=1)
        if mask.all():
            return None
        return mask
    except Exception:
        return None


def _ratio(pan, mask):
    if pan is None or mask is None or not mask.any():
        return None
    inside = pan[mask & ~np.isnan(pan)]
    outside = pan[~mask & ~np.isnan(pan)]
    if len(inside) == 0 or len(outside) == 0:
        return None
    om = float(outside.mean())
    if om <= 0:
        return None
    return float(inside.mean()) / om


def score_polygon(bbox_ll, polygon_wkb, hrefs_2008, hrefs_2026):
    expanded = _expand_bbox_ll(bbox_ll)
    pan26, n26 = _fetch_pan_median(hrefs_2026, expanded)
    if pan26 is None:
        return {"error": "no 2026 data", "n_scenes_2026": n26}
    mask = _poly_mask(pan26.shape, expanded, polygon_wkb)
    if mask is None:
        return {"error": "no mask", "n_scenes_2026": n26}
    pan08, n08 = _fetch_pan_median(hrefs_2008, expanded)
    if pan08 is None:
        return {"error": "no 2008 data", "n_scenes_2008": n08, "n_scenes_2026": n26}
    h = min(pan08.shape[0], pan26.shape[0], mask.shape[0])
    w = min(pan08.shape[1], pan26.shape[1], mask.shape[1])
    pan08, pan26, mask = pan08[:h, :w], pan26[:h, :w], mask[:h, :w]
    r08 = _ratio(pan08, mask)
    r26 = _ratio(pan26, mask)
    if r08 is None or r26 is None:
        return {"error": "no ratio", "n_scenes_2008": n08, "n_scenes_2026": n26}
    return {
        "ratio_2008": r08, "ratio_2026": r26, "change": abs(r26 - r08),
        "footprint_pixels": int(mask.sum()),
        "n_scenes_2008": n08, "n_scenes_2026": n26,
    }


def main():
    matches = pd.read_parquet(MATCHES)
    usable = matches[matches["quality"].isin(["strong", "multi", "far"])].reset_index(drop=True)
    print(f"calibration sites (strong+multi+far): {len(usable)}")

    # Load polygons + bboxes for the matched building_ids.
    polys = pd.read_parquet(POLYS).set_index("ovt_id")
    cands = pd.read_parquet(CANDS, columns=["building_id", "ovt_id", "xmin", "ymin", "xmax", "ymax"]).set_index("building_id")

    all_bids = set()
    for bids in usable["matched_building_ids"]:
        all_bids.update(bids)
    print(f"unique polygons to score: {len(all_bids)}")

    # Cache STAC by (round_lat, round_lon, year) — adjacent polygons share scenes.
    scene_cache = {}

    def _scenes(lat, lon, year):
        key = (round(lat, 1), round(lon, 1), year)  # ~10km bucket — same path/row most likely
        if key in scene_cache:
            return scene_cache[key]
        feats = _stac_search(lat, lon, year)
        hrefs = [h for h in (_scene_to_href(f) for f in feats) if h]
        scene_cache[key] = hrefs
        return hrefs

    # Build flat work list (one row per polygon-to-score).
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
                "ovt_id": c.ovt_id,
                "wkb": polys.loc[c.ovt_id]["geometry_wkb"],
                "bbox": (float(c.xmin), float(c.ymin), float(c.xmax), float(c.ymax)),
            })
    print(f"polygons to score: {len(work)}")

    def _do(w):
        bbox = w["bbox"]
        cy = (bbox[1] + bbox[3]) / 2
        cx = (bbox[0] + bbox[2]) / 2
        hrefs08 = _scenes(cy, cx, 2008)
        hrefs26 = _scenes(cy, cx, 2026)
        res = score_polygon(bbox, w["wkb"], hrefs08, hrefs26)
        res["site_idx"] = w["site_idx"]
        res["building_id"] = w["building_id"]
        return res

    t_start = time.time()
    results = []
    done = 0
    with ThreadPoolExecutor(max_workers=32) as ex:
        futs = [ex.submit(_do, w) for w in work]
        for fut in as_completed(futs):
            results.append(fut.result())
            done += 1
            if done % 20 == 0 or done == len(futs):
                print(f"  [{done}/{len(futs)}]  elapsed={time.time() - t_start:.0f}s", flush=True)

    # Aggregate per site: max change.
    by_site = {}
    for r in results:
        si = r["site_idx"]
        by_site.setdefault(si, []).append(r)

    rows = []
    for site_idx, site in enumerate(usable.itertuples(index=False)):
        per = by_site.get(site_idx, [])
        valid_changes = [r["change"] for r in per if "change" in r]
        best = max(valid_changes) if valid_changes else None
        rows.append({
            "project": site.project, "state": site.state, "lat": site.lat, "lng": site.lng,
            "quality": site.quality, "n_500m": site.n_500m, "n_polys_scored": len(per),
            "max_change": best,
            "per_poly_changes": [r.get("change") for r in per],
            "per_poly_errors": [r.get("error") for r in per],
        })
    print(f"done in {time.time() - t_start:.0f}s")

    out = pd.DataFrame(rows)
    out.to_parquet(OUT, index=False)
    print(f"\nwrote {OUT}: {len(out)} sites")

    valid = out[out["max_change"].notna()].copy()
    print(f"\n=== Valid max_change scores: {len(valid)}/{len(out)} ===")
    if len(valid):
        print(f"distribution: min={valid['max_change'].min():.3f}  "
              f"p10={valid['max_change'].quantile(.1):.3f}  "
              f"p25={valid['max_change'].quantile(.25):.3f}  "
              f"p50={valid['max_change'].quantile(.5):.3f}  "
              f"p75={valid['max_change'].quantile(.75):.3f}  "
              f"p90={valid['max_change'].quantile(.9):.3f}  "
              f"max={valid['max_change'].max():.3f}")
        print("\n=== Recall sweep (T -> fraction of known-greenfields surviving) ===")
        for t in [0.03, 0.05, 0.07, 0.10, 0.12, 0.15, 0.20, 0.30, 0.50]:
            recall = (valid["max_change"] >= t).mean()
            print(f"  T={t:.2f}  recall={recall:.2%}  ({(valid['max_change'] >= t).sum()}/{len(valid)})")
        # 95th percentile recall point
        t95 = valid["max_change"].quantile(0.05)
        print(f"\nT at recall=95%: {t95:.3f}  (lowest 5% max_change cut off)")
        t90 = valid["max_change"].quantile(0.10)
        print(f"T at recall=90%: {t90:.3f}")


if __name__ == "__main__":
    main()
