"""Multi-class 2008 land-cover classifier for Stage 2b.

Pull 6-band L7 SR median composite (Jun-Sep 2008), classify each polygon into:
  vegetation, sparse_veg, sand, bare_soil, concrete, metal_roof, asphalt,
  rust_brick, water, ambiguous.

Decision: drop {concrete, metal_roof, asphalt, rust_brick, water}, keep the rest
(recall-first on ambiguous).

Tested against:
  - 31 known greenfields (positives) -> expect ~100% kept
  - ~280 polygons sampled from long-established industrial corridors (negatives)
    -> measure specificity (fraction correctly classified as built)
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
OUT_POS = os.path.join(ROOT, "..", "data_us/phase2/v3/cover_class_positives.parquet")
OUT_NEG = os.path.join(ROOT, "..", "data_us/phase2/v3/cover_class_negatives.parquet")

STAC = "https://landsatlook.usgs.gov/stac-server/search"
MAX_SCENES = 12
MARGIN_M = 30
BANDS = ("blue", "green", "red", "nir08", "swir16", "swir22")

# Long-established industrial corridors. Picked for being demonstrably pre-2008 industrial mass.
NEG_BOXES = [
    ("Houston Ship Channel",     29.65, 29.78, -95.35, -95.10),
    ("Gary IN steel district",   41.55, 41.65, -87.40, -87.25),
    ("Pittsburgh Mon Valley",    40.30, 40.50, -80.05, -79.80),
    ("Baltimore Sparrows Point", 39.20, 39.30, -76.60, -76.45),
    ("Detroit/Dearborn",         42.30, 42.40, -83.20, -82.95),
    ("South Chicago/E Chicago",  41.65, 41.75, -87.55, -87.40),
    ("Birmingham AL steel",      33.45, 33.60, -87.00, -86.75),
]
NEG_PER_BOX = 40


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
    """L7 L2 SR scaling: refl = DN * 2.75e-5 - 0.2. Valid [0,1]."""
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


def classify(b):
    """Return (class, tier_id). Higher specificity rules fire first."""
    if any(np.isnan(b.get(k, np.nan)) for k in ("blue", "green", "red", "nir08")):
        return "missing", 0

    R, G, B, N = b["red"], b["green"], b["blue"], b["nir08"]
    S1 = b.get("swir16", np.nan)
    S2 = b.get("swir22", np.nan)

    # Tier 1: clear water
    if N < 0.05 and B > R:
        return "water", 1

    # Tier 2: dense vegetation (forest / peak cropland)
    if N > 2.0 * R:
        return "vegetation", 2

    # Tier 3: bright flat-spectrum -> concrete / metal roof
    # All visible high & similar, no NIR boost
    rb = R / B if B > 0 else 99
    if R > 0.20 and rb < 1.20 and N <= 1.10 * R:
        # discriminate metal (very bright) vs concrete (moderate)
        cls = "metal_roof" if (R > 0.35 and B > 0.25) else "concrete"
        return cls, 3

    # Tier 4: dark flat-spectrum -> asphalt / low-slope roof
    if R < 0.12 and G < 0.12 and B < 0.12 and N < 0.15:
        return "asphalt", 4

    # Tier 5: rust / red brick (yellow tint, NIR <= red — iron oxide absorption)
    if rb > 1.40 and R > 0.18 and N <= 1.05 * R:
        return "rust_brick", 5

    # Tier 6: sand / desert (yellow tint with slight NIR boost — distinguishes from rust)
    if rb > 1.20 and N > 1.05 * R:
        return "sand", 6

    # Tier 7: sparse veg / dry cropland / pasture (moderate NIR boost)
    if N > 1.20 * R:
        return "sparse_veg", 7

    # Tier 8: bare soil (no NIR boost but not flat or yellow either)
    if 0.10 < R < 0.30 and rb > 1.10 and N > 0.90 * R:
        return "bare_soil", 8

    return "ambiguous", 9


# Decision: drop {water, concrete, metal_roof, asphalt, rust_brick}; keep rest.
DROP_CLASSES = {"water", "concrete", "metal_roof", "asphalt", "rust_brick"}


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
    # Pixels valid in all bands
    valid = mask.copy()
    for v in medians.values():
        valid &= ~np.isnan(v)
    if not valid.any():
        return {"error": "no valid pixels"}
    means = {b: float(np.nanmean(medians[b][valid])) for b in BANDS}
    cls, tier = classify(means)
    return {
        **means,
        "ndvi": (means["nir08"] - means["red"]) / (means["nir08"] + means["red"] + 1e-9),
        "class": cls, "tier": tier,
        "decision": "drop" if cls in DROP_CLASSES else "keep",
        "pixels": int(valid.sum()),
    }


def _build_pos_work():
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
                "label": "positive",
                "group": site.project, "state": site.state,
                "site_idx": site_idx, "building_id": bid,
                "wkb": polys.loc[c.ovt_id]["geometry_wkb"],
                "bbox": (float(c.xmin), float(c.ymin), float(c.xmax), float(c.ymax)),
            })
    return work, usable


def _build_neg_work(seed=42):
    rng = np.random.default_rng(seed)
    polys = pd.read_parquet(POLYS).set_index("ovt_id")
    cands = pd.read_parquet(CANDS, columns=["building_id", "ovt_id", "xmin", "ymin", "xmax", "ymax", "lat", "lon"])
    work = []
    for name, lat_lo, lat_hi, lon_lo, lon_hi in NEG_BOXES:
        sub = cands[(cands.lat >= lat_lo) & (cands.lat <= lat_hi) &
                    (cands.lon >= lon_lo) & (cands.lon <= lon_hi)]
        if len(sub) == 0:
            print(f"  WARN: 0 polygons in {name} box", flush=True)
            continue
        take = sub.sample(n=min(NEG_PER_BOX, len(sub)), random_state=int(rng.integers(0, 1e9)))
        for c in take.itertuples():
            if c.ovt_id not in polys.index:
                continue
            work.append({
                "label": "negative",
                "group": name, "state": None,
                "building_id": c.building_id,
                "wkb": polys.loc[c.ovt_id]["geometry_wkb"],
                "bbox": (float(c.xmin), float(c.ymin), float(c.xmax), float(c.ymax)),
            })
        print(f"  {name}: {len(sub)} available, sampled {min(NEG_PER_BOX, len(sub))}", flush=True)
    return work


def run(work, label):
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
        res["label"] = w["label"]
        res["group"] = w.get("group")
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
            if done % 50 == 0 or done == len(futs):
                print(f"  [{label}] {done}/{len(futs)}  elapsed={time.time()-t0:.0f}s", flush=True)
    print(f"[{label}] done in {time.time()-t0:.0f}s")
    return pd.DataFrame(results)


def main():
    print("=== Building positive sample (31 known greenfields) ===")
    pos_work, usable = _build_pos_work()
    print(f"  {len(pos_work)} polygon-scoring tasks (across {len(usable)} sites)")

    print("\n=== Building negative sample (pre-2008 industrial corridors) ===")
    neg_work = _build_neg_work()
    print(f"  {len(neg_work)} negative polygons sampled")

    print("\n=== Running positives ===")
    df_pos = run(pos_work, "pos")
    df_pos.to_parquet(OUT_POS, index=False)
    print(f"wrote {OUT_POS}")

    print("\n=== Running negatives ===")
    df_neg = run(neg_work, "neg")
    df_neg.to_parquet(OUT_NEG, index=False)
    print(f"wrote {OUT_NEG}")

    # Positives report
    print("\n=== POSITIVES (31 known greenfields aggregated by site) ===")
    # Aggregate to site: keep if ANY polygon classified as keep (recall-first)
    valid_pos = df_pos[df_pos["class"].notna()].copy()
    print(f"  valid scored: {len(valid_pos)}/{len(df_pos)}")
    print(f"  per-polygon class breakdown:")
    print(valid_pos["class"].value_counts().to_string())
    # Aggregate by site_idx (need to map building_id back)
    # Use the project name from work
    pos_by_site = {}
    for w in pos_work:
        pos_by_site.setdefault(w["group"], []).append(w["building_id"])

    site_keep = {}
    for proj, bids in pos_by_site.items():
        sub = valid_pos[valid_pos["building_id"].isin(bids)]
        if len(sub) == 0:
            site_keep[proj] = "missing"
            continue
        site_keep[proj] = "keep" if (sub["decision"] == "keep").any() else "drop"
    # Dedup by lat/lng (some announcements duplicate)
    site_df = pd.Series(site_keep)
    n_keep = (site_df == "keep").sum()
    print(f"\n  per-site decision (any-polygon-keep -> site kept):")
    print(f"    keep: {n_keep}, drop: {(site_df=='drop').sum()}, missing: {(site_df=='missing').sum()}")
    print(f"  RECALL: {n_keep / len(site_df):.2%}")

    if (site_df == "drop").any():
        print("  Sites we'd wrongly DROP:")
        for proj, dec in site_df.items():
            if dec == "drop":
                bids = pos_by_site[proj]
                sub = valid_pos[valid_pos["building_id"].isin(bids)]
                classes = sub["class"].tolist()
                print(f"    {proj[:60]:<60}  classes={classes}")

    # Negatives report
    print("\n=== NEGATIVES (pre-2008 industrial sample) ===")
    valid_neg = df_neg[df_neg["class"].notna()].copy()
    print(f"  valid scored: {len(valid_neg)}/{len(df_neg)}")
    print(f"  per-polygon class breakdown:")
    print(valid_neg["class"].value_counts().to_string())
    n_drop_neg = (valid_neg["decision"] == "drop").sum()
    n_keep_neg = (valid_neg["decision"] == "keep").sum()
    print(f"\n  per-polygon decision:")
    print(f"    drop: {n_drop_neg}, keep: {n_keep_neg}")
    print(f"  SPECIFICITY (frac correctly dropped): {n_drop_neg / len(valid_neg):.2%}")

    print("\n  By corridor:")
    for group, gdf in valid_neg.groupby("group"):
        n_drop = (gdf["decision"] == "drop").sum()
        spec = n_drop / len(gdf) if len(gdf) else 0
        print(f"    {group:<30}  n={len(gdf):>3}  drop={n_drop:>3}  spec={spec:.0%}")

    print("\n  False positives (kept neg polygons) — class breakdown:")
    kept_negs = valid_neg[valid_neg["decision"] == "keep"]
    if len(kept_negs):
        print(kept_negs["class"].value_counts().to_string())
        print("\n  Sample 10 kept negatives (band signature):")
        for r in kept_negs.head(10).itertuples():
            print(f"    {r.group[:25]:<25} class={r['class'] if hasattr(r,'class') else 'NA':<12}  "
                  f"R={r.red:.3f} G={r.green:.3f} B={r.blue:.3f} NIR={r.nir08:.3f} "
                  f"S1={r.swir16:.3f} S2={r.swir22:.3f}")


if __name__ == "__main__":
    main()
