"""Phase 2 prep v2: extended exclusionary-feature analysis at the 316 anchors.

Adds to v1:
- NLCD majority class within 3×3 / 5×5 / 7×7 windows (90/150/210 m radius)
- Distance to nearest TIGER primary+secondary road (m)
- Distance to nearest NLCD developed pixel (m)

Reports per-anchor distributions and proposes thresholds that drop ~0 anchors.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import ee
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

GCP_PROJECT = os.getenv("GCP_PROJECT", "")
ANCHORS_CSV = Path(__file__).parent.parent / "data_us" / "manufacturing_announcements_geocoded.csv"
OUT_PARQUET = Path(__file__).parent / ".artifacts" / "anchor_features_v2.parquet"

NLCD_CLASS_NAMES = {
    11: "open_water", 12: "ice_snow",
    21: "dev_open", 22: "dev_low", 23: "dev_med", 24: "dev_high",
    31: "barren",
    41: "forest_deciduous", 42: "forest_evergreen", 43: "forest_mixed",
    51: "dwarf_scrub", 52: "shrub",
    71: "grassland", 72: "sedge", 73: "lichens", 74: "moss",
    81: "pasture", 82: "cultivated_crops",
    90: "woody_wetland", 95: "herbaceous_wetland",
}

EXCLUDE_NLCD = {11, 12, 90, 95}


def init_ee() -> None:
    sa = os.getenv("GEE_SERVICE_ACCOUNT") or ""
    key = os.getenv("GEE_KEY_FILE") or ""
    if sa and key:
        ee.Initialize(ee.ServiceAccountCredentials(sa, key_file=key), project=GCP_PROJECT)
    else:
        ee.Initialize(project=GCP_PROJECT)


def _build_fc(anchors: pd.DataFrame) -> ee.FeatureCollection:
    return ee.FeatureCollection([
        ee.Feature(ee.Geometry.Point([float(r["lng"]), float(r["lat"])]),
                   {"row_idx": int(i)})
        for i, r in anchors.iterrows()
    ])


def _sample(img: ee.Image, fc: ee.FeatureCollection, scale: int) -> dict:
    sampled = img.sampleRegions(
        collection=fc, scale=scale, geometries=False, tileScale=4
    ).getInfo()
    out: dict[int, dict] = {}
    for f in sampled.get("features", []):
        p = f["properties"]
        out[int(p["row_idx"])] = p
    return out


def sample_features(anchors: pd.DataFrame) -> pd.DataFrame:
    fc = _build_fc(anchors)

    nlcd = ee.Image("USGS/NLCD_RELEASES/2019_REL/NLCD/2019").select("landcover")
    dem = ee.Image("USGS/3DEP/10m")  # deprecated but still works; the
                                     # _collection variant loses native
                                     # projection when mosaicked
    slope = ee.Terrain.slope(dem)

    mode_3 = nlcd.reduceNeighborhood(
        reducer=ee.Reducer.mode(), kernel=ee.Kernel.square(radius=1)
    ).rename("nlcd_3x3")
    mode_5 = nlcd.reduceNeighborhood(
        reducer=ee.Reducer.mode(), kernel=ee.Kernel.square(radius=2)
    ).rename("nlcd_5x5")
    mode_7 = nlcd.reduceNeighborhood(
        reducer=ee.Reducer.mode(), kernel=ee.Kernel.square(radius=3)
    ).rename("nlcd_7x7")

    base = (nlcd.rename("nlcd")
            .addBands(slope.rename("slope_deg"))
            .addBands(dem.rename("elevation_m"))
            .addBands(mode_3).addBands(mode_5).addBands(mode_7))
    print("[1/3] sampling NLCD point + buffers + slope + elevation...")
    base_dict = _sample(base, fc, scale=30)
    print(f"  got {len(base_dict)} anchors")

    roads = (ee.FeatureCollection("TIGER/2016/Roads")
             .filter(ee.Filter.inList("mtfcc", ["S1100", "S1200"])))
    print("[2/3] computing per-anchor distance to nearest primary/secondary road...")

    SEARCH_M = 10000  # 10 km — every industrial site is much closer than this

    def add_road_dist(feat: ee.Feature) -> ee.Feature:
        pt = feat.geometry()
        box = pt.buffer(SEARCH_M).bounds()
        nearby = roads.filterBounds(box)
        d = ee.Algorithms.If(
            nearby.size().gt(0),
            pt.distance(nearby.geometry(), maxError=10),
            ee.Number(SEARCH_M),
        )
        return feat.set("road_dist_m", d)

    # Process in chunks to stay under per-request memory limit
    CHUNK = 64
    road_dict: dict[int, dict] = {}
    feats = fc.toList(fc.size()).getInfo()
    for start in range(0, len(feats), CHUNK):
        chunk = ee.FeatureCollection(feats[start:start + CHUNK])
        result = chunk.map(add_road_dist).getInfo()
        for f in result.get("features", []):
            p = f["properties"]
            road_dict[int(p["row_idx"])] = p
        print(f"  road dist: {len(road_dict)}/{len(feats)}")

    developed = nlcd.gte(21).And(nlcd.lte(24))
    dev_dist = (developed.fastDistanceTransform().sqrt().multiply(30)
                .rename("dev_dist_m"))
    print("[3/3] sampling distance to nearest NLCD developed pixel...")
    dev_dict = _sample(dev_dist, fc, scale=30)
    print(f"  got {len(dev_dict)} anchors")

    rows = []
    for ridx, p in base_dict.items():
        merged = dict(p)
        if ridx in road_dict:
            merged["road_dist_m"] = road_dict[ridx].get("road_dist_m")
        if ridx in dev_dict:
            merged["dev_dist_m"] = dev_dict[ridx].get("dev_dist_m")
        rows.append({
            "row_idx": ridx,
            "nlcd": int(merged["nlcd"]) if merged.get("nlcd") is not None else None,
            "slope_deg": float(merged["slope_deg"]) if merged.get("slope_deg") is not None else None,
            "elevation_m": float(merged["elevation_m"]) if merged.get("elevation_m") is not None else None,
            "nlcd_3x3": int(merged["nlcd_3x3"]) if merged.get("nlcd_3x3") is not None else None,
            "nlcd_5x5": int(merged["nlcd_5x5"]) if merged.get("nlcd_5x5") is not None else None,
            "nlcd_7x7": int(merged["nlcd_7x7"]) if merged.get("nlcd_7x7") is not None else None,
            "road_dist_m": float(merged["road_dist_m"]) if merged.get("road_dist_m") is not None else None,
            "dev_dist_m": float(merged["dev_dist_m"]) if merged.get("dev_dist_m") is not None else None,
        })
    return pd.DataFrame(rows).set_index("row_idx").sort_index()


def report(df: pd.DataFrame) -> None:
    print(f"\nrows: {len(df)}")
    for c in ["nlcd", "slope_deg", "elevation_m", "nlcd_3x3", "nlcd_5x5",
              "nlcd_7x7", "road_dist_m", "dev_dist_m"]:
        print(f"  missing {c}: {df[c].isna().sum()}")

    print("\n=== NLCD point vs majority window ===")
    for col, label in [("nlcd", "point"), ("nlcd_3x3", "3×3 (90 m)"),
                        ("nlcd_5x5", "5×5 (150 m)"), ("nlcd_7x7", "7×7 (210 m)")]:
        in_excl = df[col].isin(EXCLUDE_NLCD).sum()
        print(f"  {label:14s}  anchors in {{water, ice, wetlands}}: {in_excl} / {len(df)}")

    # Show which anchors recover with each buffer
    print("\n=== per-anchor: which weird point-class anchors get reclassified by larger buffer? ===")
    weird_pt = df[df.nlcd.isin(EXCLUDE_NLCD)].copy()
    print(f"  {len(weird_pt)} anchors have weird point-class NLCD")
    for col, label in [("nlcd_3x3", "3×3"), ("nlcd_5x5", "5×5"), ("nlcd_7x7", "7×7")]:
        recovered = weird_pt[~weird_pt[col].isin(EXCLUDE_NLCD)]
        print(f"  {label} buffer recovers {len(recovered)} of {len(weird_pt)} weird anchors → "
              f"surviving anchors in excl class: {len(weird_pt) - len(recovered)}")

    print("\n=== Slope (degrees) ===")
    print(df.slope_deg.describe(percentiles=[.5, .9, .95, .99]).to_string())

    print("\n=== Distance to nearest TIGER primary+secondary road (m) ===")
    print(df.road_dist_m.describe(percentiles=[.5, .9, .95, .99]).to_string())
    print(f"  fraction with road_dist > 1 km: {(df.road_dist_m > 1000).mean():.3f}")
    print(f"  fraction with road_dist > 2 km: {(df.road_dist_m > 2000).mean():.3f}")
    print(f"  fraction with road_dist > 5 km: {(df.road_dist_m > 5000).mean():.3f}")
    print(f"  max road_dist (m): {df.road_dist_m.max():.0f}")

    print("\n=== Distance to nearest NLCD developed pixel (m) ===")
    print(df.dev_dist_m.describe(percentiles=[.5, .9, .95, .99]).to_string())
    print(f"  fraction with dev_dist > 100 m: {(df.dev_dist_m > 100).mean():.3f}")
    print(f"  fraction with dev_dist > 500 m: {(df.dev_dist_m > 500).mean():.3f}")
    print(f"  fraction with dev_dist > 1 km: {(df.dev_dist_m > 1000).mean():.3f}")


def main() -> int:
    if not GCP_PROJECT:
        print("error: GCP_PROJECT must be set", file=sys.stderr)
        return 1
    init_ee()

    anchors = pd.read_csv(ANCHORS_CSV)
    anchors = anchors.dropna(subset=["lat", "lng"]).reset_index(drop=True)
    print(f"anchors: {len(anchors)}")

    df = sample_features(anchors)
    df = df.join(anchors[["canonical_project_name", "state", "lat", "lng"]])

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PARQUET)
    print(f"wrote {OUT_PARQUET}")

    report(df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
