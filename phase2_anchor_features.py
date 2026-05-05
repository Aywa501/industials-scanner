"""Phase 2 prep: sample exclusionary features at each of the 316 anchors.

Goal: find features that ALL anchors share, so we can use them as
"definitely-not-a-candidate" filters when sampling clean negatives.

For each anchor we sample (via GEE):
- NLCD 2019 land cover class
- Slope in degrees (USGS 3DEP 10m)
- Elevation in meters
- State code (TIGER)

Reports distributions and proposes thresholds that drop ~0 anchors.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import ee
import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

GCP_PROJECT = os.getenv("GCP_PROJECT", "")
ANCHORS_CSV = Path(__file__).parent.parent / "data_us" / "manufacturing_announcements_geocoded.csv"
OUT_PARQUET = Path(__file__).parent / ".artifacts" / "anchor_features.parquet"

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


def init_ee() -> None:
    sa = os.getenv("GEE_SERVICE_ACCOUNT") or ""
    key = os.getenv("GEE_KEY_FILE") or ""
    if sa and key:
        ee.Initialize(ee.ServiceAccountCredentials(sa, key_file=key), project=GCP_PROJECT)
    else:
        ee.Initialize(project=GCP_PROJECT)


def sample_features(anchors: pd.DataFrame) -> pd.DataFrame:
    fc = ee.FeatureCollection([
        ee.Feature(ee.Geometry.Point([float(r["lng"]), float(r["lat"])]),
                   {"row_idx": int(i)})
        for i, r in anchors.iterrows()
    ])

    nlcd = ee.Image("USGS/NLCD_RELEASES/2019_REL/NLCD/2019").select("landcover")
    dem = ee.Image("USGS/3DEP/10m")
    slope = ee.Terrain.slope(dem)
    states = ee.FeatureCollection("TIGER/2018/States")

    img = (nlcd.rename("nlcd")
           .addBands(slope.rename("slope_deg"))
           .addBands(dem.rename("elevation_m")))

    sampled = img.sampleRegions(
        collection=fc, scale=30, geometries=False, tileScale=4
    ).getInfo()

    feats = []
    for f in sampled["features"]:
        p = f["properties"]
        feats.append({
            "row_idx": int(p["row_idx"]),
            "nlcd": int(p["nlcd"]) if p.get("nlcd") is not None else None,
            "slope_deg": float(p["slope_deg"]) if p.get("slope_deg") is not None else None,
            "elevation_m": float(p["elevation_m"]) if p.get("elevation_m") is not None else None,
        })
    return pd.DataFrame(feats).set_index("row_idx").sort_index()


def report(df: pd.DataFrame) -> None:
    print(f"\nrows: {len(df)}")
    print(f"missing nlcd: {df.nlcd.isna().sum()}, slope: {df.slope_deg.isna().sum()}, "
          f"elev: {df.elevation_m.isna().sum()}")

    print("\n=== NLCD class distribution (pre-construction land cover) ===")
    nlcd_counts = df.nlcd.value_counts().sort_index()
    for cls, n in nlcd_counts.items():
        name = NLCD_CLASS_NAMES.get(int(cls), f"unknown_{cls}")
        pct = 100 * n / len(df)
        print(f"  {int(cls):3d} {name:25s}  {n:4d} ({pct:5.1f}%)")

    excluded_classes = {11, 12, 41, 42, 43, 90, 95}
    n_in_excluded = df.nlcd.isin(excluded_classes).sum()
    print(f"\n  in proposed-exclude classes (water/ice/forest/wetland): "
          f"{n_in_excluded} / {len(df)}")

    print("\n=== Slope (degrees) ===")
    print(df.slope_deg.describe(percentiles=[.5, .9, .95, .99]).to_string())

    print("\n=== Elevation (meters) ===")
    print(df.elevation_m.describe(percentiles=[.5, .9, .95, .99]).to_string())


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
