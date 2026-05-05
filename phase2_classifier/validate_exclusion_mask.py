"""Phase 2 step 2a: validate the exclusion heuristic over CONUS.

The proposed exclusion (any one excludes a pixel):
    slope > 15°
    OR  NLCD class == ice/snow (12)
    OR  distance to nearest TIGER primary+secondary road > 10 km
    OR  distance to nearest NLCD developed pixel > 1.5 km

Two checks:
1. Anchor recall: fraction of the 316 anchors excluded (target: 0).
2. CONUS reduction: fraction of N random CONUS points excluded
   (rough estimate of how much of CONUS the heuristic eliminates from
    inference / negative-sampling).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import ee
import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

GCP_PROJECT = os.getenv("GCP_PROJECT", "")
ANCHORS_CSV = Path(__file__).parent.parent.parent / "data_us" / "manufacturing_announcements_geocoded.csv"

SLOPE_THRESHOLD = 15.0
ROAD_DIST_THRESHOLD = 10000.0
DEV_DIST_THRESHOLD = 1500.0
NLCD_ICE = 12

CONUS_BUFFER_M = 20000  # bbox around each point used to find nearby roads
N_CONUS_SAMPLE = 5000
SEED = 42

CONUS_STUSPS = [
    "AL", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "ID",
    "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI",
    "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY",
    "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN",
    "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "DC",
]


def init_ee() -> None:
    sa = os.getenv("GEE_SERVICE_ACCOUNT") or ""
    key = os.getenv("GEE_KEY_FILE") or ""
    if sa and key:
        ee.Initialize(ee.ServiceAccountCredentials(sa, key_file=key), project=GCP_PROJECT)
    else:
        ee.Initialize(project=GCP_PROJECT)


def evaluate_points(fc: ee.FeatureCollection, label: str) -> pd.DataFrame:
    """Evaluate the exclusion predicate at every point in fc.

    Uses per-point computation in chunks to stay under GEE memory limits.
    Each point is evaluated against:
      - NLCD point sample (point class)
      - Slope point sample (degrees)
      - Distance to nearest TIGER primary+secondary road within 20 km
      - Distance to nearest NLCD developed pixel (sampled from a precomputed
        fastDistanceTransform of the developed mask)
    """
    nlcd = ee.Image("USGS/NLCD_RELEASES/2019_REL/NLCD/2019").select("landcover")
    dem = ee.Image("USGS/3DEP/10m")
    slope = ee.Terrain.slope(dem)
    developed = nlcd.gte(21).And(nlcd.lte(24))
    # neighborhood=64 px @ 30m = 1920m search radius — just past our 1500m
    # threshold; default of 256 is wildly more compute than we need
    dev_dist = (developed.fastDistanceTransform(neighborhood=64).sqrt().multiply(30)
                .rename("dev_dist_m"))

    roads = (ee.FeatureCollection("TIGER/2016/Roads")
             .filter(ee.Filter.inList("mtfcc", ["S1100", "S1200"])))

    # Build a small image stack for cheap point sampling
    img = (nlcd.rename("nlcd")
           .addBands(slope.rename("slope_deg"))
           .addBands(dev_dist))

    def eval_feat(feat: ee.Feature) -> ee.Feature:
        pt = feat.geometry()
        box = pt.buffer(CONUS_BUFFER_M).bounds()
        nearby = roads.filterBounds(box)
        road_d = ee.Algorithms.If(
            nearby.size().gt(0),
            pt.distance(nearby.geometry(), maxError=10),
            ee.Number(CONUS_BUFFER_M),
        )
        return feat.set("road_dist_m", road_d)

    feats_list = fc.toList(fc.size()).getInfo()
    n = len(feats_list)
    print(f"[{label}] evaluating {n} points (chunked)...")

    STACK_CHUNK = 200   # image-stack sampling — bounded by GEE compute time
    ROAD_CHUNK = 64     # road-distance per-point — bounded by memory limit

    stack_by_idx: dict[int, dict] = {}
    for start in range(0, n, STACK_CHUNK):
        chunk = ee.FeatureCollection(feats_list[start:start + STACK_CHUNK])
        result = img.sampleRegions(
            collection=chunk, scale=30, geometries=False, tileScale=4,
        ).getInfo()
        for f in result.get("features", []):
            p = f["properties"]
            stack_by_idx[int(p["row_idx"])] = p
        print(f"  stack: {len(stack_by_idx)}/{n}")

    road_by_idx: dict[int, float] = {}
    for start in range(0, n, ROAD_CHUNK):
        chunk = ee.FeatureCollection(feats_list[start:start + ROAD_CHUNK])
        result = chunk.map(eval_feat).getInfo()
        for f in result.get("features", []):
            p = f["properties"]
            road_by_idx[int(p["row_idx"])] = float(p["road_dist_m"])
        print(f"  road: {len(road_by_idx)}/{n}")

    # Combine
    rows = []
    for ridx, p in stack_by_idx.items():
        rows.append({
            "row_idx": ridx,
            "nlcd": int(p["nlcd"]) if p.get("nlcd") is not None else None,
            "slope_deg": float(p["slope_deg"]) if p.get("slope_deg") is not None else None,
            "dev_dist_m": float(p["dev_dist_m"]) if p.get("dev_dist_m") is not None else None,
            "road_dist_m": road_by_idx.get(ridx),
        })
    return pd.DataFrame(rows)


def apply_mask(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["excl_slope"] = df["slope_deg"] > SLOPE_THRESHOLD
    df["excl_ice"] = df["nlcd"] == NLCD_ICE
    df["excl_dev"] = df["dev_dist_m"] > DEV_DIST_THRESHOLD
    df["excl_road"] = df["road_dist_m"] > ROAD_DIST_THRESHOLD
    df["excluded"] = df[["excl_slope", "excl_ice", "excl_dev", "excl_road"]].any(axis=1)
    return df


def report(df: pd.DataFrame, label: str) -> None:
    n = len(df)
    print(f"\n=== {label}: n={n} ===")
    if n == 0:
        return
    for c in ["excl_slope", "excl_ice", "excl_dev", "excl_road"]:
        frac = df[c].mean()
        print(f"  {c:12s}: {df[c].sum():5d} / {n}  ({100*frac:5.2f}%)")
    print(f"  combined OR : {df['excluded'].sum():5d} / {n}  ({100*df['excluded'].mean():5.2f}%)")


def main() -> int:
    if not GCP_PROJECT:
        print("error: GCP_PROJECT must be set", file=sys.stderr)
        return 1
    init_ee()

    states = (ee.FeatureCollection("TIGER/2018/States")
              .filter(ee.Filter.inList("STUSPS", CONUS_STUSPS)))
    conus = states.geometry()

    # 1) Anchor recall check
    anchors = pd.read_csv(ANCHORS_CSV).dropna(subset=["lat", "lng"]).reset_index(drop=True)
    anchor_fc = ee.FeatureCollection([
        ee.Feature(ee.Geometry.Point([float(r["lng"]), float(r["lat"])]),
                   {"row_idx": int(i)})
        for i, r in anchors.iterrows()
    ])
    anchor_df = evaluate_points(anchor_fc, "anchors")
    anchor_df = apply_mask(anchor_df)
    report(anchor_df, "anchors (target: 0 excluded)")

    n_anchor_excl = int(anchor_df["excluded"].sum())
    if n_anchor_excl > 0:
        print("\nAnchors that the proposed mask would exclude:")
        excluded_anchors = anchor_df[anchor_df["excluded"]].copy()
        excluded_anchors = excluded_anchors.merge(
            anchors[["canonical_project_name", "state", "lat", "lng"]],
            left_on="row_idx", right_index=True,
        )
        for _, r in excluded_anchors.iterrows():
            reasons = [c for c in ["excl_slope", "excl_ice", "excl_dev", "excl_road"] if r[c]]
            print(f"  {r['canonical_project_name'][:50]:50s} ({r['state']})  reasons: {reasons}")

    # 2) CONUS reduction estimate
    sample_fc = ee.FeatureCollection.randomPoints(
        region=conus, points=N_CONUS_SAMPLE, seed=SEED,
    )
    # add a row_idx server-side
    sample_list = sample_fc.toList(N_CONUS_SAMPLE)
    sample_fc = ee.FeatureCollection(
        ee.List.sequence(0, N_CONUS_SAMPLE - 1).map(
            lambda i: ee.Feature(sample_list.get(i)).set("row_idx", i)
        )
    )

    sample_df = evaluate_points(sample_fc, "conus_sample")
    sample_df = apply_mask(sample_df)
    report(sample_df, f"random CONUS sample (n={N_CONUS_SAMPLE})")

    print("\n=== summary ===")
    print(f"  anchors excluded: {n_anchor_excl} / {len(anchor_df)}  "
          f"({100 * n_anchor_excl / max(1, len(anchor_df)):.2f}%)")
    print(f"  CONUS pixels excluded (estimate): "
          f"{100 * sample_df['excluded'].mean():.1f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
