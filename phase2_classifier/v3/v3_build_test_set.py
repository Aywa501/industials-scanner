"""Build a per-building test set for v3, disjoint from training.

Mirrors v3_build_dataset.py's OSM filter+sample logic, but:
  - Excludes every ovt_id present in v3_dataset_manifest.parquet
  - Excludes buildings within 500 m of any training building (prevents
    same-site-different-building leakage; test points should be far from
    every training point)
  - Uses TEST_SEED=44 to disambiguate from training (which uses 42/43)
  - Samples N=1000 industrial + N=1000 non-industrial

Schema matches training so v3_build_scenes_index.py runs unmodified
(V3_MANIFEST + V3_SCENES_OUT env vars point it at the test files).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

ROOT = Path(__file__).resolve().parents[3]
DATA_US = ROOT / "data_us"

OVERTURE = DATA_US / "external" / "overture_industrial_conus_2025_aligned.parquet"
TRAIN_MANIFEST = DATA_US / "phase2" / "v3_dataset_manifest.parquet"
CONUS_GEOJSON = DATA_US / "phase3_scan" / "cache" / "us-states.geojson"
OUT = DATA_US / "phase2" / "v3_test_set_manifest.parquet"

NON_CONUS_STATES = {"Alaska", "Hawaii", "Puerto Rico"}
OSM_INDUSTRIAL_CLASSES = {"industrial", "warehouse", "hangar"}
# Mirrors v3_build_dataset.py — must stay in sync so the test set evaluates the
# same class distribution the model trained on (incl. hard negatives).
OSM_NEG_CLASSES = {
    "commercial", "retail", "office", "parking",
    "residential", "house", "detached", "terrace", "apartments",
    "semidetached_house", "dormitory",
    "school", "university", "college", "kindergarten", "library",
    "hospital",
    "church", "cathedral", "mosque", "synagogue", "religious",
    "hotel", "stadium", "grandstand", "fire_station", "civic", "government",
    "public", "post_office", "greenhouse", "farm_auxiliary", "barn",
}

AREA_FLOOR_M2 = 3000.0
N_PER_CLASS = 1000
TRAIN_EXCLUDE_M = 500            # exclude test candidates within this of any training building
EARTH_R = 6_371_000.0
TEST_SEED = 44


def short_id(prefix: str, *parts) -> str:
    h = hashlib.md5("|".join(str(p) for p in parts).encode()).hexdigest()[:10]
    return f"{prefix}_{h}"


def _osm_only(bldg: pd.DataFrame) -> pd.DataFrame:
    has_osm = bldg["source_datasets"].apply(
        lambda a: a is not None and "OpenStreetMap" in list(a)
    )
    return bldg[has_osm].reset_index(drop=True)


def _row_dict(r, *, class_id: int, source: str) -> dict:
    return dict(
        building_id=short_id(source[:4], r.id),
        ovt_id=r.id,
        class_id=class_id,
        source=source,
        weight=1.0,
        lat=float(r.lat), lon=float(r.lon),
        xmin=float(r.xmin), xmax=float(r.xmax),
        ymin=float(r.ymin), ymax=float(r.ymax),
        approx_area_m2=float(r.approx_area_m2),
        ovt_class=r.ovt_class,
        ovt_subtype=r.subtype,
        ovt_name=r.name,
        site_id=None,
    )


def _load_conus_polygon():
    import json
    from shapely.geometry import shape
    from shapely.ops import unary_union
    fc = json.loads(CONUS_GEOJSON.read_text())
    geoms = [shape(f["geometry"]) for f in fc["features"]
             if f["properties"].get("name") not in NON_CONUS_STATES]
    return unary_union(geoms)


def _exclude_near_training(cands: pd.DataFrame, train_latlon: np.ndarray) -> pd.DataFrame:
    """Drop candidates within TRAIN_EXCLUDE_M of any training building."""
    if len(cands) == 0:
        return cands
    train_rad = np.radians(train_latlon)
    tree = BallTree(train_rad, metric="haversine")
    cand_rad = np.radians(cands[["lat", "lon"]].to_numpy())
    dist, _ = tree.query(cand_rad, k=1)
    keep = (dist[:, 0] * EARTH_R) >= TRAIN_EXCLUDE_M
    return cands[keep].reset_index(drop=True)


def _sample(cands: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(cands) <= n:
        return cands
    return cands.sample(n=n, random_state=seed).reset_index(drop=True)


def main() -> None:
    print(f"[v3-test] loading training manifest {TRAIN_MANIFEST.name}...")
    train = pd.read_parquet(TRAIN_MANIFEST, columns=["ovt_id", "lat", "lon"])
    train_ovt_ids = set(train["ovt_id"].dropna().astype(str).tolist())
    train_latlon = train[["lat", "lon"]].to_numpy()
    print(f"  training: {len(train):,} rows, {len(train_ovt_ids):,} unique ovt_ids")

    print(f"[v3-test] loading Overture ({OVERTURE.name})...")
    bldg = pd.read_parquet(
        OVERTURE,
        columns=["id", "lon", "lat", "xmin", "xmax", "ymin", "ymax",
                 "approx_area_m2", "class", "subtype", "name", "source_datasets"],
    )
    bldg = bldg.rename(columns={"class": "ovt_class"})
    bldg = bldg[bldg["approx_area_m2"] >= AREA_FLOOR_M2].reset_index(drop=True)
    print(f"  >= {AREA_FLOOR_M2:.0f} m^2: {len(bldg):,}")

    print("[v3-test] CONUS land filter...")
    from shapely import STRtree
    from shapely.geometry import Point
    conus = _load_conus_polygon()
    pts = [Point(lo, la) for la, lo in zip(bldg["lat"].to_numpy(), bldg["lon"].to_numpy())]
    inside_ix = STRtree([conus]).query(pts, predicate="intersects")[0]
    keep = np.zeros(len(bldg), dtype=bool); keep[inside_ix] = True
    bldg = bldg[keep].reset_index(drop=True)
    print(f"  after CONUS: {len(bldg):,}")

    bldg = _osm_only(bldg)
    print(f"  after OSM-source: {len(bldg):,}")
    bldg = bldg[~bldg["id"].astype(str).isin(train_ovt_ids)].reset_index(drop=True)
    print(f"  after training-ovt_id exclude: {len(bldg):,}")

    # Industrial candidates. Disjointness is by ovt_id only; the training set
    # used 500m dedup on the same pool, so any leftover industrial building is
    # within 500m of a training one — but it's a different building, which is
    # also what CONUS scan will see in deployment.
    ind_mask = (bldg["ovt_class"].isin(OSM_INDUSTRIAL_CLASSES)
                | (bldg["subtype"] == "industrial"))
    ind = bldg[ind_mask].reset_index(drop=True)
    print(f"[v3-test] industrial candidates: {len(ind):,}")
    ind = _sample(ind, N_PER_CLASS, TEST_SEED)
    print(f"  sampled: {len(ind):,}")

    # Non-industrial candidates.
    neg = bldg[bldg["ovt_class"].isin(OSM_NEG_CLASSES)].reset_index(drop=True)
    print(f"[v3-test] non-industrial candidates: {len(neg):,}")
    neg = _sample(neg, N_PER_CLASS, TEST_SEED + 1)
    print(f"  sampled: {len(neg):,}")

    rows = ([_row_dict(r, class_id=2, source="test_osm_industrial")
             for r in ind.itertuples(index=False)]
            + [_row_dict(r, class_id=0, source="test_osm_neg")
               for r in neg.itertuples(index=False)])
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["building_id"]).reset_index(drop=True)
    df["split"] = "test"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print()
    print(f"[v3-test] wrote {len(df):,} rows -> {OUT}")
    print()
    print(df.groupby(["class_id", "source"]).size().to_string())


if __name__ == "__main__":
    main()
