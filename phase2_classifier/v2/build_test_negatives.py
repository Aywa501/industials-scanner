"""Build a small held-out OSM-categorical-negatives test manifest.

Mirrors the osm_neg row construction from v2_build_dataset.py exactly
(same Overture file, same OSM_NEG_CLASSES, same area/source filters,
same tile_id hash) and samples N rows DISJOINT from the v2 training set.

Why this exists:
  The v2 hand_not_a_site test split is rural / random-CONUS imagery,
  which the Stage-3 v2 building pre-filter rejects for free at deploy
  time. The probes were trained against OSM-tagged confidently
  non-industrial *buildings* (residential / school / hospital / church
  / etc.) — that's the relevant negative distribution for grading.

Output:
  data_us/phase2/test_neg_v2_manifest.parquet (schema-compatible with v2 manifest)
"""
from __future__ import annotations

import math
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse the exact same constants from v2_build_dataset so we don't drift.
from phase2_classifier.v2.v2_build_dataset import (
    OSM_NEG_CLASSES, _osm_filter, short_id, DATA_YEAR,
)

ROOT = Path(__file__).resolve().parents[3]
DATA_US = ROOT / "data_us"
OVERTURE = DATA_US / "external" / "overture_industrial_conus_2025_aligned.parquet"
TRAIN_MANIFEST = DATA_US / "phase2" / "v2_dataset_manifest.parquet"
OUT = DATA_US / "phase2" / "test_neg_v2_manifest.parquet"

N_TEST = 1000
TEST_SEED = 44   # train used seed=43 inside osm_neg_rows; use a different one


def main() -> None:
    print(f"[test-neg] reading Overture {OVERTURE}")
    bldg = pd.read_parquet(OVERTURE,
                           columns=["lon", "lat", "approx_area_m2", "class",
                                    "source_datasets"])

    # Same filter chain as v2_build_dataset.osm_neg_rows
    bldg = bldg[bldg["class"].isin(OSM_NEG_CLASSES) & (bldg["approx_area_m2"] >= 1000)]
    print(f"  after class+area filter: {len(bldg):,}")
    bldg = _osm_filter(bldg)
    print(f"  after OSM-source filter: {len(bldg):,}")

    # Identify training tile_ids so we sample disjoint
    train = pd.read_parquet(TRAIN_MANIFEST)
    train_neg_ids = set(train[train.source == "osm_neg"].tile_id.unique())
    print(f"  training osm_neg tile_ids: {len(train_neg_ids):,}")

    # Compute prospective tile_id for every candidate row
    bldg = bldg.reset_index(drop=True)
    bldg["tile_id"] = [short_id("osmn", la, lo, DATA_YEAR)
                       for la, lo in zip(bldg.lat.values, bldg.lon.values)]

    # Drop rows whose tile_id is in the training set
    fresh = bldg[~bldg.tile_id.isin(train_neg_ids)].reset_index(drop=True)
    print(f"  fresh (disjoint from train): {len(fresh):,}")

    if len(fresh) < N_TEST:
        raise RuntimeError(f"only {len(fresh)} fresh rows available; need {N_TEST}")

    sample = fresh.sample(n=N_TEST, random_state=TEST_SEED).reset_index(drop=True)
    print(f"  sampled: {len(sample):,}")

    rows = []
    for r in sample.itertuples(index=False):
        rows.append(dict(
            tile_id=r.tile_id,
            class_id=0,
            source="osm_neg_test",
            weight=1.0,
            lat=float(r.lat), lon=float(r.lon),
            target_year=DATA_YEAR,
            offset_dx_m=0.0, offset_dy_m=0.0,
            tile_uri=None,
            site_id=None,
            is_inferred=False,
            split="test",
        ))
    out = pd.DataFrame(rows)
    out.to_parquet(OUT, index=False)
    print(f"[test-neg] wrote {len(out):,} rows -> {OUT}")
    print(f"  class distribution: {out['class'].value_counts().to_dict() if 'class' in out else out.class_id.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
