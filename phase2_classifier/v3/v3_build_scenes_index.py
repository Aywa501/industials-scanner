"""Per-building NAIP-COG lookup index for v3.

For each building in the v3 manifest, compute its fetch bbox (building_bbox +
BUFFER_M) and find which NAIP COGs from naip_tile_index.parquet intersect.

Each building maps to 1–4 COGs (usually 1; NAIP quads are ~3.75' wide ≈ 5 km
at mid-latitudes, well bigger than our crop). Buildings straddling a tile edge
get multiple URIs and the embed step mosaics them.

Reads:
  data_us/phase2/v3_dataset_manifest.parquet
  data_us/phase3_naip/naip_tile_index.parquet
Writes:
  data_us/phase2/v3_scenes_index.parquet  (one row per building; URIs as a list)
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from shapely import STRtree, box

ROOT = Path(__file__).resolve().parents[3]
DATA_US = ROOT / "data_us"
MANIFEST = Path(os.environ.get("V3_MANIFEST", DATA_US / "phase2" / "v3_dataset_manifest.parquet"))
TILE_INDEX = DATA_US / "phase3_naip" / "naip_tile_index.parquet"
OUT = Path(os.environ.get("V3_SCENES_OUT", DATA_US / "phase2" / "v3_scenes_index.parquet"))

BUFFER_M = 100.0          # context buffer around each building bbox
M_PER_DEG_LAT = 110_540.0
M_PER_DEG_LON_EQ = 111_320.0


def _fetch_bbox(m: pd.DataFrame, buffer_m: float) -> np.ndarray:
    coslat = np.cos(np.radians(m.lat.to_numpy()))
    dlat = buffer_m / M_PER_DEG_LAT
    dlon = buffer_m / (M_PER_DEG_LON_EQ * np.maximum(coslat, 0.1))
    return np.column_stack([
        m.xmin.to_numpy() - dlon,
        m.ymin.to_numpy() - dlat,
        m.xmax.to_numpy() + dlon,
        m.ymax.to_numpy() + dlat,
    ])


def main() -> None:
    print(f"[v3-scenes] loading manifest {MANIFEST.name}...")
    m = pd.read_parquet(MANIFEST)
    print(f"  buildings: {len(m):,}")

    print(f"[v3-scenes] loading NAIP tile index ({TILE_INDEX.name})...")
    idx = pd.read_parquet(TILE_INDEX)
    print(f"  tiles: {len(idx):,}  states: {idx.state.nunique()}")

    print(f"[v3-scenes] spatial join with buffer = {BUFFER_M:.0f} m...")
    fetch_bb = _fetch_bbox(m, BUFFER_M)
    b_boxes = box(fetch_bb[:, 0], fetch_bb[:, 1], fetch_bb[:, 2], fetch_bb[:, 3])
    t_boxes = box(idx.lon_min.values, idx.lat_min.values,
                  idx.lon_max.values, idx.lat_max.values)
    tree = STRtree(t_boxes)
    pairs = tree.query(b_boxes, predicate="intersects")
    b_ix, t_ix = pairs[0], pairs[1]
    print(f"  building↔tile pairs: {len(b_ix):,}")

    # Aggregate to one row per building (list of URIs).
    n = len(m)
    uris: list[list[str]] = [[] for _ in range(n)]
    acq: list[list[str]] = [[] for _ in range(n)]
    year_first: list[int | None] = [None] * n
    state_first: list[str | None] = [None] * n
    res_first: list[str | None] = [None] * n
    uri_arr = idx.tile_uri.to_numpy()
    acq_arr = idx.naip_acq_date.to_numpy()
    year_arr = idx.naip_year.to_numpy()
    state_arr = idx.state.to_numpy()
    res_arr = idx.naip_res.to_numpy()
    for bi, ti in zip(b_ix.tolist(), t_ix.tolist()):
        uris[bi].append(str(uri_arr[ti]))
        acq[bi].append(str(acq_arr[ti]))
        if year_first[bi] is None:
            year_first[bi] = int(year_arr[ti])
            state_first[bi] = str(state_arr[ti])
            res_first[bi] = str(res_arr[ti])

    out = m[["building_id", "lat", "lon",
             "xmin", "xmax", "ymin", "ymax", "approx_area_m2"]].copy()
    out["fetch_xmin"] = fetch_bb[:, 0]
    out["fetch_ymin"] = fetch_bb[:, 1]
    out["fetch_xmax"] = fetch_bb[:, 2]
    out["fetch_ymax"] = fetch_bb[:, 3]
    out["naip_uris"] = uris
    out["naip_acq_dates"] = acq
    out["naip_year"] = year_first
    out["naip_state"] = state_first
    out["naip_res"] = res_first
    out["n_tiles"] = [len(u) for u in uris]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print()
    print(f"[v3-scenes] wrote {len(out):,} rows -> {OUT}")
    print()
    print("=== tile-count distribution per building ===")
    print(out.n_tiles.value_counts().sort_index().to_string())
    print()
    print("=== coverage by year ===")
    print(out.naip_year.value_counts(dropna=False).sort_index().to_string())
    print()
    n_zero = (out.n_tiles == 0).sum()
    print(f"buildings with NO NAIP tile: {n_zero:,} ({n_zero/len(out)*100:.2f}%)")


if __name__ == "__main__":
    main()
