"""Build per-state NAIP tile indices for Stage 2b v2.

Two modes:
  --mode baseline  : earliest available year per state on s3://naip-analytic/
  --mode recent    : latest available year per state (mirrors the existing
                     build_naip_manifest.py behaviour)

For each CONUS state, walks year prefixes ASC (baseline) or DESC (recent),
picking the first year/resolution combo that has both a populated rgbir_cog/
and an index/*.shp shapefile. Downloads the shapefile and parses tile bounds.

Reuses discovery/download/load helpers from phase3_naip.build_naip_manifest
to avoid duplicating S3 plumbing.

Writes:
  data_us/phase3_naip/naip_tile_index_<mode>.parquet
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

SITES_US = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SITES_US))
DATA_US = SITES_US.parent / "data_us"

from phase3_naip.build_naip_manifest import (
    _s3_client, _list_prefix, _has_objects, _find_index_shp,
    detect_naming_convention, download_index, load_one_index,
    CONUS_STATES, RES_PREFERENCE,
)

OUT_DIR = DATA_US / "phase3_naip"


def discover_state_year(s3, state: str, mode: str) -> tuple[int, str] | None:
    """Like phase3_naip.build_naip_manifest.discover_state_year but with mode.
    mode='earliest' iterates years ASC; mode='latest' iterates DESC."""
    years = _list_prefix(s3, f"{state}/")
    years = sorted([int(y) for y in years if y.isdigit() and len(y) == 4],
                   reverse=(mode == "latest"))
    for year in years:
        resolutions = _list_prefix(s3, f"{state}/{year}/")
        ordered = [r for r in RES_PREFERENCE if r in resolutions]
        ordered.extend(r for r in resolutions if r not in ordered)
        for res in ordered:
            if not _has_objects(s3, f"{state}/{year}/{res}/rgbir_cog/"):
                continue
            if _find_index_shp(s3, state, year, res) is None:
                continue
            return year, res
    return None


def build_index_for_state(state: str, mode: str) -> pd.DataFrame | None:
    """Each thread builds its own boto3 client (boto3 clients are not thread-safe)."""
    s3 = _s3_client()
    t0 = time.time()
    sy = discover_state_year(s3, state, mode)
    if sy is None:
        print(f"[{mode}] {state.upper()}: no viable year (rgbir_cog + .shp)", flush=True)
        return None
    year, res = sy
    shp = download_index(s3, state, year, res)
    if shp is None:
        return None
    convention = detect_naming_convention(s3, state, year, res)
    if convention == "unknown":
        print(f"[{mode}] {state.upper()}: unknown filename convention", flush=True)
        return None
    gdf = load_one_index(shp, state, year, res, convention)
    print(f"[{mode}] {state.upper()}: {year} {res} ({convention}) -> "
          f"{len(gdf):,} tiles ({time.time()-t0:.1f}s)", flush=True)
    return pd.DataFrame(gdf.drop(columns=["geometry"]))


def build(mode: str, states: list[str], workers: int) -> pd.DataFrame:
    assert mode in ("baseline", "recent")
    naip_mode = "earliest" if mode == "baseline" else "latest"
    pieces: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(build_index_for_state, s, naip_mode): s for s in states}
        for fut in as_completed(futs):
            piece = fut.result()
            if piece is not None:
                pieces.append(piece)
    if not pieces:
        raise SystemExit("no indices loaded")
    return pd.concat(pieces, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["baseline", "recent"], required=True)
    ap.add_argument("--states", default=None, help="comma-sep; default = all CONUS")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=None,
                    help="default = data_us/phase3_naip/naip_tile_index_<mode>.parquet")
    args = ap.parse_args()

    states = ([s.strip().lower() for s in args.states.split(",")]
              if args.states else CONUS_STATES)
    out = Path(args.out) if args.out else OUT_DIR / f"naip_tile_index_{args.mode}.parquet"

    t0 = time.time()
    print(f"[build] mode={args.mode} states={len(states)} workers={args.workers}",
          flush=True)
    idx = build(args.mode, states, args.workers)
    out.parent.mkdir(parents=True, exist_ok=True)
    idx.to_parquet(out, index=False)
    print(f"[build] wrote {len(idx):,} tiles -> {out}  ({(time.time()-t0)/60:.1f} min)")
    print(f"[build] per-state earliest years:")
    print(idx.groupby("state")["naip_year"].first().sort_index().to_string())


if __name__ == "__main__":
    main()
