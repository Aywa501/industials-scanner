"""Aggregate per-MGRS prob parquets into clustered candidate sites.

Reads:
  data_us/phase3_results/*.parquet         (per-tile probs)
Writes:
  data_us/phase3_candidates.parquet        (clustered candidate sites)

Pipeline:
  1. Concat all per-MGRS result parquets
  2. Filter prob >= --min-prob (default 0.7)
  3. Project to EPSG:5070 metres
  4. DBSCAN with eps=2km, min_samples=2 (single tiles drop out — they're the
     noisiest fraction; real industrial sites span ≥2 tiles at 1.68km stride)
  5. Per cluster: centroid, max_prob, mean_prob, n_tiles, area-bbox

Usage:
    python -m phase3_scan.aggregate
    python -m phase3_scan.aggregate --min-prob 0.5 --eps-m 2500
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyproj
from sklearn.cluster import DBSCAN

ROOT = Path(__file__).resolve().parents[1]
DATA_US = ROOT.parent / "data_us"
RESULTS_DIR = DATA_US / "phase3_results"
CANDIDATES_PATH = DATA_US / "phase3_candidates.parquet"


def _load_all(min_prob: float) -> pd.DataFrame:
    parts = sorted(p for p in RESULTS_DIR.glob("*.parquet")
                   if not p.stem.endswith("_emb"))
    if not parts:
        raise SystemExit(f"no result parquets in {RESULTS_DIR}")
    print(f"[agg] reading {len(parts)} shard parquets")
    frames = []
    for p in parts:
        df = pd.read_parquet(p, columns=["tile_id", "lon", "lat", "prob"])
        df = df[df.prob >= min_prob]
        if not df.empty:
            df["mgrs_tile"] = p.stem
            frames.append(df)
    if not frames:
        raise SystemExit(f"no tiles with prob >= {min_prob}")
    df = pd.concat(frames, ignore_index=True)
    print(f"[agg] {len(df):,} tiles with prob >= {min_prob}")
    return df


def _cluster(df: pd.DataFrame, eps_m: float, min_samples: int) -> pd.DataFrame:
    to_5070 = pyproj.Transformer.from_crs(4326, 5070, always_xy=True).transform
    x, y = to_5070(df.lon.to_numpy(), df.lat.to_numpy())
    coords = np.column_stack([x, y])
    print(f"[agg] DBSCAN eps={eps_m}m, min_samples={min_samples}")
    labels = DBSCAN(eps=eps_m, min_samples=min_samples).fit_predict(coords)
    df = df.assign(cluster=labels, x_5070=x, y_5070=y)
    return df


def _summarize(df: pd.DataFrame) -> pd.DataFrame:
    clustered = df[df.cluster >= 0]
    if clustered.empty:
        raise SystemExit("DBSCAN found no clusters — try --min-samples 1 or lower --min-prob")
    g = clustered.groupby("cluster", sort=False)
    summary = pd.DataFrame({
        "n_tiles": g.size(),
        "max_prob": g.prob.max(),
        "mean_prob": g.prob.mean(),
        "lat": g.lat.mean(),
        "lon": g.lon.mean(),
        "lat_min": g.lat.min(),
        "lat_max": g.lat.max(),
        "lon_min": g.lon.min(),
        "lon_max": g.lon.max(),
    }).reset_index(drop=True)

    summary["span_m"] = np.hypot(
        (summary.lon_max - summary.lon_min) * 111_000 *
        np.cos(np.deg2rad(summary.lat)),
        (summary.lat_max - summary.lat_min) * 111_000,
    ).round(0).astype(int)

    summary["score"] = summary.max_prob * np.log1p(summary.n_tiles)
    summary = summary.sort_values("score", ascending=False).reset_index(drop=True)
    summary.insert(0, "candidate_id", [f"c_{i:07d}" for i in range(len(summary))])
    summary["maps_url"] = (
        "https://maps.google.com/?q=" +
        summary.lat.round(6).astype(str) + "," + summary.lon.round(6).astype(str)
    )
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-prob", type=float, default=0.7)
    ap.add_argument("--eps-m", type=float, default=2000.0)
    ap.add_argument("--min-samples", type=int, default=2)
    args = ap.parse_args()

    df = _load_all(args.min_prob)
    df = _cluster(df, args.eps_m, args.min_samples)
    summary = _summarize(df)

    CANDIDATES_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_parquet(CANDIDATES_PATH, index=False)

    n_singletons = int((df.cluster < 0).sum())
    print(f"\n[agg] {len(summary):,} clusters from {len(df) - n_singletons:,} clustered tiles "
          f"({n_singletons:,} singletons dropped)")
    print(f"[agg] wrote → {CANDIDATES_PATH}")
    print("\n[agg] top 20 candidate sites:")
    cols = ["candidate_id", "max_prob", "n_tiles", "lat", "lon", "span_m", "maps_url"]
    print(summary[cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
