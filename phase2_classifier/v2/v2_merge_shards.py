"""Merge per-shard embedding files written by `v2_train.py --num-shards N`.

Reads:  data_us/phase2/v2/emb_<MODEL>_shard0.npy ... emb_<MODEL>_shard{N-1}.npy
        data_us/phase2/v2/v2_embeddings_index_shard0.parquet ... shard{N-1}.parquet

Writes: data_us/phase2/v2/emb_<MODEL>.npy            (concatenated)
        data_us/phase2/v2/v2_embeddings_index.parquet (concatenated, row_idx re-sequenced)

Usage:
    python -m sites_us.phase2_classifier.v2.v2_merge_shards
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
V2_DIR = ROOT / "data_us" / "v2"
EMB_IDX_OUT = V2_DIR / "v2_embeddings_index.parquet"


def main() -> None:
    shard_idx_files = sorted(V2_DIR.glob("v2_embeddings_index_shard*.parquet"))
    if not shard_idx_files:
        print(f"[v2-merge] no shard index files in {V2_DIR}", file=sys.stderr)
        sys.exit(1)
    print(f"[v2-merge] found {len(shard_idx_files)} shard(s)")

    # Discover models from shard 0's emb files
    shard0_embs = sorted(V2_DIR.glob("emb_*_shard0.npy"))
    model_names = [p.name.removeprefix("emb_").removesuffix("_shard0.npy") for p in shard0_embs]
    print(f"[v2-merge] models: {model_names}")

    # Concatenate index parquets, re-sequence row_idx as global running index
    idx_dfs = []
    offset = 0
    for f in shard_idx_files:
        df = pd.read_parquet(f)
        df = df.copy()
        df["row_idx"] = np.arange(offset, offset + len(df))
        idx_dfs.append(df)
        offset += len(df)
    merged_idx = pd.concat(idx_dfs, ignore_index=True)
    merged_idx.to_parquet(EMB_IDX_OUT, index=False)
    print(f"[v2-merge] wrote {len(merged_idx):,} rows -> {EMB_IDX_OUT}")

    # Concatenate embeddings per model in same shard order
    for name in model_names:
        arrs = []
        for f in shard_idx_files:
            shard_id = f.stem.split("shard")[-1]
            emb_path = V2_DIR / f"emb_{name}_shard{shard_id}.npy"
            if not emb_path.exists():
                print(f"[v2-merge] WARN missing {emb_path}", file=sys.stderr)
                continue
            arrs.append(np.load(emb_path))
        merged = np.concatenate(arrs, axis=0)
        out = V2_DIR / f"emb_{name}.npy"
        np.save(out, merged)
        print(f"[v2-merge] wrote {merged.shape} -> {out}")


if __name__ == "__main__":
    main()
