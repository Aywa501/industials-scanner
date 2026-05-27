"""Embedding-NN filter for the Stage 1 negative pool.

For each `candidate_negative` chip, compute cosine distance to the nearest
`industrial` (positive) embedding and the nearest `manual_negative` embedding.

- If NN is a positive  -> drop (likely industrial leak).
- Else                 -> keep as filtered negative.

The ~SHORTLIST_N candidates with the *smallest* cosine distance to any positive
get written to a relabel shortlist for the user to confirm in the labeling
webapp before they're discarded.

Inputs:
- data_us/phase1/stage1_embeddings.npy            (N, 1024) float16
- data_us/phase1/stage1_embeddings_index.parquet  site_id, year, label, ...

Outputs:
- data_us/phase1/stage1_filtered_dataset.parquet  filtered training pool
- data_us/phase1/stage1_relabel_shortlist.json    [{site_id, year, ...}] for webapp
- data_us/phase1/stage1_filter_report.json        summary stats
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_US = ROOT.parent / "data_us"

EMB_PATH = DATA_US / "phase1" / "stage1_embeddings.npy"
INDEX_PATH = DATA_US / "phase1" / "stage1_embeddings_index.parquet"
DATASET_PATH = DATA_US / "phase1" / "stage1_dataset.parquet"

OUT_DATASET_PATH = DATA_US / "phase1" / "stage1_filtered_dataset.parquet"
OUT_SHORTLIST_PATH = DATA_US / "phase1" / "stage1_relabel_shortlist.json"
OUT_REPORT_PATH = DATA_US / "phase1" / "stage1_filter_report.json"

SHORTLIST_N = 50


def cosine_nn_distance(query: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each row in query, return (min_distance, argmin) against ref.

    Both are pre-L2-normalized; cosine distance = 1 - dot.
    """
    sims = query @ ref.T  # (Q, R)
    arg = sims.argmax(axis=1)
    best = sims[np.arange(len(query)), arg]
    return 1.0 - best, arg


def main() -> int:
    embs = np.load(EMB_PATH).astype(np.float32)
    idx = pd.read_parquet(INDEX_PATH).reset_index(drop=True)
    idx["row_idx"] = idx.index.astype(int)

    # Refresh labels/site_type/source from the current build_dataset output —
    # the embedding matrix is label-agnostic but the index's bookkeeping
    # columns can go stale across reruns.
    ds = pd.read_parquet(DATASET_PATH)[
        ["site_id", "year", "label", "site_type", "source"]
    ]
    ds["year"] = ds.year.astype(int)
    idx["year"] = idx.year.astype(int)
    idx_dropped = idx.drop(columns=["label", "site_type", "source"])
    merged = idx_dropped.merge(ds, on=["site_id", "year"], how="left")
    n_orphan = merged.label.isna().sum()
    if n_orphan:
        print(f"  dropping {n_orphan} embeddings whose chip is no longer in "
              f"the dataset (e.g. confirmed-industrial random sites)")
        keep = merged.label.notna().to_numpy()
        embs = embs[keep]
        merged = merged[keep].reset_index(drop=True)
        merged["row_idx"] = merged.index.astype(int)
    idx = merged

    print(f"loaded {len(embs)} embeddings × {embs.shape[1]}-dim")
    print(idx.label.value_counts().to_string())

    # L2-normalize once.
    norms = np.linalg.norm(embs, axis=1, keepdims=True).clip(min=1e-12)
    embs = embs / norms

    pos_mask = (idx.label == "industrial").to_numpy()
    cand_mask = (idx.label == "candidate_negative").to_numpy()
    man_mask = (idx.label == "manual_negative").to_numpy()
    print(f"positives: {pos_mask.sum()}  candidates: {cand_mask.sum()}  "
          f"manual_neg: {man_mask.sum()}")

    pos_emb = embs[pos_mask]
    man_emb = embs[man_mask]
    cand_emb = embs[cand_mask]
    cand_rows = idx[cand_mask].reset_index(drop=True)

    d_pos, arg_pos = cosine_nn_distance(cand_emb, pos_emb)
    d_man, arg_man = cosine_nn_distance(cand_emb, man_emb)

    pos_rows = idx[pos_mask].reset_index(drop=True)
    man_rows = idx[man_mask].reset_index(drop=True)

    cand_rows = cand_rows.assign(
        nn_pos_dist=d_pos,
        nn_pos_site=pos_rows.iloc[arg_pos].site_id.values,
        nn_pos_year=pos_rows.iloc[arg_pos].year.values,
        nn_man_dist=d_man,
        nn_man_site=man_rows.iloc[arg_man].site_id.values,
        nn_man_year=man_rows.iloc[arg_man].year.values,
    )

    # Drop if nearer to a positive than to any manual_negative.
    drop_mask = cand_rows.nn_pos_dist < cand_rows.nn_man_dist
    n_drop = int(drop_mask.sum())
    n_keep = int((~drop_mask).sum())
    print(f"  drop (NN is positive): {n_drop}")
    print(f"  keep:                  {n_keep}")

    kept_cands = cand_rows[~drop_mask].copy()
    kept_cands["label"] = "negative"

    # Relabel shortlist: smallest distance-to-positive among ALL candidates,
    # whether dropped or kept (the model thinks they're closest to industrial).
    shortlist = (
        cand_rows.sort_values("nn_pos_dist", ascending=True)
        .head(SHORTLIST_N)
        .copy()
    )
    shortlist["was_dropped"] = shortlist.index.isin(cand_rows[drop_mask].index)

    # ---- Assemble final filtered training pool.
    pos_rows_out = idx[pos_mask].copy()
    pos_rows_out["label"] = "industrial"
    man_rows_out = idx[man_mask].copy()
    man_rows_out["label"] = "negative"
    kept_out = kept_cands.copy()
    kept_out["label"] = "negative"

    # Re-attach tile_uri from the original dataset.
    base = pd.read_parquet(DATASET_PATH)[
        ["site_id", "year", "tile_uri", "site_type", "source"]
    ]

    def attach(df):
        return df.merge(base, on=["site_id", "year"], how="left",
                        suffixes=("", "_b"))

    final = pd.concat([
        attach(pos_rows_out[["site_id", "year"]].assign(label="industrial")),
        attach(man_rows_out[["site_id", "year"]].assign(label="negative")),
        attach(kept_out[["site_id", "year"]].assign(label="negative")),
    ], ignore_index=True)
    final = final.drop_duplicates(["site_id", "year"]).reset_index(drop=True)

    print(f"\nfinal filtered pool: {len(final)}")
    print(final.label.value_counts().to_string())
    print()
    print("by site_type:")
    print(final.groupby(["label", "site_type"]).size().to_string())

    OUT_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    final.to_parquet(OUT_DATASET_PATH, index=False)
    print(f"\nwrote {OUT_DATASET_PATH}")

    sl = []
    for _, r in shortlist.iterrows():
        sl.append({
            "site_id": str(r.site_id),
            "year": int(r.year),
            "nn_pos_site": str(r.nn_pos_site),
            "nn_pos_year": int(r.nn_pos_year),
            "nn_pos_dist": float(r.nn_pos_dist),
            "nn_man_dist": float(r.nn_man_dist),
            "was_dropped": bool(r.was_dropped),
        })
    OUT_SHORTLIST_PATH.write_text(json.dumps(sl, indent=2))
    print(f"wrote {OUT_SHORTLIST_PATH}  ({len(sl)} chips for review)")

    report = {
        "n_positives": int(pos_mask.sum()),
        "n_candidates": int(cand_mask.sum()),
        "n_manual_negatives": int(man_mask.sum()),
        "n_dropped_as_leak": n_drop,
        "n_kept": n_keep,
        "candidate_dist_to_pos": {
            "min": float(cand_rows.nn_pos_dist.min()),
            "p10": float(cand_rows.nn_pos_dist.quantile(0.10)),
            "p50": float(cand_rows.nn_pos_dist.median()),
            "p90": float(cand_rows.nn_pos_dist.quantile(0.90)),
            "max": float(cand_rows.nn_pos_dist.max()),
        },
        "candidate_dist_to_man": {
            "min": float(cand_rows.nn_man_dist.min()),
            "p10": float(cand_rows.nn_man_dist.quantile(0.10)),
            "p50": float(cand_rows.nn_man_dist.median()),
            "p90": float(cand_rows.nn_man_dist.quantile(0.90)),
            "max": float(cand_rows.nn_man_dist.max()),
        },
        "final_counts": final.label.value_counts().to_dict(),
    }
    OUT_REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"wrote {OUT_REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
