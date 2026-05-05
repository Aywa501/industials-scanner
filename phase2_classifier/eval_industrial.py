"""Eval Stage 1 on the held-out wild set + surface high-confidence FPs.

Wild set = random-CONUS chips that were dropped from training by the embedding
NN-filter (judged "likely industrial leak"). The model has never seen them.
Running inference here gives us:
  (a) a recall sanity check on training data,
  (b) top-K highest-confidence predictions on the wild set, with lat/lng for
      visual category bucketing (quarry / solar / dense suburban / parking /
      data center / mine / landfill / airport / agri-processing / other).

Outputs:
- data_us/stage1_eval_report.json   summary
- data_us/stage1_wild_topk.csv      top-K wild-set predictions for review
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
DATA_US = ROOT.parent / "data_us"

FILT_PATH = DATA_US / "stage1_filtered_dataset.parquet"
DATASET_PATH = DATA_US / "stage1_dataset.parquet"
EMB_PATH = DATA_US / "stage1_embeddings.npy"
INDEX_PATH = DATA_US / "stage1_embeddings_index.parquet"
MODEL_PATH = DATA_US / "stage1_industrial_v1.pt"

OUT_REPORT = DATA_US / "stage1_eval_report.json"
OUT_TOPK = DATA_US / "stage1_wild_topk.csv"

TOPK = 100


def main() -> int:
    embs = np.load(EMB_PATH).astype(np.float32)
    idx = pd.read_parquet(INDEX_PATH).reset_index(drop=True)
    idx["year"] = idx.year.astype(int)

    filt = pd.read_parquet(FILT_PATH)
    filt["year"] = filt.year.astype(int)
    full = pd.read_parquet(DATASET_PATH)
    full["year"] = full.year.astype(int)

    train_keys = set(zip(filt.site_id, filt.year))

    # Manifest → lat/lng for surfaced chips.
    import os
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    bucket = os.getenv("GCS_BUCKET", "")
    manifest = pd.read_parquet(f"gs://{bucket}/manifest/s2_chip_manifest.parquet")
    coords = manifest.drop_duplicates("site_id").set_index("site_id")[
        ["lat", "lng", "state", "canonical_project_name"]
    ]

    # Load model.
    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    head = nn.Linear(1024, 2)
    head.load_state_dict(ckpt["state_dict"])
    head.eval()

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    head.to(device)

    # Score all 5938 embedded chips.
    X = torch.from_numpy(embs).float().to(device)
    with torch.no_grad():
        prob = F.softmax(head(X), dim=1)[:, 1].cpu().numpy()
    idx["prob"] = prob

    # ---- Sanity: training-set metrics.
    join = filt.merge(idx[["site_id", "year", "prob"]], on=["site_id", "year"], how="left")
    y_tr = (join.label == "industrial").astype(int).to_numpy()
    p_tr = join.prob.to_numpy()
    pred_tr = (p_tr > 0.5).astype(int)
    train_metrics = {
        "n": int(len(join)),
        "f1@0.5": float(f1_score(y_tr, pred_tr, zero_division=0)),
        "auc": float(roc_auc_score(y_tr, p_tr)),
    }
    p, r, t = precision_recall_curve(y_tr, p_tr)
    for prec_t in (0.5, 0.7, 0.9, 0.95):
        m = p >= prec_t
        train_metrics[f"recall@p={prec_t:.2f}"] = float(r[m].max()) if m.any() else 0.0
    print(f"train (5523) — F1: {train_metrics['f1@0.5']:.3f}  "
          f"AUC: {train_metrics['auc']:.3f}  "
          f"R@p.9: {train_metrics['recall@p=0.90']:.3f}")

    # ---- Wild set: chips in stage1_dataset but NOT in stage1_filtered_dataset
    # AND from random-CONUS (the NN-filter-dropped chips).
    wild = idx.copy()
    wild["in_train"] = wild.apply(
        lambda r: (r.site_id, int(r.year)) in train_keys, axis=1
    )
    wild = wild[~wild.in_train].copy()
    # Keep only random-CONUS (n_*) — these are the unbiased wild set.
    wild = wild[wild.site_id.str.startswith("n_")].copy()
    print(f"\nwild set: {len(wild)} chips, {wild.site_id.nunique()} sites")
    print(f"  prob distribution: min={wild.prob.min():.3f}  "
          f"p50={np.median(wild.prob):.3f}  p90={np.quantile(wild.prob, .9):.3f}  "
          f"max={wild.prob.max():.3f}")
    print(f"  predicted positive (prob>0.5): {(wild.prob > 0.5).sum()} "
          f"({(wild.prob > 0.5).mean()*100:.1f}%)")
    print(f"  predicted positive (prob>0.9): {(wild.prob > 0.9).sum()} "
          f"({(wild.prob > 0.9).mean()*100:.1f}%)")

    wild = wild.merge(coords, left_on="site_id", right_index=True, how="left")

    # Drop sites the user already confirmed as actually-industrial in the
    # relabel-shortlist pass — they're true positives we dropped from training,
    # not FPs.
    confirmed_positive_random = ["n_3e277b325c", "n_9be9f3b756"]
    wild_fp = wild[~wild.site_id.isin(confirmed_positive_random)].copy()

    # Per-site dedup: keep the year with the highest prob, since labeling is
    # site-level for FP categorization.
    site_top = (wild_fp.sort_values("prob", ascending=False)
                .drop_duplicates("site_id")
                .head(TOPK)
                .copy())
    site_top["maps_url"] = site_top.apply(
        lambda r: f"https://www.google.com/maps/search/?api=1&query={r.lat:.5f},{r.lng:.5f}",
        axis=1,
    )
    out_cols = ["site_id", "year", "prob", "lat", "lng", "state", "maps_url"]
    site_top[out_cols].to_csv(OUT_TOPK, index=False)
    print(f"\nwrote {OUT_TOPK}  (top {TOPK} wild SITES, deduped)")
    print(f"\ntop 25 highest-confidence wild predictions (one per site):")
    for _, r in site_top.head(25).iterrows():
        print(f"  {r.site_id}  best_year={r.year}  p={r.prob:.3f}  "
              f"{r.lat:.4f},{r.lng:.4f}")

    report = {
        "train": train_metrics,
        "wild": {
            "n_chips": int(len(wild)),
            "n_sites": int(wild.site_id.nunique()),
            "prob_min": float(wild.prob.min()),
            "prob_p50": float(np.median(wild.prob)),
            "prob_p90": float(np.quantile(wild.prob, 0.9)),
            "prob_max": float(wild.prob.max()),
            "n_predicted_positive_at_0.5": int((wild.prob > 0.5).sum()),
            "n_predicted_positive_at_0.9": int((wild.prob > 0.9).sum()),
        },
    }
    OUT_REPORT.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {OUT_REPORT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
