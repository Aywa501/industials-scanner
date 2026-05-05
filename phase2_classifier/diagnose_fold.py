"""Reproduce a single fold's split + train, then surface misclassified chips.

Usage: python diagnose_fold.py [fold_idx]
Default: fold 4 (the weakest one in the v1 CV run).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
DATA_US = ROOT.parent / "data_us"

FILT_PATH = DATA_US / "stage1_filtered_dataset.parquet"
EMB_PATH = DATA_US / "stage1_embeddings.npy"
INDEX_PATH = DATA_US / "stage1_embeddings_index.parquet"

N_FOLDS = 5
EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 256
SEED = 42


def main() -> int:
    fold_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 4

    torch.manual_seed(SEED)
    filt = pd.read_parquet(FILT_PATH)
    filt["year"] = filt.year.astype(int)

    embs = np.load(EMB_PATH).astype(np.float32)
    idx = pd.read_parquet(INDEX_PATH).reset_index(drop=True)
    idx["year"] = idx.year.astype(int)
    idx_with_row = idx[["site_id", "year"]].copy()
    idx_with_row["row_idx"] = idx_with_row.index.astype(int)
    join = filt.merge(idx_with_row, on=["site_id", "year"], how="left")
    join = join[join.row_idx.notna()].copy()
    join["row_idx"] = join.row_idx.astype(int)

    X = torch.from_numpy(embs[join.row_idx.to_numpy()]).float()
    y = (join.label == "industrial").astype(int).to_numpy()

    splits = list(GroupKFold(n_splits=N_FOLDS).split(join, groups=join.site_id.values))
    tr_idx, va_idx = splits[fold_idx]
    print(f"fold {fold_idx}: train {len(tr_idx)} chips / "
          f"{join.iloc[tr_idx].site_id.nunique()} sites  "
          f"val {len(va_idx)} chips / {join.iloc[va_idx].site_id.nunique()} sites")

    val_df = join.iloc[va_idx].reset_index(drop=True)
    print(f"\nval set composition by site_type:")
    print(val_df.groupby(["label", "site_type"]).size().to_string())
    print(f"\nval positives: {(y[va_idx]==1).sum()}  negatives: {(y[va_idx]==0).sum()}")

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    head = nn.Linear(1024, 2).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    y_tr_np = y[tr_idx]
    n_pos = (y_tr_np == 1).sum()
    n_neg = (y_tr_np == 0).sum()
    w = np.where(y_tr_np == 1, 1.0 / n_pos, 1.0 / n_neg)
    sampler = torch.utils.data.WeightedRandomSampler(
        torch.from_numpy(w).double(), num_samples=len(y_tr_np), replacement=True
    )
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X[tr_idx], torch.from_numpy(y_tr_np).long()),
        batch_size=BATCH_SIZE, sampler=sampler,
    )

    best_f1 = -1
    best_state = None
    for ep in range(EPOCHS):
        head.train()
        for xb, yb in loader:
            xb = xb.to(device); yb = yb.to(device)
            loss = F.cross_entropy(head(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        head.eval()
        with torch.no_grad():
            v_prob = F.softmax(head(X[va_idx].to(device)), dim=1)[:, 1].cpu().numpy()
        v_pred = (v_prob > 0.5).astype(int)
        from sklearn.metrics import f1_score
        f1 = f1_score(y[va_idx], v_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state)
    print(f"  best F1 reproduced: {best_f1:.3f}")

    head.eval()
    with torch.no_grad():
        prob = F.softmax(head(X[va_idx].to(device)), dim=1)[:, 1].cpu().numpy()
    pred = (prob > 0.5).astype(int)
    val_df["prob"] = prob
    val_df["pred"] = pred
    val_df["true"] = y[va_idx]
    val_df["correct"] = val_df.pred == val_df.true

    fn = val_df[(val_df.true == 1) & (val_df.pred == 0)].sort_values("prob")
    fp = val_df[(val_df.true == 0) & (val_df.pred == 1)].sort_values("prob", ascending=False)

    print(f"\n=== False Negatives (industrial predicted negative): {len(fn)} ===")
    print(f"  by site_type:\n{fn.site_type.value_counts().to_string()}")
    print(f"  unique sites: {fn.site_id.nunique()}")
    print("\n  top 15 most-confident wrong (lowest prob on real industrial):")
    for _, r in fn.head(15).iterrows():
        print(f"    {r.site_id}  y{r.year}  prob={r.prob:.3f}  "
              f"type={r.site_type}  src={r.source}")

    print(f"\n=== False Positives (negative predicted industrial): {len(fp)} ===")
    print(f"  by site_type:\n{fp.site_type.value_counts().to_string()}")
    print(f"  unique sites: {fp.site_id.nunique()}")
    print("\n  top 15 most-confident wrong (highest prob on real negative):")
    for _, r in fp.head(15).iterrows():
        print(f"    {r.site_id}  y{r.year}  prob={r.prob:.3f}  "
              f"type={r.site_type}  src={r.source}")

    # Per-site recall on val positives.
    pos_val = val_df[val_df.true == 1]
    site_recall = pos_val.groupby("site_id").pred.mean()
    bad_sites = site_recall[site_recall < 0.5].sort_values()
    if len(bad_sites):
        print(f"\n=== sites where model misses majority of positive years ({len(bad_sites)}) ===")
        for sid, r in bad_sites.items():
            n = (pos_val.site_id == sid).sum()
            t = pos_val[pos_val.site_id == sid].iloc[0].site_type
            cn = pos_val[pos_val.site_id == sid].iloc[0].canonical_project_name
            print(f"  {sid}  recall={r:.2f}  n_pos={n}  type={t}  proj={cn}")

    # Save details for downstream.
    out_path = DATA_US / f"stage1_fold{fold_idx}_val_predictions.parquet"
    val_df.to_parquet(out_path, index=False)
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
