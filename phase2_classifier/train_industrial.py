"""Train Stage 1 industrial-vs-not linear probe on DINOv3 SAT-493M features.

5-fold CV by site_id (no chip leakage). For each fold: train a fresh
Linear(1024, 2) head with class-balanced sampling on the cached embeddings.
Reports per-fold + aggregated F1 / AUC / recall@target-precision. Then
retrains a final model on the full filtered pool and saves it.

Inputs:
- data_us/stage1_filtered_dataset.parquet  filtered training pool
- data_us/stage1_embeddings.npy            (N_all, 1024) float16
- data_us/stage1_embeddings_index.parquet  site_id, year, label, ...

Outputs:
- data_us/stage1_industrial_v1.pt          {'state_dict': ..., 'meta': ...}
- data_us/stage1_train_report.json         per-fold + aggregate metrics
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
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
DATA_US = ROOT.parent / "data_us"

FILT_PATH = DATA_US / "stage1_filtered_dataset.parquet"
EMB_PATH = DATA_US / "stage1_embeddings.npy"
INDEX_PATH = DATA_US / "stage1_embeddings_index.parquet"

OUT_MODEL = DATA_US / "stage1_industrial_v1.pt"
OUT_REPORT = DATA_US / "stage1_train_report.json"

N_FOLDS = 5
EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 256
EARLY_STOP_PATIENCE = 5
SEED = 42

POS = 1  # class index for "industrial"
NEG = 0  # class index for "negative"


def make_split(filt: pd.DataFrame, n_folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """GroupKFold by site_id so chips of the same site never split across folds."""
    gkf = GroupKFold(n_splits=n_folds)
    return list(gkf.split(filt, groups=filt.site_id.values))


def class_balanced_sampler(y: np.ndarray) -> torch.utils.data.WeightedRandomSampler:
    n_pos = (y == POS).sum()
    n_neg = (y == NEG).sum()
    w = np.where(y == POS, 1.0 / max(n_pos, 1), 1.0 / max(n_neg, 1))
    return torch.utils.data.WeightedRandomSampler(
        torch.from_numpy(w).double(), num_samples=len(y), replacement=True
    )


def train_one_fold(X_tr, y_tr, X_va, y_va, device) -> tuple[nn.Linear, dict]:
    head = nn.Linear(1024, 2).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    sampler = class_balanced_sampler(y_tr.numpy())
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_tr, y_tr),
        batch_size=BATCH_SIZE, sampler=sampler,
    )

    best_f1 = -1.0
    best_state = None
    best_epoch = 0
    no_improve = 0
    history = []

    for ep in range(EPOCHS):
        head.train()
        for xb, yb in loader:
            xb = xb.to(device); yb = yb.to(device)
            logits = head(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()

        head.eval()
        with torch.no_grad():
            v_logits = head(X_va.to(device))
            v_prob = F.softmax(v_logits, dim=1)[:, POS].cpu().numpy()
        v_pred = (v_prob > 0.5).astype(int)
        f1 = f1_score(y_va.numpy(), v_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_va.numpy(), v_prob)
        except ValueError:
            auc = float("nan")
        history.append({"epoch": ep, "val_f1": float(f1), "val_auc": float(auc)})
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            best_epoch = ep
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PATIENCE:
                break

    head.load_state_dict(best_state)
    return head, {"best_epoch": best_epoch, "best_val_f1": best_f1, "history": history}


def evaluate(head: nn.Linear, X: torch.Tensor, y: np.ndarray, device) -> dict:
    head.eval()
    with torch.no_grad():
        prob = F.softmax(head(X.to(device)), dim=1)[:, POS].cpu().numpy()
    pred = (prob > 0.5).astype(int)

    out = {
        "f1@0.5": float(f1_score(y, pred, zero_division=0)),
        "auc": float(roc_auc_score(y, prob)) if len(set(y)) > 1 else float("nan"),
    }

    # Recall at fixed precision targets — Stage 1 is recall-first.
    p, r, t = precision_recall_curve(y, prob)
    for prec_target in (0.50, 0.70, 0.90):
        mask = p >= prec_target
        if mask.any():
            best_r = r[mask].max()
            out[f"recall@p={prec_target:.2f}"] = float(best_r)
        else:
            out[f"recall@p={prec_target:.2f}"] = 0.0

    # Threshold that hits 0.95 recall.
    target_recall = 0.95
    valid = r >= target_recall
    if valid.any():
        # threshold array t aligns with p[:-1], r[:-1].
        idx = np.where(valid[:-1])[0]
        if len(idx):
            j = idx[np.argmax(p[:-1][idx])]
            out[f"prec@r={target_recall:.2f}"] = float(p[j])
            out[f"thresh@r={target_recall:.2f}"] = float(t[j])
    return out


def main() -> int:
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    filt = pd.read_parquet(FILT_PATH)
    filt["year"] = filt.year.astype(int)
    print(f"filtered pool: {len(filt)} chips, {filt.site_id.nunique()} sites")
    print(filt.label.value_counts().to_string())

    embs = np.load(EMB_PATH).astype(np.float32)
    idx = pd.read_parquet(INDEX_PATH).reset_index(drop=True)
    idx["year"] = idx.year.astype(int)
    print(f"embeddings: {len(embs)} × {embs.shape[1]}")

    # Join filtered pool to embedding rows.
    idx_with_row = idx[["site_id", "year"]].copy()
    idx_with_row["row_idx"] = idx_with_row.index.astype(int)
    join = filt.merge(idx_with_row, on=["site_id", "year"], how="left")
    if join.row_idx.isna().any():
        n_miss = int(join.row_idx.isna().sum())
        print(f"  warning: {n_miss} chips in filt have no embedding row; dropping")
        join = join[join.row_idx.notna()].copy()
    join["row_idx"] = join.row_idx.astype(int)

    X_all = torch.from_numpy(embs[join.row_idx.to_numpy()]).float()
    y_all = (join.label == "industrial").astype(int).to_numpy()
    sites = join.site_id.to_numpy()
    print(f"  X: {X_all.shape}  y_pos: {(y_all==POS).sum()}  y_neg: {(y_all==NEG).sum()}")

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"device: {device}")

    splits = make_split(join, N_FOLDS)
    fold_metrics = []
    for k, (tr_idx, va_idx) in enumerate(splits):
        X_tr = X_all[tr_idx]
        y_tr = torch.from_numpy(y_all[tr_idx]).long()
        X_va = X_all[va_idx]
        y_va_np = y_all[va_idx]
        n_tr_sites = len(set(sites[tr_idx]))
        n_va_sites = len(set(sites[va_idx]))
        print(f"\nfold {k}: train {len(tr_idx)} chips / {n_tr_sites} sites  "
              f"val {len(va_idx)} chips / {n_va_sites} sites")

        head, fit_info = train_one_fold(X_tr, y_tr, X_va,
                                        torch.from_numpy(y_va_np).long(), device)
        eval_metrics = evaluate(head, X_va, y_va_np, device)
        print(f"  best_epoch={fit_info['best_epoch']}  "
              f"f1={eval_metrics['f1@0.5']:.3f}  auc={eval_metrics['auc']:.3f}  "
              f"recall@p=0.90={eval_metrics['recall@p=0.90']:.3f}")
        fold_metrics.append({
            "fold": k,
            "n_train": int(len(tr_idx)),
            "n_val": int(len(va_idx)),
            **fit_info,
            **eval_metrics,
        })

    # Aggregate.
    agg = {
        "f1_mean": float(np.mean([m["f1@0.5"] for m in fold_metrics])),
        "f1_std": float(np.std([m["f1@0.5"] for m in fold_metrics])),
        "auc_mean": float(np.mean([m["auc"] for m in fold_metrics])),
        "auc_std": float(np.std([m["auc"] for m in fold_metrics])),
        "recall_at_p90_mean": float(
            np.mean([m["recall@p=0.90"] for m in fold_metrics])
        ),
    }
    print(f"\nCV: F1={agg['f1_mean']:.3f}±{agg['f1_std']:.3f}  "
          f"AUC={agg['auc_mean']:.3f}±{agg['auc_std']:.3f}  "
          f"recall@p=0.90={agg['recall_at_p90_mean']:.3f}")

    # Final model on all data.
    print("\nfitting final model on all data")
    final_head = nn.Linear(1024, 2).to(device)
    opt = torch.optim.AdamW(final_head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sampler = class_balanced_sampler(y_all)
    y_all_t = torch.from_numpy(y_all).long()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_all, y_all_t),
        batch_size=BATCH_SIZE, sampler=sampler,
    )
    median_best = int(np.median([m["best_epoch"] for m in fold_metrics])) + 1
    n_epochs_final = max(median_best, 5)
    print(f"  fitting for {n_epochs_final} epochs (median best across folds)")
    final_head.train()
    for ep in range(n_epochs_final):
        for xb, yb in loader:
            xb = xb.to(device); yb = yb.to(device)
            loss = F.cross_entropy(final_head(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()

    torch.save({
        "state_dict": {k: v.cpu() for k, v in final_head.state_dict().items()},
        "meta": {
            "backbone": "facebook/dinov3-vitl16-pretrain-sat493m",
            "input": "B4/B3/B2 from S2 chip, 1-99 percentile stretch, "
                     "SAT-493M normalization, resize 256->224",
            "head_arch": "Linear(1024, 2)",
            "classes": ["negative", "industrial"],
            "trained_n": int(len(y_all)),
            "n_pos": int((y_all == POS).sum()),
            "n_neg": int((y_all == NEG).sum()),
            "epochs": n_epochs_final,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "cv_metrics": agg,
        },
    }, OUT_MODEL)
    print(f"\nwrote {OUT_MODEL}")

    OUT_REPORT.write_text(json.dumps({
        "fold_metrics": fold_metrics, "aggregate": agg,
    }, indent=2, default=float))
    print(f"wrote {OUT_REPORT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
