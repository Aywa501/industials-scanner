"""Re-eval dino_sat493m binary probes after dropping overture_industrial rows
that don't have a building in the 2024-07-22 Overture baseline (= didn't exist
when the 2024 imagery was captured).

Reports two evals per probe:
  - on the ORIGINAL test set (directly comparable to prior leaderboards)
  - on the CLEANED test set (drops the ~3.5% of test positives that didn't exist)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[3]
V2_DIR = ROOT / "data_us" / "v2"
INDEX_PATH = V2_DIR / "v2_embeddings_index.parquet"
FLAGS_PATH = Path("/tmp/manifest_existed_flags.parquet")
EMB_PATH = V2_DIR / "emb_dino_sat493m.npy"
OUT_PATH = V2_DIR / "leaderboard_clean_positives.json"
SEED = 0


def metrics(y_true: np.ndarray, score: np.ndarray, label: str) -> dict:
    out = {
        "n_test": int(len(y_true)),
        "n_pos": int(y_true.sum()),
        "AUROC": float(roc_auc_score(y_true, score)),
        "AP": float(average_precision_score(y_true, score)),
        "acc@0.5": float(accuracy_score(y_true, (score >= 0.5).astype(int))),
    }
    for thr in (0.5, 0.7, 0.9, 0.95):
        sel = score >= thr
        out[f"recall@p>={thr}"] = float(y_true[sel].sum() / max(y_true.sum(), 1))
        out[f"precision@p>={thr}"] = float(y_true[sel].sum() / max(sel.sum(), 1))
    return out


def fit_logreg(Xtr, ytr, Xte):
    clf = LogisticRegression(max_iter=2000, solver="lbfgs",
                             class_weight="balanced", C=1.0, random_state=SEED)
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1]


def fit_hgb(Xtr, ytr, Xte):
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                         max_depth=6, class_weight="balanced",
                                         random_state=SEED)
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1]


def fit_mlp(Xtr, ytr, Xte, *, hidden=256, dropout=0.2, epochs=25, lr=1e-3,
            wd=1e-4, batch=512):
    Xt = torch.from_numpy(Xtr.astype(np.float32))
    yt = torch.from_numpy(ytr.astype(np.int64))
    Xv = torch.from_numpy(Xte.astype(np.float32))
    n_pos, n_neg = int(ytr.sum()), int((1 - ytr).sum())
    cw = torch.tensor([(n_pos + n_neg) / (2 * max(n_neg, 1)),
                       (n_pos + n_neg) / (2 * max(n_pos, 1))], dtype=torch.float32)
    torch.manual_seed(SEED)
    mlp = nn.Sequential(
        nn.Linear(Xtr.shape[1], hidden), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(hidden, 2),
    )
    opt = torch.optim.AdamW(mlp.parameters(), lr=lr, weight_decay=wd)
    n = len(Xt)
    g = torch.Generator().manual_seed(SEED)
    for _ in range(epochs):
        mlp.train()
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, batch):
            ix = perm[i:i + batch]
            loss = F.cross_entropy(mlp(Xt[ix]), yt[ix], weight=cw)
            opt.zero_grad(); loss.backward(); opt.step()
    mlp.eval()
    with torch.no_grad():
        return F.softmax(mlp(Xv), dim=1)[:, 1].numpy()


PROBES = [("logreg_l2", fit_logreg), ("mlp_256", fit_mlp), ("hist_gbm", fit_hgb)]


def main() -> None:
    print(f"[clean] loading {INDEX_PATH}")
    idx = pd.read_parquet(INDEX_PATH)
    flags = pd.read_parquet(FLAGS_PATH)[["tile_id", "existed_2024_07"]]
    df = idx.merge(flags, on="tile_id", how="left")
    # Non-overture rows have no flag — set to True (not subject to filter)
    df["existed_2024_07"] = df["existed_2024_07"].fillna(True)

    # Binary task: keep class 0 (neg) and class 2 (industrial)
    df_b = df[df.class_id.isin([0, 2])].copy()
    df_b["y"] = (df_b.class_id == 2).astype(int)

    # Original train/test (no positive cleaning)
    train_orig = df_b[df_b.split == "train"]
    test_orig  = df_b[df_b.split == "test"]
    # Cleaned: drop overture_industrial rows where building didn't exist 2024-07
    train_clean = df_b[(df_b.split == "train") & df_b.existed_2024_07]
    test_clean  = df_b[(df_b.split == "test")  & df_b.existed_2024_07]

    print(f"[clean] train_orig={len(train_orig):,}  train_clean={len(train_clean):,}  "
          f"dropped={len(train_orig)-len(train_clean):,}")
    print(f"[clean] test_orig ={len(test_orig):,}  test_clean ={len(test_clean):,}  "
          f"dropped={len(test_orig)-len(test_clean):,}")

    emb = np.load(EMB_PATH)
    print(f"[clean] emb shape: {emb.shape}")

    Xtr_orig = emb[train_orig.row_idx.to_numpy()]
    Xte_orig = emb[test_orig.row_idx.to_numpy()]
    Xtr_clean = emb[train_clean.row_idx.to_numpy()]
    Xte_clean = emb[test_clean.row_idx.to_numpy()]
    ytr_orig  = train_orig.y.to_numpy()
    ytr_clean = train_clean.y.to_numpy()
    yte_orig  = test_orig.y.to_numpy()
    yte_clean = test_clean.y.to_numpy()

    results = {}
    rows = []
    for name, fn in PROBES:
        print(f"\n[clean] === {name} ===")

        # Baseline: train on ORIG, eval on ORIG (this should reproduce prior leaderboard)
        s = fn(Xtr_orig, ytr_orig, Xte_orig)
        m = metrics(yte_orig, s, "orig->orig"); m["variant"] = "orig_train_orig_test"
        results[f"{name}__orig_train_orig_test"] = m
        rows.append(dict(probe=name, train="orig", test="orig", AUROC=m["AUROC"], AP=m["AP"],
                         rec07=m["recall@p>=0.7"], prec07=m["precision@p>=0.7"], n=m["n_test"]))
        print(f"  orig→orig:   AUROC={m['AUROC']:.4f} AP={m['AP']:.4f} rec@0.7={m['recall@p>=0.7']:.3f}")

        # Cleaned train, eval on ORIGINAL test set (direct comparison)
        s = fn(Xtr_clean, ytr_clean, Xte_orig)
        m = metrics(yte_orig, s, "clean->orig"); m["variant"] = "clean_train_orig_test"
        results[f"{name}__clean_train_orig_test"] = m
        rows.append(dict(probe=name, train="clean", test="orig", AUROC=m["AUROC"], AP=m["AP"],
                         rec07=m["recall@p>=0.7"], prec07=m["precision@p>=0.7"], n=m["n_test"]))
        print(f"  clean→orig:  AUROC={m['AUROC']:.4f} AP={m['AP']:.4f} rec@0.7={m['recall@p>=0.7']:.3f}")

        # Cleaned train and cleaned test (true ceiling estimate)
        s = fn(Xtr_clean, ytr_clean, Xte_clean)
        m = metrics(yte_clean, s, "clean->clean"); m["variant"] = "clean_train_clean_test"
        results[f"{name}__clean_train_clean_test"] = m
        rows.append(dict(probe=name, train="clean", test="clean", AUROC=m["AUROC"], AP=m["AP"],
                         rec07=m["recall@p>=0.7"], prec07=m["precision@p>=0.7"], n=m["n_test"]))
        print(f"  clean→clean: AUROC={m['AUROC']:.4f} AP={m['AP']:.4f} rec@0.7={m['recall@p>=0.7']:.3f}")

    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\n[clean] wrote {OUT_PATH}\n")
    print(pd.DataFrame(rows).to_string(index=False, float_format="%.4f"))


if __name__ == "__main__":
    main()
