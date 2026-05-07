"""Compare probe families across all 5 encoders (binary).

Same train/test split and label mapping as eval_binary.py:
  class_id 0 -> negative
  class_id 2 -> positive

Probes evaluated per encoder: linear-CE (torch), logistic-L2 (sklearn), MLP,
kNN-cosine, LinearSVC, RBF-SVC, Poly2-SVC, HistGradientBoosting,
RandomForest. RBF / Poly use Nystroem approximation when n_train > NYSTROEM_THR.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
from sklearn.svm import SVC, LinearSVC

ROOT = Path(__file__).resolve().parents[3]
V2_DIR = ROOT / "data_us" / "v2"
INDEX_PATH = V2_DIR / "v2_embeddings_index.parquet"
ENCODERS = ["dino_sat493m", "dino_vitb", "resnet50", "prithvi_300m", "prithvi_600m"]
OUT_PATH = V2_DIR / "leaderboard_probes.json"

SEED = 0


def binary_metrics(y_true: np.ndarray, score: np.ndarray) -> dict:
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


def fit_torch_linear(Xtr, ytr, Xte, *, epochs=20, lr=1e-3, wd=1e-4, batch=512):
    Xt = torch.from_numpy(Xtr.astype(np.float32))
    yt = torch.from_numpy(ytr.astype(np.int64))
    Xv = torch.from_numpy(Xte.astype(np.float32))
    n_pos, n_neg = int(ytr.sum()), int((1 - ytr).sum())
    cw = torch.tensor([(n_pos + n_neg) / (2 * max(n_neg, 1)),
                       (n_pos + n_neg) / (2 * max(n_pos, 1))], dtype=torch.float32)
    head = nn.Linear(Xtr.shape[1], 2)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    n = len(Xt)
    g = torch.Generator().manual_seed(SEED)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            loss = F.cross_entropy(head(Xt[idx]), yt[idx], weight=cw)
            opt.zero_grad(); loss.backward(); opt.step()
    head.eval()
    with torch.no_grad():
        return F.softmax(head(Xv), dim=1)[:, 1].numpy()


def fit_torch_mlp(Xtr, ytr, Xte, *, hidden=256, dropout=0.2, epochs=25, lr=1e-3,
                  wd=1e-4, batch=512):
    Xt = torch.from_numpy(Xtr.astype(np.float32))
    yt = torch.from_numpy(ytr.astype(np.int64))
    Xv = torch.from_numpy(Xte.astype(np.float32))
    n_pos, n_neg = int(ytr.sum()), int((1 - ytr).sum())
    cw = torch.tensor([(n_pos + n_neg) / (2 * max(n_neg, 1)),
                       (n_pos + n_neg) / (2 * max(n_pos, 1))], dtype=torch.float32)
    torch.manual_seed(SEED)
    mlp = nn.Sequential(
        nn.Linear(Xtr.shape[1], hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, 2),
    )
    opt = torch.optim.AdamW(mlp.parameters(), lr=lr, weight_decay=wd)
    n = len(Xt)
    g = torch.Generator().manual_seed(SEED)
    for _ in range(epochs):
        mlp.train()
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            loss = F.cross_entropy(mlp(Xt[idx]), yt[idx], weight=cw)
            opt.zero_grad(); loss.backward(); opt.step()
    mlp.eval()
    with torch.no_grad():
        return F.softmax(mlp(Xv), dim=1)[:, 1].numpy()


def fit_logreg(Xtr, ytr, Xte):
    clf = LogisticRegression(
        max_iter=2000, solver="lbfgs", class_weight="balanced",
        C=1.0, random_state=SEED,
    )
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1]


def fit_linsvc(Xtr, ytr, Xte):
    clf = LinearSVC(class_weight="balanced", C=1.0, random_state=SEED, max_iter=5000)
    clf.fit(Xtr, ytr)
    raw = clf.decision_function(Xte)
    return 1.0 / (1.0 + np.exp(-raw))


def fit_svc_rbf_nystroem(Xtr, ytr, Xte, *, n_components=1000):
    """RBF kernel approximation via Nystroem -> LinearSVC.

    True sklearn.svm.SVC with RBF on 50k x 1024 takes hours and scales as O(n^2 d).
    Nystroem with n_components=1000 typically matches true RBF within ~1% AUROC
    while running in minutes, and inference is O(n_components * d) per query.
    """
    fmap = Nystroem(kernel="rbf", gamma=1.0 / Xtr.shape[1], n_components=n_components,
                    random_state=SEED, n_jobs=-1)
    Xtr_f = fmap.fit_transform(Xtr.astype(np.float32))
    Xte_f = fmap.transform(Xte.astype(np.float32))
    clf = LinearSVC(class_weight="balanced", C=1.0, random_state=SEED, max_iter=5000)
    clf.fit(Xtr_f, ytr)
    raw = clf.decision_function(Xte_f)
    return 1.0 / (1.0 + np.exp(-raw))


def fit_svc_poly2_nystroem(Xtr, ytr, Xte, *, n_components=1000):
    """Polynomial degree-2 kernel approximation via Nystroem -> LinearSVC.

    Explicit poly-2 features on 1024 dim would be ~525k features (infeasible).
    Nystroem with poly kernel approximates the same kernel space cheaply.
    """
    fmap = Nystroem(kernel="poly", degree=2, gamma=1.0 / Xtr.shape[1], coef0=1.0,
                    n_components=n_components, random_state=SEED, n_jobs=-1)
    Xtr_f = fmap.fit_transform(Xtr.astype(np.float32))
    Xte_f = fmap.transform(Xte.astype(np.float32))
    clf = LinearSVC(class_weight="balanced", C=1.0, random_state=SEED, max_iter=5000)
    clf.fit(Xtr_f, ytr)
    raw = clf.decision_function(Xte_f)
    return 1.0 / (1.0 + np.exp(-raw))


def fit_svc_rbf_true_subsample(Xtr, ytr, Xte, *, n_subsample=20000):
    """True RBF kernel SVC on a stratified subsample. Direct datapoint to compare
    against the Nystroem approximation. SVC scales O(n^2 d) so we cap n at 20k
    to keep fit under ~10 min per encoder; full data would be ~30+ min per fit.
    """
    rng = np.random.default_rng(SEED)
    pos_idx = np.where(ytr == 1)[0]
    neg_idx = np.where(ytr == 0)[0]
    pos_keep = pos_idx if len(pos_idx) <= n_subsample // 2 else rng.choice(pos_idx, n_subsample // 2, replace=False)
    neg_keep = neg_idx if len(neg_idx) <= n_subsample - len(pos_keep) else rng.choice(neg_idx, n_subsample - len(pos_keep), replace=False)
    sel = np.concatenate([pos_keep, neg_keep])
    rng.shuffle(sel)
    Xs = Xtr[sel].astype(np.float32)
    ys = ytr[sel]
    clf = SVC(kernel="rbf", gamma="scale", C=1.0, class_weight="balanced",
              probability=False, cache_size=1000, random_state=SEED)
    clf.fit(Xs, ys)
    raw = clf.decision_function(Xte.astype(np.float32))
    return 1.0 / (1.0 + np.exp(-raw))


def fit_knn(Xtr, ytr, Xte, *, k=15):
    Xtr_n = normalize(Xtr.astype(np.float32))
    Xte_n = normalize(Xte.astype(np.float32))
    clf = KNeighborsClassifier(n_neighbors=k, weights="distance",
                               algorithm="brute", n_jobs=-1)
    clf.fit(Xtr_n, ytr)
    return clf.predict_proba(Xte_n)[:, 1]


def fit_hgb(Xtr, ytr, Xte):
    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_depth=6,
        class_weight="balanced", random_state=SEED,
    )
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1]


def fit_rf(Xtr, ytr, Xte):
    clf = RandomForestClassifier(
        n_estimators=300, n_jobs=-1, class_weight="balanced",
        max_features="sqrt", min_samples_leaf=2, random_state=SEED,
    )
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1]


PROBES = [
    ("linear_ce_torch",  fit_torch_linear),
    ("logreg_l2",        fit_logreg),
    ("mlp_256",          fit_torch_mlp),
    ("linear_svc",       fit_linsvc),
    ("svc_rbf_nys",      fit_svc_rbf_nystroem),
    ("svc_rbf_true_20k", fit_svc_rbf_true_subsample),
    ("svc_poly2_nys",    fit_svc_poly2_nystroem),
    ("knn_cosine_k15",   fit_knn),
    ("hist_gbm",         fit_hgb),
    ("random_forest",    fit_rf),
]


def main() -> None:
    df = pd.read_parquet(INDEX_PATH)
    keep = df.class_id.isin([0, 2])
    df_b = df[keep].copy()
    df_b["y"] = (df_b.class_id == 2).astype(int)
    train = df_b[df_b.split == "train"]
    test = df_b[df_b.split == "test"]
    train_idx = train.row_idx.to_numpy()
    test_idx = test.row_idx.to_numpy()
    ytr = train.y.to_numpy()
    yte = test.y.to_numpy()
    print(f"[probes] train n={len(train_idx)}  pos={ytr.sum()}  neg={(1-ytr).sum()}")
    print(f"[probes] test  n={len(test_idx)}  pos={yte.sum()}  neg={(1-yte).sum()}")

    all_results: dict[str, dict[str, dict]] = {}
    rows = []
    for enc in ENCODERS:
        emb_path = V2_DIR / f"emb_{enc}.npy"
        if not emb_path.exists():
            print(f"[probes] skip {enc} (no embedding file)")
            continue
        print(f"\n[probes] === encoder {enc} ===")
        emb = np.load(emb_path)
        Xtr = emb[train_idx]
        Xte = emb[test_idx]
        print(f"  emb_dim={emb.shape[1]}")
        per_enc: dict[str, dict] = {}
        for name, fn in PROBES:
            t0 = time.perf_counter()
            try:
                score = fn(Xtr, ytr, Xte)
                m = binary_metrics(yte, score)
                m["fit_seconds"] = round(time.perf_counter() - t0, 2)
            except Exception as e:
                print(f"  {name:18s}  FAILED ({e})")
                continue
            per_enc[name] = m
            rows.append({
                "encoder": enc,
                "probe": name,
                "AUROC": m["AUROC"],
                "AP": m["AP"],
                "acc@0.5": m["acc@0.5"],
                "rec@0.7": m["recall@p>=0.7"],
                "prec@0.7": m["precision@p>=0.7"],
                "rec@0.95": m["recall@p>=0.95"],
                "prec@0.95": m["precision@p>=0.95"],
                "fit_s": m["fit_seconds"],
            })
            print(f"  {name:18s}  AUROC={m['AUROC']:.4f}  AP={m['AP']:.4f}  "
                  f"acc={m['acc@0.5']:.3f}  rec@0.7={m['recall@p>=0.7']:.3f}  "
                  f"prec@0.7={m['precision@p>=0.7']:.3f}  ({m['fit_seconds']}s)")
        all_results[enc] = per_enc

    OUT_PATH.write_text(json.dumps(all_results, indent=2))
    print(f"\n[probes] wrote {OUT_PATH}")
    print()
    df_lb = pd.DataFrame(rows).sort_values("AUROC", ascending=False)
    print(df_lb.to_string(index=False, float_format="%.3f"))
    print()
    print("Best probe per encoder (by AUROC):")
    for enc in ENCODERS:
        sub = df_lb[df_lb.encoder == enc]
        if not sub.empty:
            best = sub.iloc[0]
            print(f"  {enc:14s}  {best.probe:18s}  AUROC={best.AUROC:.4f}  AP={best.AP:.4f}")


if __name__ == "__main__":
    main()
