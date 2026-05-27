"""Binary eval on the v2 test set.

Compares:
  - v1 probe (data_us/phase1/stage1_industrial_v1.pt) applied to v2's dino_sat493m embeddings
  - v2 retrained binary probes for each of the 5 encoders

Binary label mapping:
  class_id 0 (overture_neg + random_bg) -> 0 negative
  class_id 2 (overture_industrial)      -> 1 positive
  (class_id 1 — UC — was excised in current manifests; filter is defensive)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, average_precision_score, roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[3]
V2_DIR = ROOT / "data_us" / "v2"
V1_PROBE = ROOT / "data_us" / "stage1_industrial_v1.pt"
INDEX_PATH = V2_DIR / "v2_embeddings_index.parquet"
OUT_PATH = V2_DIR / "leaderboard_binary.json"

ENCODERS = ["dino_sat493m", "dino_vitb", "resnet50", "prithvi_300m", "prithvi_600m"]


def binary_metrics(y_true: np.ndarray, score: np.ndarray) -> dict:
    auroc = roc_auc_score(y_true, score)
    ap = average_precision_score(y_true, score)
    pred = (score >= 0.5).astype(int)
    acc = accuracy_score(y_true, pred)

    out = {
        "n_test": int(len(y_true)),
        "n_pos": int(y_true.sum()),
        "n_neg": int((1 - y_true).sum()),
        "AUROC": float(auroc),
        "AP": float(ap),
        "acc@0.5": float(acc),
    }
    for thr in (0.5, 0.7, 0.9, 0.95):
        sel = score >= thr
        out[f"recall@p>={thr}"] = float(y_true[sel].sum() / max(y_true.sum(), 1))
        out[f"precision@p>={thr}"] = float(y_true[sel].sum() / max(sel.sum(), 1))
    return out


def train_binary_probe(emb_train: np.ndarray, y_train: np.ndarray,
                       emb_test: np.ndarray, y_test: np.ndarray,
                       *, epochs=20, lr=1e-3, weight_decay=1e-4, batch=512,
                       device="cpu") -> tuple[dict, np.ndarray]:
    device = torch.device(device)
    Xt = torch.from_numpy(emb_train.astype(np.float32)).to(device)
    yt = torch.from_numpy(y_train.astype(np.int64)).to(device)
    Xv = torch.from_numpy(emb_test.astype(np.float32)).to(device)

    n_pos = int(y_train.sum())
    n_neg = int((1 - y_train).sum())
    w_pos = (n_pos + n_neg) / (2.0 * max(n_pos, 1))
    w_neg = (n_pos + n_neg) / (2.0 * max(n_neg, 1))
    cw = torch.tensor([w_neg, w_pos], dtype=torch.float32, device=device)

    head = nn.Linear(emb_train.shape[1], 2).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)

    n = len(Xt)
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            logits = head(Xt[idx])
            loss = F.cross_entropy(logits, yt[idx], weight=cw)
            opt.zero_grad()
            loss.backward()
            opt.step()

    head.eval()
    with torch.no_grad():
        score = F.softmax(head(Xv), dim=1)[:, 1].cpu().numpy()
    return binary_metrics(y_test, score), score


def main() -> None:
    print(f"[eval-binary] loading index: {INDEX_PATH}")
    df = pd.read_parquet(INDEX_PATH)

    keep = df.class_id.isin([0, 2])
    df_b = df[keep].copy()
    df_b["y"] = (df_b.class_id == 2).astype(int)

    train = df_b[df_b.split == "train"]
    test = df_b[df_b.split == "test"]
    print(f"[eval-binary] train: n={len(train)}  pos={train.y.sum()}  neg={(1-train.y).sum()}")
    print(f"[eval-binary] test:  n={len(test)}  pos={test.y.sum()}  neg={(1-test.y).sum()}")

    train_idx = train.row_idx.to_numpy()
    test_idx = test.row_idx.to_numpy()
    y_test = test.y.to_numpy()
    y_train = train.y.to_numpy()

    results: dict[str, dict] = {}

    # 1. v1 probe on dino_sat493m embeddings
    print("\n[eval-binary] === v1 probe on v2 dino_sat493m embeddings ===")
    emb_sat = np.load(V2_DIR / "emb_dino_sat493m.npy")
    print(f"  emb shape: {emb_sat.shape}")
    ckpt = torch.load(V1_PROBE, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    head_v1 = nn.Linear(sd["weight"].shape[1], sd["weight"].shape[0])
    head_v1.load_state_dict(sd)
    head_v1.eval()
    print(f"  v1 head: Linear({sd['weight'].shape[1]}, {sd['weight'].shape[0]})  classes={ckpt['meta'].get('classes')}")
    with torch.no_grad():
        Xv = torch.from_numpy(emb_sat[test_idx].astype(np.float32))
        score_v1 = F.softmax(head_v1(Xv), dim=1)[:, 1].numpy()
    m_v1 = binary_metrics(y_test, score_v1)
    results["v1_probe_on_dino_sat493m"] = m_v1
    for k, v in m_v1.items():
        print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

    # 2. Retrain v2 binary probes per encoder
    for name in ENCODERS:
        emb_path = V2_DIR / f"emb_{name}.npy"
        if not emb_path.exists():
            print(f"\n[eval-binary] skip {name} (no embedding file)")
            continue
        print(f"\n[eval-binary] === v2-binary {name} ===")
        emb = np.load(emb_path)
        print(f"  emb shape: {emb.shape}")
        m, _ = train_binary_probe(
            emb[train_idx], y_train,
            emb[test_idx], y_test,
            epochs=20, lr=1e-3, weight_decay=1e-4, batch=512, device="cpu",
        )
        results[f"v2_binary_{name}"] = m
        for k, v in m.items():
            print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\n[eval-binary] wrote {OUT_PATH}")

    # Pretty leaderboard
    rows = []
    for name, m in results.items():
        rows.append({
            "model": name,
            "AUROC": m["AUROC"],
            "AP": m["AP"],
            "acc@0.5": m["acc@0.5"],
            "rec@0.7": m["recall@p>=0.7"],
            "prec@0.7": m["precision@p>=0.7"],
            "rec@0.95": m["recall@p>=0.95"],
            "prec@0.95": m["precision@p>=0.95"],
        })
    print()
    print(pd.DataFrame(rows).sort_values("AUROC", ascending=False).to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()
