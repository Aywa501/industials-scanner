"""Fit final multi-class softmax probes for the v3 detector.

Trains on all rows in v3_embeddings_index.parquet (no held-out fold).
Labels: 11 groups derived from Overture ovt_class — industrial is group 0,
the others carve out the negative classes the binary probe couldn't separate
(commercial / retail / office / parking / education / residential / religious /
institutional / agricultural / other).

Output (per encoder):
  data_us/phase2/v3/probes/probe_<model>_multiclass.pt
    {state_dict, emb_dim, n_classes, class_labels, industrial_idx}

CV comparison vs the binary probe lives in this directory's notebooks/log —
multi_p (use p_industrial) beats binary by ~1.3 AUROC pp and lowers FPR@R=0.7
by ~2 pp on both encoders.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
DATA_US = ROOT / "data_us"
IDX_PATH = DATA_US / "phase2" / "v3" / "v3_embeddings_index.parquet"
MANI_PATH = DATA_US / "phase2" / "v3_dataset_manifest.parquet"
PROBES_DIR = DATA_US / "phase2" / "v3" / "probes"

ENCODERS = {
    "dino_vitb":    DATA_US / "phase2" / "v3" / "emb_dino_vitb.npy",
    "dino_sat493m": DATA_US / "phase2" / "v3" / "emb_dino_sat493m.npy",
}

# group label -> set of Overture ovt_class strings
GROUPS = {
    "industrial":    {"industrial", "warehouse", "hangar", "manufacture"},
    "commercial":    {"commercial"},
    "retail":        {"retail"},
    "office":        {"office"},
    "parking":       {"parking"},
    "education":     {"school", "university", "college", "kindergarten", "library"},
    "residential":   {"apartments", "residential", "house", "detached", "terrace",
                      "semidetached_house", "dormitory"},
    "religious":     {"church", "cathedral", "mosque", "synagogue", "religious"},
    "institutional": {"hospital", "hotel", "civic", "government", "post_office",
                      "public", "fire_station", "stadium", "grandstand"},
    "agricultural":  {"farm_auxiliary", "barn", "greenhouse"},
}
CLASS_LABELS = list(GROUPS.keys()) + ["other"]
G2I = {g: i for i, g in enumerate(CLASS_LABELS)}
INDUSTRIAL_IDX = G2I["industrial"]

EPOCHS = 120
LR = 1e-3
WD = 1e-4


def assign_group(c: str) -> str:
    for g, members in GROUPS.items():
        if c in members:
            return g
    return "other"


def train_probe(emb: np.ndarray, y: np.ndarray, n_classes: int, device: str) -> nn.Linear:
    D = emb.shape[1]
    net = nn.Linear(D, n_classes).to(device).float()
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WD)
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    weights = counts.sum() / (n_classes * np.maximum(counts, 1.0))
    w_t = torch.from_numpy(weights.astype(np.float32)).to(device)
    Xt = torch.from_numpy(emb.astype(np.float32)).to(device)
    yt = torch.from_numpy(y.astype(np.int64)).to(device)
    for ep in range(EPOCHS):
        opt.zero_grad()
        loss = F.cross_entropy(net(Xt), yt, weight=w_t)
        loss.backward(); opt.step()
        if (ep + 1) % 30 == 0:
            with torch.no_grad():
                pred = net(Xt).argmax(dim=1)
                acc = (pred == yt).float().mean().item()
                print(f"    epoch {ep+1}: loss={loss.item():.4f} train_acc={acc:.4f}")
    return net


def main() -> None:
    print(f"[fit-mc] loading embeddings index ({IDX_PATH.name})...")
    idx = pd.read_parquet(IDX_PATH)
    mani = pd.read_parquet(MANI_PATH, columns=["building_id", "ovt_class"])
    df = idx.merge(mani, on="building_id", how="left")
    df["ovt_class"] = df["ovt_class"].fillna("hand_unknown")
    df["group"] = df["ovt_class"].apply(assign_group)
    df["g_idx"] = df["group"].map(G2I).astype(int)
    y = df["g_idx"].values.astype(np.int64)

    print("[fit-mc] group counts:")
    counts = df["group"].value_counts().reindex(CLASS_LABELS, fill_value=0)
    print(counts.to_string())
    print()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[fit-mc] device: {device}")
    PROBES_DIR.mkdir(parents=True, exist_ok=True)

    for name, emb_path in ENCODERS.items():
        print(f"\n[fit-mc] === {name} ===")
        emb = np.load(emb_path).astype(np.float32)
        # Drop rows where the embedding index doesn't have a matching manifest row
        # (handful of stragglers from chunk concatenation). We have row_idx in idx,
        # which is the row in the .npy file.
        row_idx = df["row_idx"].values
        emb_aligned = emb[row_idx]
        print(f"  emb: {emb_aligned.shape}, labels: {len(y)}")
        net = train_probe(emb_aligned, y, n_classes=len(CLASS_LABELS), device=device)

        ckpt = {
            "state_dict": {k: v.cpu() for k, v in net.state_dict().items()},
            "emb_dim": emb_aligned.shape[1],
            "n_classes": len(CLASS_LABELS),
            "class_labels": CLASS_LABELS,
            "industrial_idx": INDUSTRIAL_IDX,
        }
        out = PROBES_DIR / f"probe_{name}_multiclass.pt"
        torch.save(ckpt, out)
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
