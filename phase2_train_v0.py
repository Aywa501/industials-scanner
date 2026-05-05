"""Phase 2 step 2: train v0 Sentinel-2 patch change classifier (siamese ResNet-18).

Pulls chips from local cache (downloads from GCS if missing). Constructs pair
labels from the manifest using announcement_date as the event year. Trains a
siamese ResNet-18 (4-band input) with BCE on chip pairs (chip_a, chip_b, label).
Evaluates with PR curve on a held-out site-level 20% split.

Recall-first: we report recall at multiple precision points so we can pick the
operating threshold the national scan should run at.

Usage:
    python phase2_train_v0.py --download-only       # just hydrate the cache
    python phase2_train_v0.py                        # cache + build pairs + train
    python phase2_train_v0.py --epochs 30 --lr 5e-4
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn as nn
from dotenv import load_dotenv
from google.cloud import storage
from rasterio.errors import RasterioIOError
from sklearn.metrics import precision_recall_curve, average_precision_score
from torch.utils.data import Dataset, DataLoader
from torchvision import models

load_dotenv(Path(__file__).parent / ".env")

GCP_PROJECT = os.getenv("GCP_PROJECT", "")
GCS_BUCKET = os.getenv("GCS_BUCKET", "")
MANIFEST_URI = f"gs://{GCS_BUCKET}/manifest/s2_chip_manifest.parquet"

CACHE_DIR = Path(__file__).parent / ".cache" / "s2_chips"
ARTIFACT_DIR = Path(__file__).parent / ".artifacts" / "v0"

S2_BANDS = 4
TILE_PX = 256
CROP_PX = 128                  # center-crop from 256→128 (focuses on site; 1.28 km × 1.28 km)
PAIR_DELTA_MIN = 1
PAIR_DELTA_MAX = 4
POST_LAG_YEARS = 2             # positives require year_b ≥ ann_year + POST_LAG_YEARS
VAL_SPLIT = 0.2
SEED = 42

REFLECTANCE_DIVISOR = 3000.0


# ─────────────────────────── chip cache ────────────────────────────

def _gcs_blob_path(tile_uri: str) -> str:
    return tile_uri[len(f"gs://{GCS_BUCKET}/"):]


def _local_path(site_id: str, year: int) -> Path:
    return CACHE_DIR / site_id / f"{year}.tif"


def hydrate_cache(manifest: pd.DataFrame, workers: int = 24) -> None:
    """Download all COMPLETED chips to local cache if missing."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    gcs = storage.Client(project=GCP_PROJECT)
    bucket = gcs.bucket(GCS_BUCKET)
    completed = manifest[manifest.export_status == "COMPLETED"]
    todo: list[tuple[str, int, str]] = []
    for _, row in completed.iterrows():
        local = _local_path(row["site_id"], int(row["year"]))
        if not local.exists() or local.stat().st_size == 0:
            todo.append((row["site_id"], int(row["year"]), _gcs_blob_path(row["tile_uri"])))
    print(f"cache: {len(completed) - len(todo)} present, {len(todo)} to download")
    if not todo:
        return

    def fetch(item):
        sid, year, blob_path = item
        local = _local_path(sid, year)
        local.parent.mkdir(parents=True, exist_ok=True)
        bucket.blob(blob_path).download_to_filename(str(local))
        return sid, year

    n_done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for _ in ex.map(fetch, todo):
            n_done += 1
            if n_done % 500 == 0 or n_done == len(todo):
                print(f"  downloaded {n_done}/{len(todo)}")


# ────────────────────── pair-label construction ────────────────────

def build_pairs(manifest: pd.DataFrame) -> pd.DataFrame:
    df = manifest[manifest.export_status == "COMPLETED"].copy()
    df["year"] = df["year"].astype(int)

    def deltas(years: list[int]) -> list[tuple[int, int]]:
        out = []
        for i, ya in enumerate(years):
            for yb in years[i + 1:]:
                if PAIR_DELTA_MIN <= yb - ya <= PAIR_DELTA_MAX:
                    out.append((ya, yb))
        return out

    rows: list[dict] = []
    for site_id, grp in df.groupby("site_id"):
        site_type = grp["site_type"].iloc[0]
        ann = grp["announcement_date"].iloc[0]
        ann_year: int | None = None
        if isinstance(ann, str) and ann:
            try:
                ann_year = int(ann[:4])
            except ValueError:
                ann_year = None
        years = sorted(grp["year"].unique().tolist())
        for ya, yb in deltas(years):
            if site_type == "anchor":
                if ann_year is None:
                    label = None
                elif ya <= ann_year - 1 and yb >= ann_year + POST_LAG_YEARS:
                    label = 1  # construction window: pre-announcement vs post-construction-visible
                elif yb < ann_year:
                    label = 0  # entirely before announcement (stable pre)
                elif ya > ann_year + POST_LAG_YEARS:
                    label = 0  # entirely after construction settled
                else:
                    label = None  # ambiguous: spans only part of the construction window
            else:
                label = 0  # negative site
            if label is None:
                continue
            rows.append({
                "site_id": site_id, "site_type": site_type,
                "year_a": ya, "year_b": yb, "label": int(label),
                "ann_year": ann_year if ann_year is not None else -1,
            })
    pairs = pd.DataFrame(rows)
    print(f"pairs: {len(pairs)} ({pairs['label'].sum()} pos, "
          f"{(pairs['label'] == 0).sum()} neg)  "
          f"by site_type: {dict(pairs.groupby('site_type')['label'].value_counts())}")
    return pairs


def site_split(pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    anchors = pairs[pairs.site_type == "anchor"].site_id.unique().tolist()
    negs = pairs[pairs.site_type == "negative"].site_id.unique().tolist()
    rng.shuffle(anchors); rng.shuffle(negs)
    a_val = set(anchors[: int(len(anchors) * VAL_SPLIT)])
    n_val = set(negs[: int(len(negs) * VAL_SPLIT)])
    val_sites = a_val | n_val
    val = pairs[pairs.site_id.isin(val_sites)].reset_index(drop=True)
    train = pairs[~pairs.site_id.isin(val_sites)].reset_index(drop=True)
    print(f"split: train={len(train)} ({train.label.sum()} pos), "
          f"val={len(val)} ({val.label.sum()} pos)")
    return train, val


# ─────────────────────────── dataset ──────────────────────────────

def read_chip(site_id: str, year: int) -> np.ndarray:
    path = _local_path(site_id, year)
    with rasterio.open(path) as src:
        arr = src.read(out_dtype="float32")  # (C, H, W)
    if arr.shape[0] != S2_BANDS:
        raise RasterioIOError(f"{path}: expected {S2_BANDS} bands, got {arr.shape[0]}")
    if CROP_PX < TILE_PX:
        off = (TILE_PX - CROP_PX) // 2
        arr = arr[:, off:off + CROP_PX, off:off + CROP_PX]
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.clip(arr / REFLECTANCE_DIVISOR, 0.0, 1.0)
    return arr


class PairDataset(Dataset):
    def __init__(self, pairs: pd.DataFrame, augment: bool = False):
        self.df = pairs.reset_index(drop=True)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        r = self.df.iloc[idx]
        a = read_chip(r["site_id"], int(r["year_a"]))
        b = read_chip(r["site_id"], int(r["year_b"]))
        if self.augment:
            # random flip + random transpose, applied identically to both
            if random.random() < 0.5:
                a, b = a[:, :, ::-1].copy(), b[:, :, ::-1].copy()
            if random.random() < 0.5:
                a, b = a[:, ::-1, :].copy(), b[:, ::-1, :].copy()
        return torch.from_numpy(a), torch.from_numpy(b), torch.tensor([float(r["label"])])


# ───────────────────────────── model ──────────────────────────────

class SiameseResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Replace first conv: 4-channel input. Init RGB weights from pretrained,
        # NIR channel from R weights as a sensible warm start.
        old = backbone.conv1
        new = nn.Conv2d(S2_BANDS, old.out_channels, kernel_size=old.kernel_size,
                         stride=old.stride, padding=old.padding, bias=False)
        with torch.no_grad():
            new.weight[:, :3] = old.weight
            new.weight[:, 3:4] = old.weight[:, 0:1]
        backbone.conv1 = new
        backbone.fc = nn.Identity()
        self.encoder = backbone
        self.head = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        ea, eb = self.encoder(a), self.encoder(b)
        return self.head(eb - ea)


# ───────────────────────────── train ──────────────────────────────

def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def evaluate(model: nn.Module, loader: DataLoader, dev: torch.device) -> dict:
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for a, b, y in loader:
            a, b = a.to(dev), b.to(dev)
            logit = model(a, b).cpu().squeeze(-1)
            ys.append(y.squeeze(-1).numpy())
            ps.append(torch.sigmoid(logit).numpy())
    y = np.concatenate(ys); p = np.concatenate(ps)
    ap = average_precision_score(y, p)
    prec, rec, thr = precision_recall_curve(y, p)
    out = {"ap": float(ap), "n_pos": int(y.sum()), "n": len(y)}
    for target_p in (0.5, 0.7, 0.9):
        # recall at first threshold where precision >= target_p
        idx = np.where(prec[:-1] >= target_p)[0]
        out[f"recall@p{int(target_p*100)}"] = float(rec[idx].max()) if len(idx) else 0.0
    return out


def train(epochs: int = 20, lr: float = 1e-3, batch_size: int = 32,
          weight_pos: float | None = None) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    manifest = pd.read_parquet(MANIFEST_URI)
    print(f"manifest: {len(manifest)} rows")
    hydrate_cache(manifest)

    pairs = build_pairs(manifest)
    pairs.to_parquet(ARTIFACT_DIR / "pairs.parquet")
    train_df, val_df = site_split(pairs)

    if weight_pos is None:
        n_pos = max(1, int(train_df.label.sum()))
        n_neg = max(1, len(train_df) - n_pos)
        weight_pos = n_neg / n_pos
    print(f"pos_weight: {weight_pos:.2f}")

    dev = device()
    print(f"device: {dev}")
    model = SiameseResNet18().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight_pos]).to(dev))

    train_loader = DataLoader(PairDataset(train_df, augment=True),
                               batch_size=batch_size, shuffle=True,
                               num_workers=4, persistent_workers=True)
    val_loader = DataLoader(PairDataset(val_df, augment=False),
                             batch_size=batch_size, shuffle=False,
                             num_workers=4, persistent_workers=True)

    best_ap = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0; n = 0
        for a, b, y in train_loader:
            a, b, y = a.to(dev), b.to(dev), y.to(dev)
            logit = model(a, b)
            loss = loss_fn(logit, y)
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item() * a.size(0); n += a.size(0)
        train_loss = running / max(1, n)

        m = evaluate(model, val_loader, dev)
        print(f"epoch {epoch:02d}  train_loss={train_loss:.4f}  "
              f"val_AP={m['ap']:.3f}  R@P50={m['recall@p50']:.3f}  "
              f"R@P70={m['recall@p70']:.3f}  R@P90={m['recall@p90']:.3f}")
        if m["ap"] > best_ap:
            best_ap = m["ap"]
            torch.save({"model": model.state_dict(), "metrics": m},
                       ARTIFACT_DIR / "v0_best.pt")

    print(f"best val AP: {best_ap:.3f}; checkpoint at {ARTIFACT_DIR / 'v0_best.pt'}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--download-only", action="store_true")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--pos-weight", type=float, default=None)
    args = p.parse_args()

    if not (GCP_PROJECT and GCS_BUCKET):
        p.error("GCP_PROJECT and GCS_BUCKET must be set")

    if args.download_only:
        manifest = pd.read_parquet(MANIFEST_URI)
        hydrate_cache(manifest)
        return 0

    train(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
          weight_pos=args.pos_weight)
    return 0


if __name__ == "__main__":
    sys.exit(main())
