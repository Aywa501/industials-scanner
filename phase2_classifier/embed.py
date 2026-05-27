"""Embed every chip in stage1_dataset.parquet with DINOv3 ViT-L/16 SAT-493M.

Outputs:
- data_us/phase1/stage1_embeddings.npy             (N, 1024) float16
- data_us/phase1/stage1_embeddings_index.parquet   site_id, year, row_idx, label,
                                            site_type, source

Notes:
- Backbone frozen. CLS-token feature only.
- B4/B3/B2 -> RGB. 1-99 percentile per-chip stretch (same as labeling webapp),
  then SAT-493M normalization. Resize 256 -> 224.
- Resumable: skips chips already in the index parquet.
"""

from __future__ import annotations

import io
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from google.cloud import storage
from PIL import Image
from rasterio.io import MemoryFile
from transformers import AutoModel

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

GCP_PROJECT = os.getenv("GCP_PROJECT", "")
GCS_BUCKET = os.getenv("GCS_BUCKET", "")

DATA_US = ROOT.parent / "data_us"
DATASET_PATH = DATA_US / "phase1" / "stage1_dataset.parquet"
EMB_PATH = DATA_US / "phase1" / "stage1_embeddings.npy"
INDEX_PATH = DATA_US / "phase1" / "stage1_embeddings_index.parquet"

MODEL_ID = "facebook/dinov3-vitl16-pretrain-sat493m"
HIDDEN = 1024
IMG_SIZE = 224

# SAT-493M normalization (per spec).
MEAN = torch.tensor([0.430, 0.411, 0.296]).view(1, 3, 1, 1)
STD = torch.tensor([0.213, 0.156, 0.143]).view(1, 3, 1, 1)

BATCH_SIZE = 16
NUM_DOWNLOAD_WORKERS = 16


def parse_uri(uri: str) -> str:
    return uri[len("gs://"):].partition("/")[2]


def fetch_and_prep(idx: int, tile_uri: str, bucket) -> tuple[int, torch.Tensor]:
    """Download GeoTIFF from GCS and prep an unnormalized RGB tensor."""
    blob_path = parse_uri(tile_uri)
    tiff = bucket.blob(blob_path).download_as_bytes()

    with MemoryFile(tiff) as mf, mf.open() as src:
        # Bands: B4, B3, B2, B8 in that order.
        rgb = src.read([1, 2, 3]).astype(np.float32)  # (3, H, W)

    lo, hi = np.percentile(rgb, (1, 99))
    if hi <= lo:
        hi = lo + 1
    x = np.clip((rgb - lo) / (hi - lo), 0, 1)  # (3, H, W) in [0, 1]

    img = Image.fromarray(
        (x.transpose(1, 2, 0) * 255).astype(np.uint8)
    ).resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr = np.asarray(img).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)
    return idx, t


def main() -> int:
    if not (GCP_PROJECT and GCS_BUCKET):
        print("error: GCP_PROJECT and GCS_BUCKET must be set", file=sys.stderr)
        return 1

    df = pd.read_parquet(DATASET_PATH)
    print(f"dataset: {len(df)} chips")

    # Resume: skip already-indexed (site_id, year).
    done_keys: set[tuple[str, int]] = set()
    existing_emb: np.ndarray | None = None
    existing_idx: pd.DataFrame | None = None
    if INDEX_PATH.exists() and EMB_PATH.exists():
        existing_idx = pd.read_parquet(INDEX_PATH)
        existing_emb = np.load(EMB_PATH)
        if len(existing_idx) != len(existing_emb):
            print(f"  warning: index/emb length mismatch "
                  f"({len(existing_idx)} vs {len(existing_emb)}); ignoring cache")
            existing_idx, existing_emb = None, None
        else:
            done_keys = set(zip(existing_idx.site_id,
                                existing_idx.year.astype(int)))
            print(f"  resuming, {len(done_keys)} chips already embedded")

    todo = df[~df.apply(
        lambda r: (r.site_id, int(r.year)) in done_keys, axis=1
    )].reset_index(drop=True)
    print(f"  to embed: {len(todo)}")
    if len(todo) == 0:
        print("nothing to do")
        return 0

    # Pick device.
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"device: {device}")

    print(f"loading {MODEL_ID}...")
    t0 = time.time()
    model = AutoModel.from_pretrained(MODEL_ID, dtype=torch.float32)
    model.eval()
    model.to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"  loaded in {time.time()-t0:.1f}s")

    mean = MEAN.to(device)
    std = STD.to(device)

    gcs = storage.Client(project=GCP_PROJECT)
    bucket = gcs.bucket(GCS_BUCKET)

    embs = np.zeros((len(todo), HIDDEN), dtype=np.float16)
    t_start = time.time()
    n_done = 0
    futures: dict = {}

    with ThreadPoolExecutor(max_workers=NUM_DOWNLOAD_WORKERS) as pool, \
            torch.inference_mode():
        # Prime the pump.
        for i in range(min(NUM_DOWNLOAD_WORKERS * 2, len(todo))):
            row = todo.iloc[i]
            futures[i] = pool.submit(fetch_and_prep, i, row.tile_uri, bucket)
        next_submit = min(NUM_DOWNLOAD_WORKERS * 2, len(todo))

        i = 0
        while i < len(todo):
            batch_idx: list[int] = []
            batch_t: list[torch.Tensor] = []
            while len(batch_idx) < BATCH_SIZE and i < len(todo):
                idx, t = futures.pop(i).result()
                batch_idx.append(idx)
                batch_t.append(t)
                i += 1
                if next_submit < len(todo):
                    row = todo.iloc[next_submit]
                    futures[next_submit] = pool.submit(
                        fetch_and_prep, next_submit, row.tile_uri, bucket
                    )
                    next_submit += 1

            xs = torch.stack(batch_t, dim=0).to(device, non_blocking=True)
            xs = (xs - mean) / std
            out = model(xs)
            cls = out.last_hidden_state[:, 0, :]
            embs[np.array(batch_idx)] = cls.detach().to(torch.float16).cpu().numpy()

            n_done += len(batch_idx)
            if n_done % (BATCH_SIZE * 10) < BATCH_SIZE or n_done == len(todo):
                rate = n_done / (time.time() - t_start)
                eta = (len(todo) - n_done) / max(rate, 1e-6)
                print(f"  {n_done}/{len(todo)}  "
                      f"{rate:.1f} chips/s  ETA {eta/60:.1f} min")

    new_idx = todo[["site_id", "year", "label", "site_type", "source"]].copy()
    new_idx["year"] = new_idx.year.astype(int)

    if existing_emb is not None and existing_idx is not None:
        all_emb = np.concatenate([existing_emb, embs], axis=0)
        all_idx = pd.concat([existing_idx, new_idx], ignore_index=True)
    else:
        all_emb = embs
        all_idx = new_idx

    all_idx = all_idx.reset_index(drop=True)
    all_idx["row_idx"] = all_idx.index.astype(int)

    EMB_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(EMB_PATH, all_emb)
    all_idx.to_parquet(INDEX_PATH, index=False)
    print(f"\nwrote {EMB_PATH}  shape={all_emb.shape}  dtype={all_emb.dtype}")
    print(f"wrote {INDEX_PATH}  rows={len(all_idx)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
