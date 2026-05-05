"""GPU worker: score one MGRS shard against the Stage 1 industrial probe.

Run on a g6.xlarge in us-west-2 (so reads from sentinel-cogs are in-region).

For each pixel tile in the shard:
  1. Project centroid → the MGRS tile's UTM zone CRS
  2. For each of N scenes (~8): window-read 256×256 of B04, B03, B02, SCL
  3. Mask cloudy pixels via SCL (classes 3, 8, 9, 10)
  4. Pixel-median across scenes → (3, 256, 256) composite
  5. 1–99 percentile stretch per band → resize 256→224 (LANCZOS) → SAT-493M normalize
  6. Batch into the GPU; DINOv3 ViT-L → CLS → linear probe → softmax
  7. Append (tile_id, lon, lat, prob) to the shard's result parquet

Usage:
  python -m phase3_scan.infer_shard --mgrs 14TMQ
  python -m phase3_scan.infer_shard --mgrs-list mgrs_todo.txt

Reads:
  data_us/phase3_grid.parquet
  data_us/phase3_scenes.parquet
  data_us/stage1_industrial_v1.pt
Writes:
  results/{mgrs}.parquet           (locally; uploaded to S3 by bootstrap.sh)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import NamedTuple

import boto3
import numpy as np
import pandas as pd
import pyproj
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from dotenv import load_dotenv
from PIL import Image
from rasterio.session import AWSSession
from rasterio.windows import Window, from_bounds

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
DATA_US = ROOT.parent / "data_us"
GRID_PATH = DATA_US / "phase3_grid.parquet"
SCENES_PATH = DATA_US / "phase3_scenes.parquet"
MODEL_PATH = DATA_US / "stage1_industrial_v1.pt"
RESULTS_DIR = DATA_US / "phase3_results"

MODEL_ID = "facebook/dinov3-vitl16-pretrain-sat493m"
IMG_NATIVE = 256          # pixels read per side at S2 native 10m
IMG_INPUT = 224           # ViT input
GSD_M = 10.0
HALF_M = (IMG_NATIVE / 2) * GSD_M  # 1280 m half-width

# SCL classes treated as bad pixels:
#   3 cloud shadow, 8 cloud medium, 9 cloud high, 10 thin cirrus
SCL_BAD = np.array([3, 8, 9, 10], dtype=np.uint8)
MIN_VALID_PIXELS = 256        # per-scene gate; chips below this are skipped
MIN_VALID_SCENES = 1          # tile is skipped if 0 scenes survive masking

BATCH_SIZE = 32

# CLS embeddings are saved for tiles above this prob, so we can train a
# nonlinear refiner downstream without re-running the ViT pass.
EMBED_SAVE_PROB = 0.30

MEAN = torch.tensor([0.430, 0.411, 0.296]).view(1, 3, 1, 1)
STD = torch.tensor([0.213, 0.156, 0.143]).view(1, 3, 1, 1)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_model(device: torch.device):
    from transformers import AutoModel
    print(f"[infer] loading {MODEL_ID} on {device}")
    model = AutoModel.from_pretrained(MODEL_ID, dtype=torch.float16 if device.type == "cuda" else torch.float32)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)

    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    head = nn.Linear(1024, 2)
    head.load_state_dict(ckpt["state_dict"])
    head.eval().to(device)
    if device.type == "cuda":
        head.half()
    return model, head


# ---------------------------------------------------------------------------
# Per-tile read/composite
# ---------------------------------------------------------------------------

class SceneReaders(NamedTuple):
    scene_id: str
    b04: rasterio.DatasetReader
    b03: rasterio.DatasetReader
    b02: rasterio.DatasetReader
    scl: rasterio.DatasetReader


def _utm_epsg(mgrs_tile: str) -> int:
    """Sentinel-2 CONUS scenes are in northern UTM. EPSG = 32600 + zone."""
    zone = int(mgrs_tile[:-3])
    return 32600 + zone


def _open_scene_readers(scene_rows: pd.DataFrame) -> list[SceneReaders]:
    out = []
    for _, r in scene_rows.iterrows():
        try:
            out.append(SceneReaders(
                scene_id=r.scene_id,
                b04=rasterio.open(r.b04_s3),
                b03=rasterio.open(r.b03_s3),
                b02=rasterio.open(r.b02_s3),
                scl=rasterio.open(r.scl_s3),
            ))
        except Exception as e:
            print(f"[infer]   skipping scene {r.scene_id}: {e!r}")
    return out


def _close_scenes(scenes: list[SceneReaders]) -> None:
    for s in scenes:
        for r in (s.b04, s.b03, s.b02, s.scl):
            try:
                r.close()
            except Exception:
                pass


def _read_window(reader, utm_x: float, utm_y: float, size: int) -> np.ndarray | None:
    win = from_bounds(
        utm_x - HALF_M, utm_y - HALF_M, utm_x + HALF_M, utm_y + HALF_M,
        transform=reader.transform,
    )
    col = int(round(win.col_off))
    row = int(round(win.row_off))
    if col < 0 or row < 0 or col + size > reader.width or row + size > reader.height:
        return None
    return reader.read(1, window=Window(col, row, size, size))


def _build_composite(scenes: list[SceneReaders], utm_x: float, utm_y: float
                     ) -> np.ndarray | None:
    """Returns (3, IMG_NATIVE, IMG_NATIVE) float32 median composite, or None."""
    chips: list[np.ndarray] = []
    for s in scenes:
        b04 = _read_window(s.b04, utm_x, utm_y, IMG_NATIVE)
        b03 = _read_window(s.b03, utm_x, utm_y, IMG_NATIVE)
        b02 = _read_window(s.b02, utm_x, utm_y, IMG_NATIVE)
        scl = _read_window(s.scl, utm_x, utm_y, IMG_NATIVE)
        if any(x is None for x in (b04, b03, b02, scl)):
            continue
        ok = ~np.isin(scl, SCL_BAD)
        if ok.sum() < MIN_VALID_PIXELS:
            continue
        rgb = np.stack([b04, b03, b02]).astype(np.float32)
        rgb[:, ~ok] = np.nan
        chips.append(rgb)

    if len(chips) < MIN_VALID_SCENES:
        return None

    stacked = np.stack(chips, axis=0)              # (N, 3, H, W)
    with np.errstate(all="ignore"):
        composite = np.nanmedian(stacked, axis=0)  # (3, H, W)
    for c in range(3):
        col = composite[c]
        nans = np.isnan(col)
        if nans.any():
            fill = np.nanmedian(col) if not np.isnan(col).all() else 0.0
            composite[c, nans] = fill
    return composite


def _to_input(composite: np.ndarray) -> torch.Tensor:
    """(3, IMG_NATIVE, IMG_NATIVE) → (3, IMG_INPUT, IMG_INPUT) normalized float32."""
    out = np.empty_like(composite, dtype=np.float32)
    for c in range(3):
        lo, hi = np.percentile(composite[c], [1, 99])
        out[c] = np.clip((composite[c] - lo) / max(hi - lo, 1e-6), 0, 1)
    arr_u8 = (out.transpose(1, 2, 0) * 255).astype(np.uint8)
    img = Image.fromarray(arr_u8).resize((IMG_INPUT, IMG_INPUT), Image.LANCZOS)
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr.transpose(2, 0, 1))


# ---------------------------------------------------------------------------
# Per-shard driver
# ---------------------------------------------------------------------------

def process_shard(mgrs_tile: str, model, head, device: torch.device) -> Path:
    grid = pd.read_parquet(GRID_PATH)
    grid = grid[grid.mgrs_tile == mgrs_tile].reset_index(drop=True)
    if grid.empty:
        print(f"[infer] {mgrs_tile}: no tiles in grid, skipping")
        return None

    scenes_df = pd.read_parquet(SCENES_PATH)
    scenes_df = scenes_df[scenes_df.mgrs_tile == mgrs_tile]
    if scenes_df.empty:
        print(f"[infer] {mgrs_tile}: no scenes, skipping")
        return None

    out_path = RESULTS_DIR / f"{mgrs_tile}.parquet"
    if out_path.exists():
        print(f"[infer] {mgrs_tile}: result already exists, skipping")
        return out_path

    print(f"[infer] {mgrs_tile}: {len(grid):,} tiles × {len(scenes_df)} scenes")

    epsg = _utm_epsg(mgrs_tile)
    to_utm = pyproj.Transformer.from_crs(4326, epsg, always_xy=True).transform
    utm_xy = np.column_stack(to_utm(grid.lon.to_numpy(), grid.lat.to_numpy()))

    mean_t = MEAN.to(device, dtype=torch.float16 if device.type == "cuda" else torch.float32)
    std_t = STD.to(device, dtype=torch.float16 if device.type == "cuda" else torch.float32)

    scenes = _open_scene_readers(scenes_df)
    if not scenes:
        print(f"[infer] {mgrs_tile}: failed to open any scene, skipping")
        return None

    t0 = time.time()
    results: list[tuple[str, float, float, float]] = []
    embeddings: list[tuple[str, np.ndarray]] = []
    batch_buf: list[torch.Tensor] = []
    batch_meta: list[tuple[str, float, float]] = []

    def flush():
        if not batch_buf:
            return
        x = torch.stack(batch_buf).to(device)
        if device.type == "cuda":
            x = x.half()
        x = (x - mean_t) / std_t
        with torch.inference_mode():
            cls = model(x).last_hidden_state[:, 0, :]   # (B, 1024)
            logits = head(cls)
            probs = F.softmax(logits, dim=1)[:, 1].float().cpu().numpy()
        cls_np = cls.float().cpu().numpy()
        for (tid, lon, lat), p, emb in zip(batch_meta, probs, cls_np):
            results.append((tid, lon, lat, float(p)))
            if p > EMBED_SAVE_PROB:
                embeddings.append((tid, emb.astype(np.float16)))
        batch_buf.clear()
        batch_meta.clear()

    skipped = 0
    try:
        for i, (tid, lon, lat, ux, uy) in enumerate(zip(
            grid.tile_id.to_numpy(), grid.lon.to_numpy(), grid.lat.to_numpy(),
            utm_xy[:, 0], utm_xy[:, 1],
        )):
            comp = _build_composite(scenes, float(ux), float(uy))
            if comp is None:
                skipped += 1
                continue
            batch_buf.append(_to_input(comp))
            batch_meta.append((tid, float(lon), float(lat)))
            if len(batch_buf) >= BATCH_SIZE:
                flush()
            if i and i % 1000 == 0:
                rate = i / max(time.time() - t0, 1e-6)
                eta = (len(grid) - i) / max(rate, 1e-6) / 60
                print(f"[infer]   {mgrs_tile} {i}/{len(grid)} "
                      f"({rate:.1f} tiles/s, ~{eta:.1f} min left, skipped={skipped})")
        flush()
    finally:
        _close_scenes(scenes)

    out_df = pd.DataFrame(results, columns=["tile_id", "lon", "lat", "prob"])
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)

    if embeddings:
        emb_path = RESULTS_DIR / f"{mgrs_tile}_emb.parquet"
        emb_df = pd.DataFrame({
            "tile_id": [t for t, _ in embeddings],
            "embedding": [e.tolist() for _, e in embeddings],
        })
        emb_df.to_parquet(emb_path, index=False)

    dt = time.time() - t0
    print(f"[infer] {mgrs_tile} done: {len(out_df):,} probs "
          f"({len(embeddings):,} embeddings saved) in {dt:.1f}s "
          f"({len(grid) / max(dt, 1e-6):.1f} tiles/s, skipped={skipped}) → {out_path}")
    return out_path


def _setup_rasterio_env() -> rasterio.Env:
    """Tune GDAL for COG range-reads from S3."""
    return rasterio.Env(
        AWSSession(boto3.Session(), requester_pays=True),
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        GDAL_HTTP_MULTIPLEX="YES",
        GDAL_HTTP_VERSION="2",
        VSI_CACHE="TRUE",
        VSI_CACHE_SIZE="200000000",
        CPL_VSIL_CURL_CHUNK_SIZE="1048576",
        CPL_VSIL_CURL_CACHE_SIZE="200000000",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mgrs", help="single MGRS tile, e.g. 14TMQ")
    ap.add_argument("--mgrs-list", help="text file with one MGRS tile per line")
    args = ap.parse_args()

    if args.mgrs:
        tiles = [args.mgrs]
    elif args.mgrs_list:
        tiles = [t.strip() for t in Path(args.mgrs_list).read_text().splitlines() if t.strip()]
    else:
        sys.exit("must pass --mgrs or --mgrs-list")

    device = _device()
    model, head = _load_model(device)

    with _setup_rasterio_env():
        for t in tiles:
            try:
                process_shard(t, model, head, device)
            except Exception as e:
                print(f"[infer] {t} FAILED: {e!r}")


if __name__ == "__main__":
    main()
