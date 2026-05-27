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
  python -m phase3_scan.v1.infer_shard --mgrs 14TMQ
  python -m phase3_scan.v1.infer_shard --mgrs-list mgrs_todo.txt

Reads:
  data_us/phase3_scan/phase3_grid.parquet
  data_us/phase3_scan/phase3_scenes.parquet
  data_us/phase1/stage1_industrial_v1.pt
Writes:
  results/{mgrs}.parquet           (locally; uploaded to S3 by bootstrap.sh)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
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
from rasterio.transform import rowcol
from rasterio.windows import Window, from_bounds

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
DATA_US = ROOT.parent / "data_us"
GRID_PATH = DATA_US / "phase3_scan" / "phase3_grid.parquet"
SCENES_PATH = DATA_US / "phase3_scan" / "phase3_scenes.parquet"
MODEL_PATH = DATA_US / "phase1" / "stage1_industrial_v1.pt"
RESULTS_DIR = DATA_US / "phase3_scan" / "results"

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

# Parallel range reads. Each rasterio dataset reader is single-threaded, but
# we open one reader per (scene, band) so the 32 (= 8 scenes × 4 bands) reads
# for a single tile can fan out across threads. ~10× speedup vs serial reads.
IO_WORKERS = 32

# Memory budget for the bulk-loaded scene arrays. On g6.2xlarge (32 GB host),
# 20 GB for arrays + ~5 GB model + Python/working still leaves comfortable
# headroom. Full 110×110 km × 8 scenes is ~7.7 GB, so this is single-chunk
# for any realistic shard.
MEMORY_BUDGET_BYTES = 20 * 1024**3

# Tile-prep workers. After bulk-loading scene arrays the per-tile work is
# numpy/PIL ops that release the GIL — running multiple tiles in parallel
# threads gives near-linear speedup up to vCPU count.
PREP_WORKERS = 8
PREP_CHUNK = 256  # tiles per prep chunk; bounds memory of pending-future pile

# Cap scenes per shard. 8 cleanest by cloud cover is plenty for a median
# composite; more scenes scale memory cost without much marginal quality.
MAX_SCENES_PER_SHARD = 8

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


def _plan_chunks(utm_xy: np.ndarray, n_scenes: int) -> list[np.ndarray]:
    """Split tile indices into spatial chunks whose bulk-load fits memory.

    Estimates the per-chunk array footprint as
        (W_px × H_px × n_scenes × 4 bands × 2 bytes)
    and picks the smallest N that brings it under MEMORY_BUDGET_BYTES.
    Tiles are sorted by UTM-Y and split into N roughly-equal latitude bands."""
    n = len(utm_xy)
    if n == 0:
        return []
    w_px = (utm_xy[:, 0].max() - utm_xy[:, 0].min()) / GSD_M + IMG_NATIVE
    h_px = (utm_xy[:, 1].max() - utm_xy[:, 1].min()) / GSD_M + IMG_NATIVE
    bytes_est = w_px * h_px * n_scenes * 4 * 2
    n_chunks = max(1, int(np.ceil(bytes_est / MEMORY_BUDGET_BYTES)))
    if n_chunks == 1:
        return [np.arange(n)]
    sorted_idx = np.argsort(utm_xy[:, 1])
    return np.array_split(sorted_idx, n_chunks)


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


def _bulk_read(reader, xmin: float, ymin: float, xmax: float, ymax: float,
               out_shape: tuple[int, int] | None = None):
    """Read the rectangular UTM bbox once, return (array, transform) or (None, None).
    Clips the window to the reader's valid extent. If `out_shape` is given, the
    band is resampled (nearest) to that shape — used to bring 20m SCL onto the
    10m B04 grid so we can slice them with the same row/col."""
    win = from_bounds(xmin, ymin, xmax, ymax, transform=reader.transform)
    col = max(0, int(round(win.col_off)))
    row = max(0, int(round(win.row_off)))
    col_end = min(reader.width, int(round(win.col_off + win.width)))
    row_end = min(reader.height, int(round(win.row_off + win.height)))
    if col >= col_end or row >= row_end:
        return None, None
    actual = Window(col, row, col_end - col, row_end - row)
    native_tr = reader.window_transform(actual)
    if out_shape is None:
        arr = reader.read(1, window=actual)
        return arr, native_tr
    # Resampled read — collapse the actual window into out_shape px nearest.
    arr = reader.read(
        1, window=actual, out_shape=out_shape,
        resampling=rasterio.enums.Resampling.nearest,
    )
    scale_x = (col_end - col) / out_shape[1]
    scale_y = (row_end - row) / out_shape[0]
    out_tr = native_tr * native_tr.scale(scale_x, scale_y)
    return arr, out_tr


def _load_shard_arrays(scenes: list[SceneReaders], executor: ThreadPoolExecutor,
                       bbox: tuple[float, float, float, float]) -> list[dict | None]:
    """Bulk-read every (scene × band) into memory once for the shard.
    Returns one dict per scene with keys b04/b03/b02/scl/transform, or None
    for scenes that don't intersect the bbox.
    SCL (20m) is resampled onto the B04 (10m) grid so all four slice identically."""
    xmin, ymin, xmax, ymax = bbox
    # First pass: B04/B03/B02 at native 10m.
    rgb_readers = []
    for s in scenes:
        rgb_readers.extend([s.b04, s.b03, s.b02])
    rgb_futures = [executor.submit(_bulk_read, r, xmin, ymin, xmax, ymax)
                   for r in rgb_readers]
    rgb_results = [f.result() for f in rgb_futures]

    # Second pass: SCL resampled onto the per-scene B04 shape.
    scl_futures = []
    for i, s in enumerate(scenes):
        b04 = rgb_results[3 * i + 0][0]
        if b04 is None:
            scl_futures.append(None)
        else:
            scl_futures.append(executor.submit(
                _bulk_read, s.scl, xmin, ymin, xmax, ymax, out_shape=b04.shape,
            ))
    scl_results = [f.result() if f is not None else (None, None) for f in scl_futures]

    out: list[dict | None] = []
    for i in range(len(scenes)):
        b04, t = rgb_results[3 * i + 0]
        b03, _ = rgb_results[3 * i + 1]
        b02, _ = rgb_results[3 * i + 2]
        scl, _ = scl_results[i]
        if any(x is None for x in (b04, b03, b02, scl)):
            out.append(None)
        else:
            out.append({"b04": b04, "b03": b03, "b02": b02, "scl": scl, "transform": t})
    return out


def _build_composite_from_arrays(scene_data: list[dict | None],
                                  utm_x: float, utm_y: float
                                  ) -> np.ndarray | None:
    """Slice 256×256 chips out of the in-memory bulk reads and median-composite.
    Returns (3, IMG_NATIVE, IMG_NATIVE) float32, or None."""
    chips: list[np.ndarray] = []
    for s in scene_data:
        if s is None:
            continue
        # Upper-left UTM corner → (row, col) in the loaded array
        row, col = rowcol(s["transform"], utm_x - HALF_M, utm_y + HALF_M)
        row = int(row)
        col = int(col)
        H, W = s["b04"].shape
        if row < 0 or col < 0 or row + IMG_NATIVE > H or col + IMG_NATIVE > W:
            continue
        b04 = s["b04"][row:row + IMG_NATIVE, col:col + IMG_NATIVE]
        b03 = s["b03"][row:row + IMG_NATIVE, col:col + IMG_NATIVE]
        b02 = s["b02"][row:row + IMG_NATIVE, col:col + IMG_NATIVE]
        scl = s["scl"][row:row + IMG_NATIVE, col:col + IMG_NATIVE]
        ok = ~np.isin(scl, SCL_BAD)
        if ok.sum() < MIN_VALID_PIXELS:
            continue
        rgb = np.stack([b04, b03, b02]).astype(np.float32)
        rgb[:, ~ok] = np.nan
        chips.append(rgb)

    if len(chips) < MIN_VALID_SCENES:
        return None

    stacked = np.stack(chips, axis=0)
    with np.errstate(all="ignore"):
        composite = np.nanmedian(stacked, axis=0)
    for c in range(3):
        col_arr = composite[c]
        nans = np.isnan(col_arr)
        if nans.any():
            fill = np.nanmedian(col_arr) if not np.isnan(col_arr).all() else 0.0
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
    # Cap scenes per shard. The find_s2_scenes fallback pass can produce up
    # to ~16 scenes per remapped MGRS tile, which blows past 16 GB RAM at
    # full 110×110 km bbox. Take the 8 cleanest by cloud cover.
    if len(scenes_df) > MAX_SCENES_PER_SHARD:
        scenes_df = (scenes_df.sort_values("cloud_cover")
                     .head(MAX_SCENES_PER_SHARD).reset_index(drop=True))

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

    io_pool = ThreadPoolExecutor(max_workers=IO_WORKERS)
    t0 = time.time()
    results: list[tuple[str, float, float, float]] = []
    embeddings: list[tuple[str, np.ndarray]] = []
    batch_buf: list[torch.Tensor] = []
    batch_meta: list[tuple[str, float, float]] = []

    # Holder so the inner closures see the current chunk's arrays after each
    # spatial sub-chunk swap.
    scene_data_holder: list[list[dict | None] | None] = [None]

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

    def _prep(ux: float, uy: float):
        comp = _build_composite_from_arrays(scene_data_holder[0], ux, uy)
        if comp is None:
            return None
        return _to_input(comp)

    # Plan spatial sub-chunks so each bulk-load fits the memory budget.
    chunks = _plan_chunks(utm_xy, len(scenes))
    print(f"[infer]   {mgrs_tile}: split into {len(chunks)} sub-chunk(s) "
          f"under {MEMORY_BUDGET_BYTES // 1024**3} GB budget")

    prep_pool = ThreadPoolExecutor(max_workers=PREP_WORKERS)
    skipped = 0
    n = len(grid)
    tids = grid.tile_id.to_numpy()
    lons = grid.lon.to_numpy()
    lats = grid.lat.to_numpy()
    processed = 0
    try:
      for chunk_i, indices in enumerate(chunks):
        chunk_xy = utm_xy[indices]
        bbox = (
            float(chunk_xy[:, 0].min() - HALF_M),
            float(chunk_xy[:, 1].min() - HALF_M),
            float(chunk_xy[:, 0].max() + HALF_M),
            float(chunk_xy[:, 1].max() + HALF_M),
        )
        bbox_w = (bbox[2] - bbox[0]) / 1000
        bbox_h = (bbox[3] - bbox[1]) / 1000
        load_t = time.time()
        scene_data = _load_shard_arrays(scenes, io_pool, bbox)
        n_loaded = sum(1 for s in scene_data if s is not None)
        if n_loaded == 0:
            print(f"[infer]   chunk {chunk_i+1}/{len(chunks)}: no scenes intersect, skipping")
            continue
        mem_mb = sum(s["b04"].nbytes + s["b03"].nbytes + s["b02"].nbytes + s["scl"].nbytes
                     for s in scene_data if s is not None) / 1e6
        print(f"[infer]   chunk {chunk_i+1}/{len(chunks)}: "
              f"{bbox_w:.0f}×{bbox_h:.0f} km, {len(indices):,} tiles, "
              f"loaded {n_loaded}/{len(scenes)} scenes in {time.time()-load_t:.1f}s ({mem_mb:.0f} MB)")
        scene_data_holder[0] = scene_data
        idx_list = list(indices)
        for chunk_start in range(0, len(idx_list), PREP_CHUNK):
            chunk_end = min(chunk_start + PREP_CHUNK, len(idx_list))
            sub_idx = idx_list[chunk_start:chunk_end]
            futs = [
                (j, prep_pool.submit(_prep, float(utm_xy[j, 0]), float(utm_xy[j, 1])))
                for j in sub_idx
            ]
            for j, fut in futs:
                inp = fut.result()
                if inp is None:
                    skipped += 1
                    continue
                batch_buf.append(inp)
                batch_meta.append((str(tids[j]), float(lons[j]), float(lats[j])))
                if len(batch_buf) >= BATCH_SIZE:
                    flush()
            processed += len(sub_idx)
            rate = processed / max(time.time() - t0, 1e-6)
            eta = (n - processed) / max(rate, 1e-6) / 60
            print(f"[infer]   {mgrs_tile} {processed}/{n} "
                  f"({rate:.1f} tiles/s, ~{eta:.1f} min left, skipped={skipped})")
        # Free chunk arrays before next chunk's bulk-load.
        scene_data_holder[0] = None
        del scene_data
      flush()
    finally:
        prep_pool.shutdown(wait=False)
        io_pool.shutdown(wait=False)
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
    """Tune GDAL for COG range-reads from S3.
    Caches kept small — bulk reads only touch each block once per shard, so
    larger caches just eat RAM that we need for the band arrays themselves."""
    return rasterio.Env(
        AWSSession(boto3.Session(), requester_pays=True),
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        GDAL_HTTP_MULTIPLEX="YES",
        GDAL_HTTP_VERSION="2",
        VSI_CACHE="TRUE",
        VSI_CACHE_SIZE=200000000,           # 200 MB per dataset (process-wide)
        CPL_VSIL_CURL_CHUNK_SIZE=1048576,
        CPL_VSIL_CURL_CACHE_SIZE=200000000, # 200 MB total CURL cache
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
