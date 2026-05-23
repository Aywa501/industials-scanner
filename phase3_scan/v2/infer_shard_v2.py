"""GPU worker: score CONUS against dino_vitb v2 probe with Overture pre-filter.

This is the v2 variant of infer_shard.py. Key differences from v1:
  - Model: DINOv3 ViT-B/16 LVD-1689M (768-dim, ImageNet normalization)
  - Probe: dino_vitb trained on OSM building-centric tiles only
  - Pre-filter: skip tiles with no nearby Overture building >= 3000 m^2
    (the v2 probe was NOT trained on rural empty scenes, so we pre-filter
    to building-centric tiles only to avoid applying the model outside its
    training domain)

Per-tile pipeline:
  1. Pre-filter: skip if no Overture building >= 3000 m^2 within 1200m
  2. Project centroid → UTM zone CRS
  3. For each of N scenes (~16): window-read 256×256 of B04, B03, B02, SCL
  4. Mask nodata + cloudy pixels via SCL (classes 0, 1, 3, 8, 9, 10)
  5. Pixel-median across scenes → (3, 256, 256) composite
  6. 1–99 percentile stretch per band → resize 256→224 (LANCZOS) → ImageNet normalize
  7. Batch into the GPU; DINOv3 ViT-B → CLS → linear probe → softmax
  8. Append (tile_id, lon, lat, prob) to the shard's result parquet

Usage:
  python -m phase3_scan.v2.infer_shard_v2 --mgrs 14TMQ
  python -m phase3_scan.v2.infer_shard_v2 --mgrs-list mgrs_todo.txt

Reads:
  data_us/phase3_grid.parquet
  data_us/phase3_scenes.parquet
  data_us/v2/probes/probe_dino_vitb.pt
  data_us/overture_industrial_conus_2025_aligned.parquet (for building pre-filter)
Writes:
  data_us/phase3_results_v2/{mgrs}.parquet
"""

from __future__ import annotations

import argparse
import collections
import math
import os
import queue
import sys
import threading
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
from sklearn.neighbors import BallTree

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
DATA_US = ROOT.parent / "data_us"
GRID_PATH = DATA_US / "phase3_grid.parquet"
SCENES_PATH = DATA_US / "phase3_scenes.parquet"
MODEL_PATH = DATA_US / "v2" / "probes" / "probe_dino_vitb.pt"
OVERTURE_PATH = DATA_US / "overture_industrial_conus_2025_aligned.parquet"
RESULTS_DIR = DATA_US / "phase3_results_v2"

MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"
IMG_NATIVE = 256
IMG_INPUT = 224
GSD_M = 10.0
HALF_M = (IMG_NATIVE / 2) * GSD_M

# 0=no_data, 1=saturated/defective, 3=cloud_shadow, 8/9=cloud, 10=thin_cirrus.
# 0 is load-bearing: an unmasked nodata scene is all-zero, and when nodata
# scenes outnumber real ones the per-scene pixel-median collapses the whole
# composite to a constant.
SCL_BAD = np.array([0, 1, 3, 8, 9, 10], dtype=np.uint8)
MIN_VALID_PIXELS = 256
MIN_VALID_SCENES = 1

BATCH_SIZE = 32
IO_WORKERS = int(os.environ.get("INFER_IO_WORKERS", "32"))
PREP_WORKERS = int(os.environ.get("INFER_PREP_WORKERS", "8"))
# In-flight loaded-array byte budget for the streaming pipeline. The producer
# blocks before starting a new sub-bbox load if the next load would push the
# total beyond this. Set well below RAM to leave headroom for prep / GPU.
MEMORY_BUDGET_BYTES = int(os.environ.get("INFER_MEMORY_BUDGET_GB", "20")) * 1024**3
# Spatial cell width for sub-bbox grouping. Smaller = more sub-bboxes (deeper
# pipeline, smaller bursts); larger = fewer, bigger loads. ~12 km gives ~100 MB
# per scene-bands load on Sentinel-2 at 10 m GSD — well under libcurl HTTP/2
# saturation under 24-way concurrent reads.
SUB_BBOX_KM = float(os.environ.get("INFER_SUB_BBOX_KM", "12"))
# Producer can stage this many sub-bboxes ahead of the consumer. Memory budget
# is the real backpressure; this is just a safety cap.
PIPELINE_DEPTH = int(os.environ.get("INFER_PIPELINE_DEPTH", "8"))
# Number of parallel producer threads. Each runs its own bulk_read concurrently
# (sharing the io_pool + memory budget). With 1 producer, load latency
# serializes; with N, the consumer/GPU sees a steady stream.
PRODUCER_THREADS = int(os.environ.get("INFER_PRODUCER_THREADS", "4"))
# How many sub-bboxes the consumer keeps in prep at once. Higher = better
# prep_pool utilization (small sub-bboxes don't saturate 30 workers alone), at
# the cost of more scene_data held in memory.
PREP_PIPELINE = int(os.environ.get("INFER_PREP_PIPELINE", "4"))
MAX_SCENES_PER_SHARD = 16
EMBED_SAVE_PROB = 0.30

# Overture pre-filter: only process tiles with a building >= this area within this radius
OVERTURE_MIN_AREA_M2 = 3000.0
OVERTURE_RADIUS_M = 1200.0

# ImageNet normalization for dino_vitb
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


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
    t0 = time.time()
    print(f"[infer-v2] loading {MODEL_ID} on {device}", flush=True)
    model = AutoModel.from_pretrained(MODEL_ID, dtype=torch.float16 if device.type == "cuda" else torch.float32)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[infer-v2] backbone loaded: {n_params/1e6:.0f}M params in {time.time()-t0:.1f}s",
          flush=True)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"probe checkpoint not found: {MODEL_PATH}")
    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    emb_dim = ckpt["emb_dim"]  # Read from checkpoint (768 for dino_vitb)
    head = nn.Linear(emb_dim, 2)
    head.load_state_dict(ckpt["state_dict"])
    head.eval().to(device)
    if device.type == "cuda":
        head.half()
    print(f"[infer-v2] probe head loaded: emb_dim={emb_dim}", flush=True)
    return model, head


# ---------------------------------------------------------------------------
# Overture pre-filter
# ---------------------------------------------------------------------------

def _filter_tiles_by_overture(grid_df: pd.DataFrame, overture_df: pd.DataFrame) -> pd.DataFrame:
    """Keep only tiles with an Overture building >= OVERTURE_MIN_AREA_M2 within OVERTURE_RADIUS_M.

    Uses a haversine BallTree query_radius for O((N+M) log M) instead of the
    O(N*M) iterrows + boolean-mask scan it replaces.
    """
    if overture_df.empty:
        print(f"[infer-v2]   no Overture buildings in shard bbox, skipping all tiles")
        return grid_df.iloc[:0]

    overture_df = overture_df[overture_df["approx_area_m2"] >= OVERTURE_MIN_AREA_M2].reset_index(drop=True)
    if overture_df.empty:
        print(f"[infer-v2]   no Overture buildings >= {OVERTURE_MIN_AREA_M2} m^2, skipping all tiles")
        return grid_df.iloc[:0]

    t0 = time.time()
    bldg_rad = np.radians(overture_df[["lat", "lon"]].to_numpy())
    tile_rad = np.radians(grid_df[["lat", "lon"]].to_numpy())
    radius_rad = OVERTURE_RADIUS_M / 6_371_000.0
    tree = BallTree(bldg_rad, metric="haversine")
    counts = tree.query_radius(tile_rad, r=radius_rad, count_only=True)
    keep = counts > 0
    filtered = grid_df.iloc[keep].reset_index(drop=True)
    print(f"[infer-v2]   overture filter: {len(filtered):,}/{len(grid_df):,} tiles kept "
          f"(buildings>={OVERTURE_MIN_AREA_M2:.0f}m²: {len(overture_df):,}, "
          f"radius={OVERTURE_RADIUS_M:.0f}m, {time.time()-t0:.1f}s)")
    return filtered


# ---------------------------------------------------------------------------
# Per-tile read/composite (identical to v1)
# ---------------------------------------------------------------------------

class SceneReaders(NamedTuple):
    scene_id: str
    b04: rasterio.DatasetReader
    b03: rasterio.DatasetReader
    b02: rasterio.DatasetReader
    scl: rasterio.DatasetReader


def _utm_epsg(mgrs_tile: str) -> int:
    zone = int(mgrs_tile[:-3])
    return 32600 + zone


def _plan_sub_bboxes(utm_xy: np.ndarray) -> list[np.ndarray]:
    """Group tiles into ~SUB_BBOX_KM spatial cells for pipelined bulk loads.

    Each entry is an np.array of tile indices (into utm_xy) sharing one cell.
    Sub-bboxes are ordered north→south so successive loads are spatially adjacent
    (better COG-block cache locality)."""
    if len(utm_xy) == 0:
        return []
    cell_m = SUB_BBOX_KM * 1000.0
    cx = np.floor((utm_xy[:, 0] - utm_xy[:, 0].min()) / cell_m).astype(int)
    cy = np.floor((utm_xy[:, 1] - utm_xy[:, 1].min()) / cell_m).astype(int)
    keys = cy * (cx.max() + 1) + cx
    groups = [np.where(keys == k)[0] for k in np.unique(keys)]
    groups.sort(key=lambda idx: -float(utm_xy[idx, 1].mean()))
    return groups


# GDAL/rasterio Dataset objects are not thread-safe for concurrent .read()
# calls on the same Dataset. With >1 producer thread sharing pre-opened
# readers, concurrent reads on the same reader corrupt the heap (glibc
# `realloc(): invalid next size`). Serialize per-Dataset via a lock keyed by
# id(reader); cleared in _close_scenes.
_READ_LOCKS: dict[int, threading.Lock] = {}
_READ_LOCKS_GUARD = threading.Lock()


def _reader_lock(reader) -> threading.Lock:
    rid = id(reader)
    lock = _READ_LOCKS.get(rid)
    if lock is not None:
        return lock
    with _READ_LOCKS_GUARD:
        lock = _READ_LOCKS.get(rid)
        if lock is None:
            lock = threading.Lock()
            _READ_LOCKS[rid] = lock
        return lock


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
            print(f"[infer-v2]   skipping scene {r.scene_id}: {e!r}")
    return out


def _close_scenes(scenes: list[SceneReaders]) -> None:
    for s in scenes:
        for r in (s.b04, s.b03, s.b02, s.scl):
            _READ_LOCKS.pop(id(r), None)
            try:
                r.close()
            except Exception:
                pass


def _bulk_read(reader, xmin: float, ymin: float, xmax: float, ymax: float,
               out_shape: tuple[int, int] | None = None):
    lock = _reader_lock(reader)
    with lock:
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
    xmin, ymin, xmax, ymax = bbox
    rgb_readers = []
    for s in scenes:
        rgb_readers.extend([s.b04, s.b03, s.b02])
    rgb_futures = [executor.submit(_bulk_read, r, xmin, ymin, xmax, ymax)
                   for r in rgb_readers]
    rgb_results = [f.result() for f in rgb_futures]

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
    chips: list[np.ndarray] = []
    for s in scene_data:
        if s is None:
            continue
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
    # Reject a composite with no dynamic range: _to_input's max(hi-lo, 1e-6)
    # guard would silently turn it into an all-zero tensor and yield a
    # constant, meaningless probability. Skip the tile instead.
    for c in range(3):
        lo, hi = np.percentile(composite[c], [1, 99])
        if hi - lo < 1.0:
            return None
    return composite


def _to_input(composite: np.ndarray) -> torch.Tensor:
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

def _sys_telemetry() -> dict:
    """Resource-headroom snapshot for the [stats] line — CPU load, process RSS,
    free system memory, GPU utilisation/memory. Every field degrades to -1 if
    its source is unavailable (e.g. /proc absent on a non-Linux dev box)."""
    t = {"load1": -1.0, "ncpu": os.cpu_count() or 1,
         "rss_gb": -1.0, "sys_avail_gb": -1.0,
         "gpu_util": -1, "gpu_mem_gb": -1.0}
    try:
        t["load1"] = os.getloadavg()[0]
    except (OSError, AttributeError):
        pass
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    t["rss_gb"] = int(line.split()[1]) / 1024 / 1024
                    break
    except OSError:
        pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    t["sys_avail_gb"] = int(line.split()[1]) / 1024 / 1024
                    break
    except OSError:
        pass
    try:
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            t["gpu_mem_gb"] = (total - free) / 1e9
            try:
                t["gpu_util"] = int(torch.cuda.utilization())
            except Exception:
                pass
    except Exception:
        pass
    return t


class StreamStats:
    """Per-stage throughput counters. A background thread prints rolling
    deltas every `interval` seconds so we can see which stage is bottlenecking.

    Stages instrumented:
      - IO   : bytes loaded by producers from S3 (after rasterio bulk_read)
      - PREP : composites/tile prep completions (per _prep future)
      - GPU  : tiles forwarded through model + head (per flush call)
    """

    def __init__(self, interval: float = 5.0):
        self.interval = interval
        self.lock = threading.Lock()
        self.bytes_loaded = 0
        self.composites_done = 0
        self.gpu_tiles = 0
        self.gpu_seconds = 0.0
        self.gpu_batches = 0
        self.stop_event = threading.Event()
        self.last = time.time()
        self.snap = (0, 0, 0, 0.0, 0)
        self.ready_q = None
        self.batch_q = None
        self.in_flight_subs = None
        self.state = None
        self.peak_io = 0.0
        self.peak_prep = 0.0
        self.peak_gpu = 0.0

    def record_bytes(self, n: int):
        with self.lock:
            self.bytes_loaded += n

    def record_composite(self):
        with self.lock:
            self.composites_done += 1

    def record_gpu(self, n_tiles: int, seconds: float):
        with self.lock:
            self.gpu_tiles += n_tiles
            self.gpu_seconds += seconds
            self.gpu_batches += 1

    def _tick(self):
        now = time.time()
        dt = max(now - self.last, 1e-6)
        with self.lock:
            cur = (self.bytes_loaded, self.composites_done, self.gpu_tiles,
                   self.gpu_seconds, self.gpu_batches)
        db = cur[0] - self.snap[0]
        dc = cur[1] - self.snap[1]
        dg = cur[2] - self.snap[2]
        dgs = cur[3] - self.snap[3]
        dgb = cur[4] - self.snap[4]
        self.snap = cur
        self.last = now
        rq = self.ready_q.qsize() if self.ready_q is not None else -1
        bq = self.batch_q.qsize() if self.batch_q is not None else -1
        ifs = len(self.in_flight_subs) if self.in_flight_subs is not None else -1
        inf_gb = (self.state["in_flight"] / 1e9) if self.state is not None else 0.0
        batch_ms = (dgs / dgb * 1000.0) if dgb > 0 else 0.0
        io_mbs = db / 1e6 / dt
        prep_tps = dc / dt
        gpu_tps = dg / dt
        self.peak_io = max(self.peak_io, io_mbs)
        self.peak_prep = max(self.peak_prep, prep_tps)
        self.peak_gpu = max(self.peak_gpu, gpu_tps)
        # Bottleneck attribution from queue occupancy: a near-full batch_q means
        # the GPU is the limiter; a near-empty ready_q means prep is starved of
        # raw sub-bbox IO; prep saturated with a low batch_q means prep itself.
        if bq >= 3:
            bottleneck = "GPU"
        elif 0 <= rq <= 1:
            bottleneck = "IO  (prep starved)"
        elif ifs >= PREP_PIPELINE and bq <= 1:
            bottleneck = "PREP"
        else:
            bottleneck = "~balanced"
        st = _sys_telemetry()
        print(f"[stats] IO: {io_mbs:5.0f} MB/s | PREP: {prep_tps:5.1f} t/s | "
              f"GPU: {gpu_tps:5.1f} t/s ({batch_ms:4.0f}ms/batch x{dgb}) | "
              f"ready_q={rq:2d}/{PIPELINE_DEPTH} prep={ifs:2d}/{PREP_PIPELINE} "
              f"batch_q={bq}/4 in_flight={inf_gb:4.1f}GB",
              flush=True)
        print(f"[stats] cpu={st['load1']:5.1f}/{st['ncpu']:<3d} "
              f"rss={st['rss_gb']:4.1f}G sysfree={st['sys_avail_gb']:6.1f}G "
              f"gpu={st['gpu_util']:3d}%/{st['gpu_mem_gb']:4.1f}G | "
              f"peaks IO={self.peak_io:4.0f} PREP={self.peak_prep:5.1f} "
              f"GPU={self.peak_gpu:5.1f} | BOTTLENECK={bottleneck}",
              flush=True)

    def run(self):
        while not self.stop_event.wait(self.interval):
            self._tick()


def process_shard(mgrs_tile: str, model, head, device: torch.device) -> Path:
    shard_start = time.time()
    out_path = RESULTS_DIR / f"{mgrs_tile}.parquet"
    if out_path.exists():
        print(f"[infer-v2] {mgrs_tile}: result already exists, skipping")
        return out_path

    print(f"[infer-v2] === {mgrs_tile} start ({time.strftime('%Y-%m-%d %H:%M:%S')}) ===",
          flush=True)
    grid = pd.read_parquet(GRID_PATH)
    grid = grid[grid.mgrs_tile == mgrs_tile].reset_index(drop=True)
    if grid.empty:
        print(f"[infer-v2] {mgrs_tile}: no tiles in grid, skipping")
        return None
    print(f"[infer-v2] {mgrs_tile}: {len(grid):,} grid tiles before Overture filter", flush=True)

    print(f"[infer-v2] {mgrs_tile}: loading Overture buildings...", flush=True)
    bbox = (grid.lon.min() - 0.1, grid.lat.min() - 0.1,
            grid.lon.max() + 0.1, grid.lat.max() + 0.1)
    try:
        overture = pd.read_parquet(
            OVERTURE_PATH,
            filters=[
                ("lon", ">=", bbox[0]),
                ("lon", "<=", bbox[2]),
                ("lat", ">=", bbox[1]),
                ("lat", "<=", bbox[3]),
            ],
            columns=["lon", "lat", "approx_area_m2"]
        )
    except Exception as e:
        print(f"[infer-v2] {mgrs_tile}: failed to load Overture: {e!r}, skipping shard")
        return None
    print(f"[infer-v2] {mgrs_tile}: {len(overture):,} Overture buildings in bbox", flush=True)

    grid = _filter_tiles_by_overture(grid, overture)
    if grid.empty:
        print(f"[infer-v2] {mgrs_tile}: no tiles with nearby buildings, skipping")
        return None

    scenes_df = pd.read_parquet(SCENES_PATH)
    scenes_df = scenes_df[scenes_df.mgrs_tile == mgrs_tile]
    if scenes_df.empty:
        print(f"[infer-v2] {mgrs_tile}: no scenes, skipping")
        return None
    if len(scenes_df) > MAX_SCENES_PER_SHARD:
        scenes_df = (scenes_df.sort_values("cloud_cover")
                     .head(MAX_SCENES_PER_SHARD).reset_index(drop=True))

    print(f"[infer-v2] {mgrs_tile}: {len(grid):,} tiles × {len(scenes_df)} scenes "
          f"(setup {time.time() - shard_start:.1f}s)", flush=True)

    epsg = _utm_epsg(mgrs_tile)
    to_utm = pyproj.Transformer.from_crs(4326, epsg, always_xy=True).transform
    utm_xy = np.column_stack(to_utm(grid.lon.to_numpy(), grid.lat.to_numpy()))

    mean_t = MEAN.to(device, dtype=torch.float16 if device.type == "cuda" else torch.float32)
    std_t = STD.to(device, dtype=torch.float16 if device.type == "cuda" else torch.float32)

    scenes = _open_scene_readers(scenes_df)
    if not scenes:
        print(f"[infer-v2] {mgrs_tile}: failed to open any scene, skipping")
        return None

    io_pool = ThreadPoolExecutor(max_workers=IO_WORKERS)
    prep_pool = ThreadPoolExecutor(max_workers=PREP_WORKERS)
    t0 = time.time()
    results: list[tuple[str, float, float, float]] = []
    embeddings: list[tuple[str, np.ndarray]] = []
    batch_buf: list[torch.Tensor] = []
    batch_meta: list[tuple[str, float, float]] = []
    stats = StreamStats(interval=5.0)
    # Batch queue: consumer (prep) thread pushes ready batches; gpu_worker
    # drains them. Decoupling lets GPU forward and prep run concurrently.
    batch_q: queue.Queue = queue.Queue(maxsize=4)

    def flush():
        if not batch_buf:
            return
        # Avoid deadlock if gpu_worker died: put with timeout, then check err.
        while True:
            if state.get("err") is not None or stop_event.is_set():
                batch_buf.clear()
                batch_meta.clear()
                return
            try:
                batch_q.put((batch_buf[:], batch_meta[:]), timeout=5.0)
                break
            except queue.Full:
                continue
        batch_buf.clear()
        batch_meta.clear()

    def gpu_worker():
        try:
            while True:
                item = batch_q.get()
                if item is None:
                    return
                b_buf, b_meta = item
                gpu_t0 = time.perf_counter()
                x = torch.stack(b_buf).to(device)
                if device.type == "cuda":
                    x = x.half()
                x = (x - mean_t) / std_t
                with torch.inference_mode():
                    cls = model(x).last_hidden_state[:, 0, :]
                    logits = head(cls)
                    probs = F.softmax(logits, dim=1)[:, 1].float().cpu().numpy()
                cls_np = cls.float().cpu().numpy()
                if device.type == "cuda":
                    torch.cuda.synchronize()
                stats.record_gpu(len(b_buf), time.perf_counter() - gpu_t0)
                for (tid, lon, lat), p, emb in zip(b_meta, probs, cls_np):
                    results.append((tid, lon, lat, float(p)))
                    if p > EMBED_SAVE_PROB:
                        embeddings.append((tid, emb.astype(np.float16)))
        except Exception as e:
            print(f"[infer-v2] GPU worker FATAL: {e!r}", flush=True)
            with state_lock:
                if state["err"] is None:
                    state["err"] = e
            stop_event.set()
            # Drain remaining items so producer/consumer don't deadlock on a
            # full batch_q. Re-raise via state["err"] is handled in main loop.
            try:
                while True:
                    item = batch_q.get_nowait()
                    if item is None:
                        return
            except queue.Empty:
                return

    def _prep(scene_data, ux: float, uy: float):
        comp = _build_composite_from_arrays(scene_data, ux, uy)
        if comp is None:
            return None
        return _to_input(comp)

    sub_bboxes = _plan_sub_bboxes(utm_xy)
    n_producers = max(1, min(PRODUCER_THREADS, len(sub_bboxes)))
    print(f"[infer-v2]   {mgrs_tile}: {len(sub_bboxes)} sub-bbox(es), "
          f"pipeline depth={PIPELINE_DEPTH}, {n_producers} producers, "
          f"mem budget={MEMORY_BUDGET_BYTES // 1024**3} GB")

    # Producer / consumer pipeline: N producer threads pull sub-bboxes from
    # work_q and each issues its own bulk_read (sharing io_pool + memory budget),
    # pushing ready tuples to ready_q. The main thread (consumer) drains the
    # queue, runs prep_pool composites, and flushes batches to GPU. Memory
    # budget is the cross-thread backpressure: producers block before charging
    # a new load if it would push in-flight bytes over budget; consumer releases
    # after each sub-bbox is processed.
    work_q: queue.Queue = queue.Queue()
    for sub_indices in sub_bboxes:
        work_q.put(sub_indices)
    for _ in range(n_producers):
        work_q.put(None)

    ready_q: queue.Queue = queue.Queue(maxsize=PIPELINE_DEPTH)
    state = {"in_flight": 0, "err": None, "active": n_producers}
    budget_cv = threading.Condition()
    state_lock = threading.Lock()
    stop_event = threading.Event()

    def producer():
        try:
            while not stop_event.is_set():
                try:
                    sub_indices = work_q.get(timeout=5.0)
                except queue.Empty:
                    continue
                if sub_indices is None:
                    break
                sub_xy = utm_xy[sub_indices]
                bbox = (
                    float(sub_xy[:, 0].min() - HALF_M),
                    float(sub_xy[:, 1].min() - HALF_M),
                    float(sub_xy[:, 0].max() + HALF_M),
                    float(sub_xy[:, 1].max() + HALF_M),
                )
                w_px = (bbox[2] - bbox[0]) / GSD_M
                h_px = (bbox[3] - bbox[1]) / GSD_M
                est_bytes = int(w_px * h_px * len(scenes) * 4 * 2)
                with budget_cv:
                    while (state["in_flight"] + est_bytes > MEMORY_BUDGET_BYTES
                           and state["in_flight"] > 0
                           and not stop_event.is_set()):
                        budget_cv.wait(timeout=5.0)
                    if stop_event.is_set():
                        break
                    state["in_flight"] += est_bytes
                load_t = time.time()
                scene_data = _load_shard_arrays(scenes, io_pool, bbox)
                actual = sum(
                    s["b04"].nbytes + s["b03"].nbytes + s["b02"].nbytes + s["scl"].nbytes
                    for s in scene_data if s is not None
                )
                with budget_cv:
                    state["in_flight"] += actual - est_bytes
                    budget_cv.notify_all()
                stats.record_bytes(actual)
                ready_q.put((sub_indices, scene_data, actual, bbox, time.time() - load_t))
        except Exception as e:
            with state_lock:
                if state["err"] is None:
                    state["err"] = e
            stop_event.set()
        finally:
            with state_lock:
                state["active"] -= 1
                last = state["active"] == 0
            if last:
                ready_q.put(None)

    producer_threads = [
        threading.Thread(target=producer, name=f"loader-{i}", daemon=True)
        for i in range(n_producers)
    ]
    for t in producer_threads:
        t.start()

    # in_flight_subs: sub-bboxes whose prep futures are running but not yet
    # drained. Holds up to PREP_PIPELINE entries so prep_pool stays saturated.
    in_flight_subs: collections.deque = collections.deque()

    stats.ready_q = ready_q
    stats.batch_q = batch_q
    stats.in_flight_subs = in_flight_subs
    stats.state = state
    stats_thread = threading.Thread(target=stats.run, name="stats", daemon=True)
    stats_thread.start()

    # GPU worker drains batch_q in parallel with prep. Started before consumer
    # so the first flush() can hand off immediately.
    gpu_thread = threading.Thread(target=gpu_worker, name="gpu", daemon=True)
    gpu_thread.start()

    skipped = 0
    n = len(grid)
    tids = grid.tile_id.to_numpy()
    lons = grid.lon.to_numpy()
    lats = grid.lat.to_numpy()
    processed = 0
    sub_done = 0
    eos = False
    try:
        while True:
            # Top up in_flight_subs by reading from ready_q. Block only when
            # the pipeline is empty; otherwise drain pending work first.
            while not eos and len(in_flight_subs) < PREP_PIPELINE:
                if in_flight_subs:
                    try:
                        item = ready_q.get_nowait()
                    except queue.Empty:
                        break
                else:
                    item = ready_q.get()
                if item is None:
                    eos = True
                    break
                sub_indices, scene_data, actual_bytes, bbox, load_s = item
                sub_done += 1
                n_loaded = sum(1 for s in scene_data if s is not None)
                bbox_w = (bbox[2] - bbox[0]) / 1000
                bbox_h = (bbox[3] - bbox[1]) / 1000
                if n_loaded == 0:
                    print(f"[infer-v2]   sub {sub_done}/{len(sub_bboxes)}: "
                          f"{bbox_w:.0f}×{bbox_h:.0f} km, no scenes intersect, skipping")
                    with budget_cv:
                        state["in_flight"] -= actual_bytes
                        budget_cv.notify_all()
                    continue
                futs = [
                    (j, prep_pool.submit(_prep, scene_data,
                                         float(utm_xy[j, 0]), float(utm_xy[j, 1])))
                    for j in sub_indices
                ]
                in_flight_subs.append({
                    "sub_id": sub_done,
                    "sub_indices": sub_indices,
                    "scene_data": scene_data,
                    "actual_bytes": actual_bytes,
                    "bbox_w": bbox_w,
                    "bbox_h": bbox_h,
                    "n_loaded": n_loaded,
                    "load_s": load_s,
                    "futs": futs,
                })

            if not in_flight_subs:
                break

            sub = in_flight_subs.popleft()
            for j, fut in sub["futs"]:
                inp = fut.result()
                stats.record_composite()
                if inp is None:
                    skipped += 1
                    continue
                batch_buf.append(inp)
                batch_meta.append((str(tids[j]), float(lons[j]), float(lats[j])))
                if len(batch_buf) >= BATCH_SIZE:
                    flush()
            processed += len(sub["sub_indices"])
            rate = processed / max(time.time() - t0, 1e-6)
            eta = (n - processed) / max(rate, 1e-6) / 60
            print(f"[infer-v2]   sub {sub['sub_id']}/{len(sub_bboxes)}: "
                  f"{sub['bbox_w']:.0f}×{sub['bbox_h']:.0f} km, {len(sub['sub_indices'])} tiles, "
                  f"{sub['n_loaded']}/{len(scenes)} scenes ({sub['actual_bytes']/1e6:.0f} MB, "
                  f"load {sub['load_s']:.1f}s) → {processed}/{n} ({rate:.1f} t/s, "
                  f"~{eta:.1f} min, skip={skipped}, in_flight={state['in_flight']/1e9:.1f}GB)")
            with budget_cv:
                state["in_flight"] -= sub["actual_bytes"]
                budget_cv.notify_all()
            del sub
        flush()
        if state["err"] is not None:
            raise state["err"]
    finally:
        stop_event.set()
        stats.stop_event.set()
        # Drain any remaining queue items so producers can exit cleanly.
        try:
            while True:
                work_q.get_nowait()
        except queue.Empty:
            pass
        # Wake any producer stuck in budget_cv.wait() so they observe stop_event.
        with budget_cv:
            budget_cv.notify_all()
        # Signal GPU worker to finish its queue and exit.
        try:
            batch_q.put(None, timeout=5.0)
        except queue.Full:
            pass
        gpu_thread.join(timeout=30)
        if gpu_thread.is_alive():
            print(f"[infer-v2] {mgrs_tile}: WARN gpu_thread did not exit in 30s", flush=True)
        prep_pool.shutdown(wait=False)
        io_pool.shutdown(wait=False)
        _close_scenes(scenes)
        for t in producer_threads:
            t.join(timeout=5)
            if t.is_alive():
                print(f"[infer-v2] {mgrs_tile}: WARN {t.name} did not exit in 5s", flush=True)
        stats_thread.join(timeout=5)

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
    print(f"[infer-v2] {mgrs_tile} done: {len(out_df):,} probs "
          f"({len(embeddings):,} embeddings saved) in {dt:.1f}s "
          f"({len(grid) / max(dt, 1e-6):.1f} tiles/s, skipped={skipped}) → {out_path}")
    return out_path


def _setup_rasterio_env() -> rasterio.Env:
    return rasterio.Env(
        AWSSession(boto3.Session(), requester_pays=True),
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        GDAL_HTTP_MULTIPLEX="YES",
        GDAL_HTTP_VERSION="2",
        VSI_CACHE="TRUE",
        VSI_CACHE_SIZE=200000000,
        CPL_VSIL_CURL_CHUNK_SIZE=1048576,
        CPL_VSIL_CURL_CACHE_SIZE=200000000,
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

    import traceback
    with _setup_rasterio_env():
        for t in tiles:
            try:
                process_shard(t, model, head, device)
            except Exception as e:
                print(f"[infer-v2] {t} FAILED: {e!r}", flush=True)
                traceback.print_exc()


if __name__ == "__main__":
    main()
