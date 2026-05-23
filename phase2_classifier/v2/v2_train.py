"""End-to-end v2 detector training on a GPU spot box — multi-model comparison.

Tested on g6.2xlarge (L4) and g4dn.2xlarge (T4). Bandwidth-bound, so the
cheaper T4 box matches L4 wall-clock — see launch_v2.md for instance picking.

Reads `data_us/v2_dataset_manifest.parquet` and produces, for each model M in
the registry:
  - data_us/v2/emb_<M>.npy             (fp16, N x emb_dim_M)
  - data_us/v2/probes/probe_<M>.pt
  - data_us/v2/probes/probe_<M>_train.json
  - data_us/v2/probes/probe_<M>_eval.json
plus a shared:
  - data_us/v2/v2_embeddings_index.parquet  (one row per tile, alignment for all emb_*.npy)
  - data_us/v2/leaderboard.json             (cross-model summary)

Pipeline:
  * Group manifest rows by (mgrs_tile, target_year). STAC search Element84 for
    cleanest 8 Sentinel-2 L2A scenes per group within May–Sep.
  * DBSCAN-cluster tiles within each group (eps=5km) so each chunk's bbox is
    tight (singleton tiles get their own 2.56km bbox).
  * Per cluster: bulk-read 6 bands (B02 B03 B04 B8A B11 B12) + SCL via rasterio
    window reads from S3 COGs. Resample 20m bands (B8A/B11/B12) to the 10m grid.
  * Per tile: pixel-median composite across valid scenes (SCL-masked) → (6,256,256).
  * Run all registered models on the same composite (each with its own
    normalization + input prep) and append per-model embedding to its buffer.

Usage on EC2:
  python v2_train.py                # full pipeline
  python v2_train.py --skip-embed   # train probes on existing embeddings
  python v2_train.py --models dino_sat493m,prithvi_300m  # subset
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import resource
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3
import mgrs as mgrs_lib
import numpy as np
import pandas as pd
import pyproj
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
from PIL import Image
from rasterio.session import AWSSession
from rasterio.transform import rowcol
from rasterio.windows import Window, from_bounds
from sklearn.cluster import DBSCAN
from transformers import AutoModel

ROOT = Path(__file__).resolve().parents[3]
DATA_US = ROOT / "data_us"
MANIFEST = DATA_US / "v2_dataset_manifest.parquet"
SCENES_INDEX = DATA_US / "v2_scenes_index.parquet"

# v2.1 outputs go under data_us/v2/
V2_DIR = DATA_US / "v2"
V2_DIR.mkdir(parents=True, exist_ok=True)
PROBES_DIR = V2_DIR / "probes"
PROBES_DIR.mkdir(parents=True, exist_ok=True)
EMB_IDX_OUT = V2_DIR / "v2_embeddings_index.parquet"
LEADERBOARD = V2_DIR / "leaderboard.json"
EMBED_CHUNKS_DIR = V2_DIR / "embed_chunks"   # per-group resume artifacts
STATS_LOG = V2_DIR / "stats.jsonl"            # one structured line per group: per-stage timings, rates, RSS


N_CLASSES = 2  # binary: 0=non, 1=industrial. Manifest class_id 2 is mapped to label 1.
IMG_NATIVE = 256
IMG_INPUT = 224
GSD_M = 10.0
HALF_M = (IMG_NATIVE / 2) * GSD_M
SCL_BAD = np.array([3, 8, 9, 10], dtype=np.uint8)
MIN_VALID_PIXELS = 256
BATCH_SIZE = 32
IO_WORKERS = 96           # bumped from 32 — py-spy showed only ~6-8 of 32 io threads active per group
PREP_WORKERS = 8          # parallel per-tile composite + prep (GIL-released numpy/PIL)
PREP_CHUNK = 256          # bound pending-futures pile in the prep_pool
PREFETCH_GROUPS = 2       # (mgrs_tile, year) groups to fetch ahead in parallel; lifts S3 connection count
MEMORY_BUDGET_BYTES = 16 * 1024**3   # per-sub-chunk bulk-read budget. Only ONE sub_chunk is loaded at a time across the whole process (prefetch only opens lightweight scene readers, doesn't preload arrays). Peak host RAM ≈ this + ~5 GB Python/PyTorch overhead. 16 GB on a 32 GB box leaves comfortable headroom; v1 used 20 GB.
CLUSTER_EPS_M = 5000.0

# 6 S2 bands fetched (HLS-style, matches Prithvi expectations).
# RGB models pull B02/B03/B04 (idx 0/1/2 in this order).
BANDS = ["B02", "B03", "B04", "B8A", "B11", "B12"]
BAND_ASSETS = {
    "B02": "blue",
    "B03": "green",
    "B04": "red",
    "B8A": "nir08",
    "B11": "swir16",
    "B12": "swir22",
}

# ---------- Normalization stats ---------------------------------------------

# DINOv3 SAT-493M: ImageNet-style 0..1 inputs with their own mean/std.
DINO_SAT_MEAN = [0.430, 0.411, 0.296]
DINO_SAT_STD  = [0.213, 0.156, 0.143]

# DINOv3 ViT-B / ResNet-50: standard ImageNet stats on 0..1 inputs.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Prithvi-EO-2.0: per-band mean/std from HLS pretraining (reflectance scale 0..1
# after dividing S2 L2A integer reflectance by 10000). Order: B02 B03 B04 B8A B11 B12.
PRITHVI_MEAN = [0.0473, 0.0518, 0.0571, 0.2299, 0.1786, 0.1064]
PRITHVI_STD  = [0.0252, 0.0270, 0.0354, 0.0846, 0.0887, 0.0712]

S2_REFLECTANCE_SCALE = 10000.0  # divide raw L2A int → reflectance


# ---------------------------------------------------------------------------
# Per-model load + forward
# ---------------------------------------------------------------------------

def _to_device(model, device):
    if device.type == "cuda":
        model = model.half()
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_dinov3(spec, device):
    m = AutoModel.from_pretrained(spec["hf_id"],
                                  dtype=torch.float16 if device.type == "cuda" else torch.float32)
    return _to_device(m, device)


def load_resnet50(spec, device):
    m = tvm.resnet50(weights="IMAGENET1K_V2")
    m.fc = nn.Identity()
    return _to_device(m, device)


def load_prithvi(spec, device):
    """Prithvi-EO-2.0 is registered under terratorch's BACKBONE_REGISTRY (not transformers)."""
    import terratorch  # noqa: F401  (registers backbones via side-effect import)
    from terratorch.registry import BACKBONE_REGISTRY
    m = BACKBONE_REGISTRY.build(spec["hf_id"], pretrained=True)
    return _to_device(m, device)


def forward_dinov3(model, x):
    return model(x).last_hidden_state[:, 0, :]  # CLS token


def forward_resnet50(model, x):
    return model(x)  # already (B, 2048) since fc=Identity


def forward_prithvi(model, x):
    """terratorch's PrithviViT returns a list of layer hidden states (B, T, D); take last-layer CLS."""
    out = model(x)
    if isinstance(out, (list, tuple)):
        h = out[-1]
    elif hasattr(out, "last_hidden_state"):
        h = out.last_hidden_state
    else:
        h = out
    return h[:, 0, :] if h.dim() == 3 else h


# ---------- Model registry ---------------------------------------------------

MODEL_REGISTRY = [
    {"name": "dino_sat493m", "hf_id": "facebook/dinov3-vitl16-pretrain-sat493m",
     "input": "rgb_dinosat",   "emb_dim": 1024,
     "load_fn": load_dinov3,   "forward_fn": forward_dinov3},
    {"name": "dino_vitb",      "hf_id": "facebook/dinov3-vitb16-pretrain-lvd1689m",
     "input": "rgb_imagenet",  "emb_dim": 768,
     "load_fn": load_dinov3,   "forward_fn": forward_dinov3},
    {"name": "resnet50",       "hf_id": None,
     "input": "rgb_imagenet",  "emb_dim": 2048,
     "load_fn": load_resnet50, "forward_fn": forward_resnet50},
    {"name": "prithvi_300m",   "hf_id": "prithvi_eo_v2_300",
     "input": "prithvi_6band", "emb_dim": 1024,
     "load_fn": load_prithvi,  "forward_fn": forward_prithvi},
    {"name": "prithvi_600m",   "hf_id": "prithvi_eo_v2_600",
     "input": "prithvi_6band", "emb_dim": 1280,
     "load_fn": load_prithvi,  "forward_fn": forward_prithvi},
]


def load_models(device, names):
    """Returns dict name → (model, spec). Skips any that fail to load."""
    loaded = {}
    for spec in MODEL_REGISTRY:
        if spec["name"] not in names:
            continue
        print(f"[v2-train] loading {spec['name']} ({spec['hf_id']})")
        try:
            m = spec["load_fn"](spec, device)
            loaded[spec["name"]] = (m, spec)
        except Exception as e:
            print(f"[v2-train]   {spec['name']} load failed: {e!r}; skipping")
    print(f"[v2-train] models active: {list(loaded.keys())}")
    return loaded


# ---------------------------------------------------------------------------
# STAC + Sentinel-2 fetch
# ---------------------------------------------------------------------------

def _utm_epsg(mgrs_tile: str) -> int:
    return 32600 + int(mgrs_tile[:-3])


def _https_to_s3(href: str) -> str:
    from urllib.parse import urlparse
    p = urlparse(href)
    bucket = p.netloc.split(".")[0]
    return f"s3://{bucket}{p.path}"


def compute_mgrs(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    m = mgrs_lib.MGRS()
    out = np.empty(len(lats), dtype=object)
    for i, (la, lo) in enumerate(zip(lats, lons)):
        try:
            full = m.toMGRS(float(la), float(lo), MGRSPrecision=0)
            out[i] = full[:5]
        except Exception:
            out[i] = None
    return out


def open_scene_readers(scene_rows: list[dict], executor):
    """Parallelize rasterio.open across all (scene, band) URIs through the
    shared I/O pool. 8 scenes × 7 reads = 56 sequential opens at ~200 ms each
    was the dominant per-group bottleneck before this change."""
    keys = ["B02", "B03", "B04", "B8A", "B11", "B12", "scl"]
    futs = []
    for r in scene_rows:
        for k in keys:
            href = r["scl_s3"] if k == "scl" else r[f"{k}_s3"]
            futs.append(executor.submit(rasterio.open, href))
    out = []
    for i, r in enumerate(scene_rows):
        readers = {}
        ok = True
        err = None
        for j, k in enumerate(keys):
            try:
                readers[k] = futs[i * len(keys) + j].result()
            except Exception as e:
                ok = False
                err = e
                break
        if ok:
            out.append(readers)
        else:
            for v in readers.values():
                try: v.close()
                except Exception: pass
            print(f"[v2-train]   skipping scene {r['scene_id']}: {err!r}")
    return out


def close_scenes(scenes):
    for s in scenes:
        for r in s.values():
            try:
                r.close()
            except Exception:
                pass


def _window_for_bbox(reader, xmin, ymin, xmax, ymax):
    """Compute the rasterio Window covering bbox in `reader`'s pixel grid,
    clipped to dataset extent. Pure metadata math — no data read.

    Used by BOTH `bulk_read` (to actually read) and `load_group_arrays`
    (to predict per-scene target shapes upfront so all reads can be issued
    in a single wave). They MUST agree on rounding/clipping or 20m bands
    won't align to the 10m grid for scenes near the edge."""
    win = from_bounds(xmin, ymin, xmax, ymax, transform=reader.transform)
    col = max(0, int(round(win.col_off)))
    row = max(0, int(round(win.row_off)))
    col_end = min(reader.width, int(round(win.col_off + win.width)))
    row_end = min(reader.height, int(round(win.row_off + win.height)))
    if col >= col_end or row >= row_end:
        return None
    return Window(col, row, col_end - col, row_end - row)


def bulk_read(reader, xmin, ymin, xmax, ymax, out_shape=None):
    actual = _window_for_bbox(reader, xmin, ymin, xmax, ymax)
    if actual is None:
        return None, None
    native_tr = reader.window_transform(actual)
    if out_shape is None:
        return reader.read(1, window=actual), native_tr
    arr = reader.read(1, window=actual, out_shape=out_shape,
                      resampling=rasterio.enums.Resampling.bilinear)
    sx = actual.width / out_shape[1]
    sy = actual.height / out_shape[0]
    return arr, native_tr * native_tr.scale(sx, sy)


def load_group_arrays(scenes, executor, bbox):
    """Single-wave fetch of all 56 reads (8 scenes × 7 bands).

    Old version did two waves: read anchor bands → wait → use shapes to
    submit resampled-band reads → wait. py-spy showed io_pool draining
    between the two waves while main computed target shapes; each barrier
    cost ~max(read latency in wave) ≈ 1–2s of pure idle.

    New version computes per-scene target shapes from B02 reader metadata
    (no data read) using `_window_for_bbox` — same math `bulk_read` uses
    internally — then submits all 56 reads at once and waits for the
    slowest single read instead of slowest-of-24 + slowest-of-32.
    """
    target_shapes = []  # per scene: (height, width) for resample target, or None if scene doesn't intersect bbox
    for s in scenes:
        win = _window_for_bbox(s["B02"], *bbox)
        target_shapes.append((win.height, win.width) if win is not None else None)

    # Submit all reads in one wave. For scenes with no overlap, skip 20m bands entirely.
    by_scene_band: dict[tuple[int, str], object] = {}  # (scene_idx, band_key) -> future or None
    for i, s in enumerate(scenes):
        for b in ("B02", "B03", "B04"):
            by_scene_band[(i, b)] = executor.submit(bulk_read, s[b], *bbox)
        ts = target_shapes[i]
        for b in ("B8A", "B11", "B12", "scl"):
            if ts is None:
                by_scene_band[(i, b)] = None
            else:
                by_scene_band[(i, b)] = executor.submit(bulk_read, s[b], *bbox, ts)

    # Drain — wait for the slowest single read across all 56
    out = []
    for i in range(len(scenes)):
        bands = {}
        transform = None
        for b in ("B02", "B03", "B04", "B8A", "B11", "B12", "scl"):
            fut = by_scene_band[(i, b)]
            if fut is None:
                bands[b] = None
            else:
                arr, tr = fut.result()
                bands[b] = arr
                if b == "B02":
                    transform = tr
        if any(bands[b] is None for b in ("B02", "B03", "B04", "B8A", "B11", "B12", "scl")):
            out.append(None)
        else:
            bands["transform"] = transform
            out.append(bands)
    return out


def build_composite(scene_data, utm_x, utm_y):
    """Build a (6, IMG_NATIVE, IMG_NATIVE) per-pixel-median composite across
    valid scenes. Returns reflectance values in [0,1] approximately (raw / 10000)."""
    chips = []
    for s in scene_data:
        if s is None:
            continue
        row, col = rowcol(s["transform"], utm_x - HALF_M, utm_y + HALF_M)
        row, col = int(row), int(col)
        H, W = s["B02"].shape
        if row < 0 or col < 0 or row + IMG_NATIVE > H or col + IMG_NATIVE > W:
            continue
        arrs = [s[b][row:row+IMG_NATIVE, col:col+IMG_NATIVE] for b in BANDS]
        scl = s["scl"][row:row+IMG_NATIVE, col:col+IMG_NATIVE]
        ok = ~np.isin(scl, SCL_BAD)
        if ok.sum() < MIN_VALID_PIXELS:
            continue
        chip = np.stack(arrs).astype(np.float32) / S2_REFLECTANCE_SCALE
        chip[:, ~ok] = np.nan
        chips.append(chip)
    if not chips:
        return None
    stacked = np.stack(chips, axis=0)
    with np.errstate(all="ignore"):
        comp = np.nanmedian(stacked, axis=0)
    for c in range(comp.shape[0]):
        nans = np.isnan(comp[c])
        if nans.any():
            fill = np.nanmedian(comp[c]) if not np.isnan(comp[c]).all() else 0.0
            comp[c, nans] = fill
    return comp  # shape (6, 256, 256), reflectance ~[0,1]


# ---------------------------------------------------------------------------
# Per-model input preparation
# ---------------------------------------------------------------------------

def prep_rgb_dinosat(comp):
    """RGB percentile-stretch + LANCZOS 256→224 + normalize with DINO-SAT stats."""
    rgb = comp[:3].copy()  # B02, B03, B04 in fetch order (blue, green, red)
    # DINOv3-SAT was trained on R,G,B order; we need to swap to (R,G,B)
    rgb = rgb[::-1].copy()
    out = np.empty_like(rgb, dtype=np.float32)
    for c in range(3):
        lo, hi = np.percentile(rgb[c], [1, 99])
        out[c] = np.clip((rgb[c] - lo) / max(hi - lo, 1e-6), 0, 1)
    arr_u8 = (out.transpose(1, 2, 0) * 255).astype(np.uint8)
    img = Image.fromarray(arr_u8).resize((IMG_INPUT, IMG_INPUT), Image.LANCZOS)
    return torch.from_numpy(np.asarray(img).astype(np.float32).transpose(2, 0, 1) / 255.0)


def prep_prithvi_6band(comp):
    """6-band reflectance, resized to IMG_INPUT, no percentile stretch (Prithvi expects
    raw reflectance scale); per-band normalization applied at flush."""
    out = np.empty((6, IMG_INPUT, IMG_INPUT), dtype=np.float32)
    for c in range(6):
        # Bilinear resize 256→224
        img = Image.fromarray(comp[c].astype(np.float32))
        img = img.resize((IMG_INPUT, IMG_INPUT), Image.BILINEAR)
        out[c] = np.asarray(img, dtype=np.float32)
    return torch.from_numpy(out)


PREP_FNS = {
    "rgb_dinosat":   prep_rgb_dinosat,
    "rgb_imagenet":  prep_rgb_dinosat,  # same percentile-stretch + RGB-order prep; model norm differs
    "prithvi_6band": prep_prithvi_6band,
}


def build_norm_tensors(device):
    """Return dict of (mean, std) tensors per input type, shaped (1,C,1,1) on device."""
    dt = torch.float16 if device.type == "cuda" else torch.float32
    def t(v, c):
        return torch.tensor(v, dtype=dt, device=device).view(1, c, 1, 1)
    return {
        "rgb_dinosat":   (t(DINO_SAT_MEAN, 3),  t(DINO_SAT_STD, 3)),
        "rgb_imagenet":  (t(IMAGENET_MEAN, 3),  t(IMAGENET_STD, 3)),
        "prithvi_6band": (t(PRITHVI_MEAN, 6),   t(PRITHVI_STD, 6)),
    }


# ---------------------------------------------------------------------------
# Spatial chunking (DBSCAN cluster → tight bbox per chunk)
# ---------------------------------------------------------------------------

def plan_chunks(utm_xy: np.ndarray, n_scenes: int, budget_bytes: int) -> list[np.ndarray]:
    n = len(utm_xy)
    if n == 0:
        return []
    if n == 1:
        return [np.arange(1)]
    labels = DBSCAN(eps=CLUSTER_EPS_M, min_samples=1).fit_predict(utm_xy)
    chunks: list[np.ndarray] = []
    # We're now fetching 6 bands (vs 3) so memory budget calc must account for that.
    bytes_per_pixel = n_scenes * 7 * 2  # 6 bands + SCL, 2 bytes each
    for lbl in np.unique(labels):
        idx = np.where(labels == lbl)[0]
        cxy = utm_xy[idx]
        w_px = (cxy[:, 0].max() - cxy[:, 0].min()) / GSD_M + IMG_NATIVE
        h_px = (cxy[:, 1].max() - cxy[:, 1].min()) / GSD_M + IMG_NATIVE
        bytes_est = w_px * h_px * bytes_per_pixel
        n_split = max(1, int(np.ceil(bytes_est / budget_bytes)))
        if n_split == 1:
            chunks.append(idx)
        else:
            sub_sorted = idx[np.argsort(utm_xy[idx, 1])]
            chunks.extend(np.array_split(sub_sorted, n_split))
    return chunks


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def setup_rasterio_env():
    return rasterio.Env(
        AWSSession(boto3.Session(), requester_pays=True),
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        GDAL_HTTP_MULTIPLEX="YES",
        GDAL_HTTP_VERSION="2",
        VSI_CACHE="TRUE",
        VSI_CACHE_SIZE=200_000_000,
        CPL_VSIL_CURL_CHUNK_SIZE=1_048_576,
        CPL_VSIL_CURL_CACHE_SIZE=200_000_000,
    )


def flush_all_models(batch_inputs, batch_meta, models, norms, device, emb_bufs, idx_rows):
    """Run every loaded model on this batch, append to per-model emb buffer, append
    a single shared row to idx_rows. batch_inputs is dict input_type -> list of tensors."""
    # Stack each input type
    stacked = {k: torch.stack(v).to(device) for k, v in batch_inputs.items() if v}
    if device.type == "cuda":
        stacked = {k: v.half() for k, v in stacked.items()}

    # Normalize and forward per model
    embeddings = {}  # model_name -> (B, D) float16 cpu
    with torch.inference_mode():
        for name, (model, spec) in models.items():
            inp_type = spec["input"]
            x = stacked[inp_type]
            mean, std = norms[inp_type]
            xn = (x - mean) / std
            emb = spec["forward_fn"](model, xn)
            embeddings[name] = emb.float().cpu().numpy().astype(np.float16)

    base_slot = len(idx_rows)
    for k, r in enumerate(batch_meta):
        slot = base_slot + k
        for name in models:
            emb_bufs[name][slot] = embeddings[name][k]
        idx_rows.append(dict(
            tile_id=r.tile_id, row_idx=slot, class_id=int(r.class_id),
            source=r.source, weight=float(r.weight), split=r.split,
            lat=float(r.lat), lon=float(r.lon), target_year=int(r.target_year),
        ))


def _prep_one_tile(scene_data, utm_x, utm_y, needed_inputs):
    """Composite + per-input prep in one call. Runs on a prep_pool worker thread.
    numpy/PIL ops release the GIL → multiple of these run truly in parallel."""
    comp = build_composite(scene_data, utm_x, utm_y)
    if comp is None:
        return None
    return {k: PREP_FNS[k](comp) for k in needed_inputs}


# ---------- Per-group checkpoint helpers (resume support) ------------------

def _group_paths(model_names, mgrs_tile, year):
    """Disk paths for one group's per-encoder embedding chunk + index.
    `__empty__` marks groups skipped for no-scenes/no-readers/all-skipped so
    we don't retry them on resume."""
    key = f"{mgrs_tile}_{int(year)}"
    return {
        "__idx__":   EMBED_CHUNKS_DIR / "_index" / f"{key}.parquet",
        "__empty__": EMBED_CHUNKS_DIR / "_empty" / f"{key}.json",
        **{n: EMBED_CHUNKS_DIR / n / f"{key}.npy" for n in model_names},
    }


def _group_is_done(paths):
    if paths["__empty__"].exists():
        return True
    return paths["__idx__"].exists() and all(
        p.exists() for k, p in paths.items() if k not in ("__idx__", "__empty__")
    )


def _atomic_save(write_fn, path):
    """Write via .tmp + rename so a crash mid-write never leaves a half-file
    that resume logic would mistake for valid."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    write_fn(tmp)
    tmp.replace(path)


def _atomic_save_npy(arr, path):
    """np.save auto-appends .npy if the path doesn't already end in .npy, so
    plain `_atomic_save` would write to <path>.tmp.npy and then fail to
    rename the (nonexistent) <path>.tmp."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npy")  # ends in .npy → np.save won't double-append
    np.save(tmp, arr)
    tmp.replace(path)


def _save_group_chunk(paths, model_names, embs, idx_rows):
    _atomic_save(lambda p: pd.DataFrame(idx_rows).to_parquet(p, index=False),
                 paths["__idx__"])
    for n in model_names:
        _atomic_save_npy(embs[n], paths[n])


def _mark_group_empty(paths, reasons):
    _atomic_save(lambda p: p.write_text(json.dumps(reasons)), paths["__empty__"])


def _concat_chunks(model_names):
    """Stitch per-group chunk files into final concatenated arrays + index df.
    Re-sequences row_idx as a global running index in chunk-file order."""
    idx_dir = EMBED_CHUNKS_DIR / "_index"
    if not idx_dir.exists():
        empty = {n: np.zeros((0, 0), dtype=np.float16) for n in model_names}
        return pd.DataFrame(), empty
    idx_files = sorted(idx_dir.glob("*.parquet"))
    if not idx_files:
        empty = {n: np.zeros((0, 0), dtype=np.float16) for n in model_names}
        return pd.DataFrame(), empty
    idx_dfs = []
    embs_per_model = {n: [] for n in model_names}
    offset = 0
    for f in idx_files:
        key = f.stem
        df = pd.read_parquet(f).copy()
        df["row_idx"] = np.arange(offset, offset + len(df))
        idx_dfs.append(df)
        for n in model_names:
            embs_per_model[n].append(np.load(EMBED_CHUNKS_DIR / n / f"{key}.npy"))
        offset += len(df)
    idx_df = pd.concat(idx_dfs, ignore_index=True)
    embs = {n: np.concatenate(embs_per_model[n], axis=0) for n in model_names}
    return idx_df, embs


# ---------- Group prefetch (parallel I/O across PREFETCH_GROUPS) -----------

# ---------- Per-stage telemetry ---------------------------------------------
# Main-thread time accounting: every blocking call main makes is wrapped in
# `time_stage` so we can attribute wallclock to {prefetch_wait, io, prep_wait,
# gpu, save}. Sum of stages ≈ wallclock; "other" is whatever's left
# (mostly chunk planning + dict construction + GC). One JSONL line per group
# to STATS_LOG; a window summary printed every 10 groups.

class StageTimer:
    """Cumulative seconds and call counts per stage; snapshot/reset for window stats."""
    def __init__(self):
        self.t: dict[str, float] = {}
        self.n: dict[str, int] = {}

    def add(self, stage: str, dt: float) -> None:
        self.t[stage] = self.t.get(stage, 0.0) + dt
        self.n[stage] = self.n.get(stage, 0) + 1

    def snapshot(self) -> tuple[dict[str, float], dict[str, int]]:
        return dict(self.t), dict(self.n)

    def reset(self) -> None:
        self.t.clear()
        self.n.clear()


from contextlib import contextmanager  # noqa: E402

@contextmanager
def time_stage(timer: StageTimer, stage: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        timer.add(stage, time.perf_counter() - t0)


def _append_stats(line: dict) -> None:
    STATS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with STATS_LOG.open("a") as f:
        f.write(json.dumps(line) + "\n")


def _prefetch_group(mgrs_tile, year, gdf, scene_lookup, io_pool):
    """Light prefetch: only opens scene readers and computes the chunk plan.
    Bulk-reads (load_group_arrays) are NOT done here — main thread does them
    one sub-chunk at a time so peak memory = one chunk, not all chunks.

    Returns (status, n_or_none, payload).
      status='no_scenes' / 'no_readers' → payload=None, n_or_none=len(gdf)
      status='ok' → payload={scenes, sub_chunks, tile_rows, utm_xy}; CALLER must close_scenes.
    """
    scene_rows = scene_lookup.get((mgrs_tile, int(year)), [])
    if not scene_rows:
        return ("no_scenes", len(gdf), None)
    scenes = open_scene_readers(scene_rows, io_pool)
    if not scenes:
        return ("no_readers", len(gdf), None)
    try:
        to_utm = pyproj.Transformer.from_crs(4326, _utm_epsg(mgrs_tile), always_xy=True).transform
        ux, uy = to_utm(gdf["lon"].to_numpy(), gdf["lat"].to_numpy())
        utm_xy = np.column_stack([ux, uy])
        tile_rows = list(gdf.itertuples(index=False))
        sub_chunks = plan_chunks(utm_xy, len(scenes), MEMORY_BUDGET_BYTES)
        return ("ok", None, {
            "scenes": scenes, "sub_chunks": sub_chunks,
            "tile_rows": tile_rows, "utm_xy": utm_xy,
        })
    except Exception:
        close_scenes(scenes)
        raise


def fetch_and_embed(manifest: pd.DataFrame, models: dict, device):
    """Group-prefetched, per-group-checkpointed embed pass.

    Per group:
      * group_pool prefetches PREFETCH_GROUPS groups in parallel (open scenes +
        bulk-read all sub-chunk arrays), so io_pool is fed by multiple groups
        concurrently and we expose more parallel S3 connections.
      * main thread drains prefetched groups in submission order, runs prep_pool
        + GPU forward, writes per-group chunk files (atomic rename) so a crash
        only loses work-in-progress, not completed groups.
      * On startup, groups whose chunk files already exist are skipped.

    Returns (n_total_tiles, idx_df, embs) — embs is dict[name -> (N, D) fp16].
    """
    norms = build_norm_tensors(device)
    needed_inputs = sorted({spec["input"] for _, spec in models.values()})
    model_names = list(models.keys())

    print(f"[v2-train] reading scenes index {SCENES_INDEX}")
    scenes_df = pd.read_parquet(SCENES_INDEX)
    print(f"[v2-train] {len(scenes_df):,} scene rows for "
          f"{scenes_df.groupby(['mgrs_tile','year']).ngroups} (mgrs, year) groups")

    scene_lookup: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in scenes_df.itertuples(index=False):
        scene_lookup[(r.mgrs_tile, int(r.year))].append({
            "scene_id": r.scene_id,
            "B02_s3": r.B02_s3, "B03_s3": r.B03_s3, "B04_s3": r.B04_s3,
            "B8A_s3": r.B8A_s3, "B11_s3": r.B11_s3, "B12_s3": r.B12_s3,
            "scl_s3": r.scl_s3,
        })

    groups = list(manifest.groupby(["mgrs_tile", "target_year"]))

    # Resume: skip groups whose chunk files already exist for the requested
    # model set. _group_is_done requires all current model_names AND the index
    # parquet to be present.
    todo: list[tuple[str, int, pd.DataFrame, dict]] = []
    n_resumed_groups = 0
    n_resumed_tiles = 0
    for (mgrs_tile, year), gdf in groups:
        paths = _group_paths(model_names, mgrs_tile, int(year))
        if _group_is_done(paths):
            n_resumed_groups += 1
            if not paths["__empty__"].exists():
                try:
                    n_resumed_tiles += len(pd.read_parquet(paths["__idx__"]))
                except Exception:
                    pass
            continue
        todo.append((mgrs_tile, int(year), gdf, paths))
    print(f"[v2-train] resume: {n_resumed_groups}/{len(groups)} groups already done "
          f"(+{n_resumed_tiles} tiles); {len(todo)} groups remaining")
    print(f"[v2-train] memory budget: {MEMORY_BUDGET_BYTES // 1024**3} GB; "
          f"prep_pool={PREP_WORKERS} io_pool={IO_WORKERS} prefetch={PREFETCH_GROUPS}")

    io_pool = ThreadPoolExecutor(max_workers=IO_WORKERS)
    prep_pool = ThreadPoolExecutor(max_workers=PREP_WORKERS)
    group_pool = ThreadPoolExecutor(max_workers=max(PREFETCH_GROUPS, 1))
    sk = {"no_scenes": 0, "no_readers": 0, "no_scene_data": 0, "no_composite": 0}
    n_filled_this_run = 0
    t0 = time.time()
    # Telemetry: per-stage main-thread time + per-group JSONL + window summary
    stage_t = StageTimer()
    last_snap_t, last_snap_n = stage_t.snapshot()
    window_start_wall = time.perf_counter()
    window_start_filled = 0
    window_start_gi = 0
    window_t0_t, window_t0_n = stage_t.snapshot()
    print(f"[v2-train] stats jsonl -> {STATS_LOG}")

    try:
        with setup_rasterio_env():
            inflight: list = []
            next_to_submit = 0

            def _submit(idx):
                mt, yr, gdf, paths = todo[idx]
                fut = group_pool.submit(_prefetch_group, mt, yr, gdf, scene_lookup, io_pool)
                inflight.append((mt, yr, gdf, paths, fut))

            while next_to_submit < min(PREFETCH_GROUPS, len(todo)):
                _submit(next_to_submit)
                next_to_submit += 1

            for gi in range(len(todo)):
                group_t0 = time.perf_counter()
                mt, yr, gdf, paths, fut = inflight.pop(0)
                # Keep the prefetch pipeline full
                if next_to_submit < len(todo):
                    _submit(next_to_submit)
                    next_to_submit += 1

                try:
                    with time_stage(stage_t, "prefetch_wait"):
                        status, n_or_none, payload = fut.result()
                except Exception as e:
                    print(f"[v2-train]   group {mt}/{yr} prefetch FAILED: {e!r}")
                    _mark_group_empty(paths, {"prefetch_error": repr(e)})
                    continue

                if status == "no_scenes":
                    sk["no_scenes"] += n_or_none
                    _mark_group_empty(paths, {"no_scenes": n_or_none})
                    continue
                if status == "no_readers":
                    sk["no_readers"] += n_or_none
                    _mark_group_empty(paths, {"no_readers": n_or_none})
                    continue

                scenes = payload["scenes"]
                sub_chunks = payload["sub_chunks"]
                tile_rows = payload["tile_rows"]
                utm_xy = payload["utm_xy"]
                del payload  # don't keep dict alive while we process

                group_embs = {n: [] for n in model_names}
                group_idx_rows: list = []

                try:
                    for sub_indices in sub_chunks:
                        sub_xy = utm_xy[sub_indices]
                        bbox = (
                            float(sub_xy[:, 0].min() - HALF_M),
                            float(sub_xy[:, 1].min() - HALF_M),
                            float(sub_xy[:, 0].max() + HALF_M),
                            float(sub_xy[:, 1].max() + HALF_M),
                        )
                        with time_stage(stage_t, "io"):
                            scene_data = load_group_arrays(scenes, io_pool, bbox)
                        if not any(s is not None for s in scene_data):
                            sk["no_scene_data"] += len(sub_indices)
                            del scene_data
                            continue

                        batch_inputs = {k: [] for k in needed_inputs}
                        batch_meta: list = []
                        idx_list = list(sub_indices)

                        def _do_flush():
                            nonlocal n_filled_this_run
                            if not batch_meta:
                                return
                            with time_stage(stage_t, "gpu"):
                                stacked = {k: torch.stack(v).to(device)
                                           for k, v in batch_inputs.items() if v}
                                if device.type == "cuda":
                                    stacked = {k: v.half() for k, v in stacked.items()}
                                with torch.inference_mode():
                                    for name, (model, spec) in models.items():
                                        inp_type = spec["input"]
                                        x = stacked[inp_type]
                                        mean, std = norms[inp_type]
                                        xn = (x - mean) / std
                                        emb = spec["forward_fn"](model, xn)
                                        emb_np = emb.float().cpu().numpy().astype(np.float16)
                                        for k in range(emb_np.shape[0]):
                                            group_embs[name].append(emb_np[k])
                            for r in batch_meta:
                                group_idx_rows.append(dict(
                                    tile_id=r.tile_id, class_id=int(r.class_id),
                                    source=r.source, weight=float(r.weight), split=r.split,
                                    lat=float(r.lat), lon=float(r.lon),
                                    target_year=int(r.target_year),
                                ))
                            n_filled_this_run += len(batch_meta)
                            for k in needed_inputs:
                                batch_inputs[k] = []
                            batch_meta.clear()

                        for cs in range(0, len(idx_list), PREP_CHUNK):
                            ce = min(cs + PREP_CHUNK, len(idx_list))
                            futs = [
                                (j, prep_pool.submit(_prep_one_tile, scene_data,
                                                     float(utm_xy[j, 0]),
                                                     float(utm_xy[j, 1]), needed_inputs))
                                for j in idx_list[cs:ce]
                            ]
                            for j, fut2 in futs:
                                with time_stage(stage_t, "prep_wait"):
                                    prepped = fut2.result()
                                if prepped is None:
                                    sk["no_composite"] += 1
                                    continue
                                for k in needed_inputs:
                                    batch_inputs[k].append(prepped[k])
                                batch_meta.append(tile_rows[j])
                                if len(batch_meta) >= BATCH_SIZE:
                                    _do_flush()
                        _do_flush()
                        del scene_data

                    if group_idx_rows:
                        with time_stage(stage_t, "save"):
                            _save_group_chunk(
                                paths, model_names,
                                {n: np.stack(group_embs[n]).astype(np.float16) for n in model_names},
                                group_idx_rows,
                            )
                    else:
                        _mark_group_empty(paths, {"all_skipped": True})
                finally:
                    close_scenes(scenes)
                    # Force release of any retained per-group objects (large
                    # numpy/torch buffers via cycles) — a 30 GB OOM in the
                    # first iteration of this design proved this matters.
                    if (gi + 1) % 25 == 0:
                        gc.collect()

                # Per-group stats
                group_wall = time.perf_counter() - group_t0
                cur_t, cur_n = stage_t.snapshot()
                group_t_delta = {k: round(cur_t[k] - last_snap_t.get(k, 0.0), 4)
                                 for k in cur_t if cur_t[k] - last_snap_t.get(k, 0.0) > 0}
                last_snap_t, last_snap_n = cur_t, cur_n
                rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
                _append_stats({
                    "ts": round(time.time(), 1),
                    "gi": gi + 1,
                    "group": f"{mt}_{yr}",
                    "n_tiles_group": len(group_idx_rows),
                    "n_filled_cum": n_filled_this_run,
                    "n_skipped_cum": sum(sk.values()),
                    "wallclock_group_s": round(group_wall, 3),
                    "stages_s_group": group_t_delta,
                    "rss_gb": round(rss_gb, 2),
                })

                if (gi + 1) % 10 == 0 or gi == len(todo) - 1:
                    # Window stats (since last summary). ETA is computed in
                    # groups/sec NOT tiles/sec — per-tile rate swings 1–300×
                    # with manifest density inside each group, while per-group
                    # wallclock is much tighter (median 4–8s) so it's the
                    # stable ETA unit.
                    window_wall = time.perf_counter() - window_start_wall
                    window_tiles = n_filled_this_run - window_start_filled
                    window_groups = (gi + 1) - window_start_gi
                    window_grp_rate = window_groups / max(window_wall, 1e-6)
                    win_t = {k: cur_t.get(k, 0.0) - window_t0_t.get(k, 0.0) for k in cur_t}
                    pct = {k: f"{(win_t[k] / max(window_wall, 1e-6) * 100):.0f}%" for k in win_t if win_t[k] > 0}
                    accounted = sum(win_t.values())
                    pct["other"] = f"{((window_wall - accounted) / max(window_wall, 1e-6) * 100):.0f}%"
                    # Cumulative (groups since this run started, not since process start)
                    skipped = sum(sk.values())
                    elapsed = time.time() - t0
                    cum_grp_rate = (gi + 1) / max(elapsed, 1e-6)
                    rem_groups = len(todo) - (gi + 1)
                    eta = rem_groups / max(cum_grp_rate, 1e-6) / 60
                    print(f"[v2-train]   group {gi+1}/{len(todo)}  "
                          f"filled={n_filled_this_run} (+{n_resumed_tiles} resumed) "
                          f"skipped={skipped} (no_scn={sk['no_scenes']} no_rdr={sk['no_readers']} "
                          f"no_dat={sk['no_scene_data']} no_cmp={sk['no_composite']}) "
                          f"win={window_grp_rate:.2f}g/s cum={cum_grp_rate:.2f}g/s ETA={eta:.0f}min rss={rss_gb:.1f}GB")
                    print(f"[v2-train]     stage% (last {window_groups} groups, {window_tiles} tiles, {window_wall:.1f}s): "
                          f"{' '.join(f'{k}={v}' for k, v in pct.items())}")
                    window_start_wall = time.perf_counter()
                    window_start_filled = n_filled_this_run
                    window_start_gi = gi + 1
                    window_t0_t, window_t0_n = cur_t, cur_n
    finally:
        prep_pool.shutdown(wait=False)
        io_pool.shutdown(wait=False)
        group_pool.shutdown(wait=False)

    skipped = sum(sk.values())
    print(f"[v2-train] embed pass done: {n_filled_this_run} new, {n_resumed_tiles} resumed, "
          f"{skipped} skipped (no_scn={sk['no_scenes']} no_rdr={sk['no_readers']} "
          f"no_dat={sk['no_scene_data']} no_cmp={sk['no_composite']})")

    print("[v2-train] concatenating chunk files...")
    idx_df, embs = _concat_chunks(model_names)
    print(f"[v2-train] concatenated: {len(idx_df)} index rows across {len(model_names)} encoders")
    return n_filled_this_run + n_resumed_tiles, idx_df, embs


# ---------------------------------------------------------------------------
# Probe training (per model)
# ---------------------------------------------------------------------------

def train_probe(name: str, emb_dim: int, emb: np.ndarray, idx: pd.DataFrame, device) -> dict:
    # Binary: keep class_id ∈ {0, 2} (drops any UC stragglers from old manifests).
    idx = idx[idx["class_id"].isin([0, 2])].reset_index(drop=True)
    train = idx[idx["split"] == "train"].reset_index(drop=True)
    test = idx[idx["split"] == "test"].reset_index(drop=True)

    def _to_label(cid: np.ndarray) -> np.ndarray:
        return (cid == 2).astype(np.int64)

    X_tr = torch.from_numpy(emb[train["row_idx"].values]).float()
    y_tr = torch.from_numpy(_to_label(train["class_id"].values))
    w_tr = torch.from_numpy(train["weight"].values).float()
    X_te = torch.from_numpy(emb[test["row_idx"].values]).float()
    y_te = torch.from_numpy(_to_label(test["class_id"].values))

    n_per_class = pd.Series(y_tr.numpy()).value_counts().to_dict()
    print(f"[v2-train] [{name}] train per-class: {n_per_class}")

    head = nn.Linear(emb_dim, N_CLASSES).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)

    BATCH = 4096
    n = len(X_tr)
    history = []
    for ep in range(50):
        perm = torch.randperm(n)
        head.train()
        ep_loss = 0.0
        for i in range(0, n, BATCH):
            sel = perm[i:i+BATCH]
            xb, yb, wb = X_tr[sel].to(device), y_tr[sel].to(device), w_tr[sel].to(device)
            logits = head(xb)
            loss_per = F.cross_entropy(logits, yb, reduction="none")
            loss = (loss_per * wb).sum() / wb.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += float(loss) * len(sel)
        sched.step()
        ep_loss /= n
        head.eval()
        with torch.inference_mode():
            te_logits = head(X_te.to(device))
            te_pred = te_logits.argmax(dim=1).cpu()
            acc = float((te_pred == y_te).float().mean())
            per_class_acc = {}
            for c in range(N_CLASSES):
                m = y_te == c
                if m.any():
                    per_class_acc[int(c)] = float((te_pred[m] == y_te[m]).float().mean())
        history.append(dict(epoch=ep, train_loss=ep_loss, test_acc=acc, per_class_acc=per_class_acc))
        if (ep + 1) % 10 == 0:
            print(f"[v2-train] [{name}] epoch {ep+1}: loss={ep_loss:.4f} test_acc={acc:.4f} per_class={per_class_acc}")

    probe_path = PROBES_DIR / f"probe_{name}.pt"
    torch.save({"state_dict": head.state_dict(), "n_classes": N_CLASSES, "emb_dim": emb_dim}, probe_path)
    print(f"[v2-train] [{name}] saved probe -> {probe_path}")
    return dict(history=history, final=history[-1])


def eval_probe(name: str, emb_dim: int, emb: np.ndarray, idx: pd.DataFrame, device) -> dict:
    probe_path = PROBES_DIR / f"probe_{name}.pt"
    if not probe_path.exists():
        return {}
    ckpt = torch.load(probe_path, map_location=device, weights_only=False)
    head = nn.Linear(emb_dim, N_CLASSES).to(device)
    head.load_state_dict(ckpt["state_dict"])
    head.eval()
    with torch.inference_mode():
        logits = head(torch.from_numpy(emb).float().to(device))
        probs = F.softmax(logits, dim=1).cpu().numpy()
    idx = idx.copy()
    idx["p_non"] = probs[idx["row_idx"].values, 0]
    idx["p_industrial"] = probs[idx["row_idx"].values, 1]

    # Binary scope: keep only class_id ∈ {0, 2}
    idx = idx[idx["class_id"].isin([0, 2])]
    test = idx[idx["split"] == "test"]
    out = {"model": name, "test_n": int(len(test))}
    y_bin = (test["class_id"] == 2).astype(int).values
    p_ind = test["p_industrial"].values
    out["non_n"] = int((y_bin == 0).sum())
    out["complete_n"] = int((y_bin == 1).sum())
    if len(test):
        try:
            from sklearn.metrics import roc_auc_score, average_precision_score
            out["auroc_industrial"] = float(roc_auc_score(y_bin, p_ind)) if y_bin.any() and (1 - y_bin).any() else None
            out["ap_industrial"] = float(average_precision_score(y_bin, p_ind)) if y_bin.any() else None
        except Exception:
            out["auroc_industrial"] = None
        out["recall_p_industrial>=0.7"] = float((p_ind[y_bin == 1] >= 0.7).mean()) if (y_bin == 1).any() else None
        out["recall_p_industrial>=0.95"] = float((p_ind[y_bin == 1] >= 0.95).mean()) if (y_bin == 1).any() else None

    # Hand-label-only test slice (highest trust)
    hand_test = test[test["source"].astype(str).str.startswith("hand_")]
    out["hand_test_n"] = int(len(hand_test))
    if len(hand_test):
        y_h = (hand_test["class_id"] == 2).astype(int).values
        p_h = hand_test["p_industrial"].values
        try:
            from sklearn.metrics import roc_auc_score
            out["hand_auroc_industrial"] = float(roc_auc_score(y_h, p_h)) if y_h.any() and (1 - y_h).any() else None
        except Exception:
            out["hand_auroc_industrial"] = None
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-embed", action="store_true",
                    help="train probes on existing embeddings")
    ap.add_argument("--models", default=",".join(s["name"] for s in MODEL_REGISTRY),
                    help="comma-separated model names from registry")
    args = ap.parse_args()
    requested = set(args.models.split(","))

    manifest = pd.read_parquet(MANIFEST)
    print(f"[v2-train] manifest rows: {len(manifest):,}")
    print(f"[v2-train] requested models: {sorted(requested)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[v2-train] device: {device}")

    emb_paths = {s["name"]: V2_DIR / f"emb_{s['name']}.npy"
                 for s in MODEL_REGISTRY if s["name"] in requested}

    # Skip-if-exists: if all expected outputs are on disk, just load and go to probes.
    if EMB_IDX_OUT.exists() and all(p.exists() for p in emb_paths.values()):
        idx_df = pd.read_parquet(EMB_IDX_OUT)
        embs = {n: np.load(p) for n, p in emb_paths.items()}
        print(f"[v2-train] reusing existing embeddings (N={len(idx_df)}); skipping embed pass")
    elif args.skip_embed:
        print("[v2-train] --skip-embed set but artifacts missing; aborting")
        sys.exit(1)
    else:
        print(f"[v2-train] computing MGRS for {len(manifest):,} manifest rows...")
        manifest = manifest.copy()
        manifest["mgrs_tile"] = compute_mgrs(manifest["lat"].values, manifest["lon"].values)
        manifest = manifest.dropna(subset=["mgrs_tile"]).reset_index(drop=True)

        models = load_models(device, requested)
        if not models:
            print("[v2-train] no models loaded; aborting")
            sys.exit(1)

        _, idx_df, embs = fetch_and_embed(manifest, models, device)

        if len(idx_df) == 0:
            print("[v2-train] no embeddings produced; aborting before probe step")
            sys.exit(1)

        idx_df.to_parquet(EMB_IDX_OUT, index=False)
        print(f"[v2-train] wrote index ({len(idx_df)} rows) -> {EMB_IDX_OUT}")
        for n, arr in embs.items():
            if n not in emb_paths:
                continue
            np.save(emb_paths[n], arr)
            print(f"[v2-train] wrote emb {n} {arr.shape} -> {emb_paths[n]}")

    leaderboard = {}
    for spec in MODEL_REGISTRY:
        name = spec["name"]
        if name not in embs:
            continue
        try:
            train_rep = train_probe(name, spec["emb_dim"], embs[name], idx_df, device)
            eval_rep = eval_probe(name, spec["emb_dim"], embs[name], idx_df, device)
        except Exception as e:
            print(f"[v2-train] [{name}] probe failed: {e!r}")
            continue
        (PROBES_DIR / f"probe_{name}_train.json").write_text(json.dumps(train_rep, indent=2))
        (PROBES_DIR / f"probe_{name}_eval.json").write_text(json.dumps(eval_rep, indent=2))
        leaderboard[name] = eval_rep

    LEADERBOARD.write_text(json.dumps(leaderboard, indent=2))
    print(f"[v2-train] leaderboard -> {LEADERBOARD}")
    print(json.dumps(leaderboard, indent=2))


if __name__ == "__main__":
    main()
