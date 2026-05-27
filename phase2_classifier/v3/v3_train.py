"""End-to-end v3 detector training — per-building NAIP crops, 2-encoder bake-off.

Reads `data_us/phase2/v3_scenes_index.parquet` and produces, for each model M:
  - data_us/phase2/v3/emb_<M>.npy           (fp16, N x emb_dim_M)
  - data_us/phase2/v3/probes/probe_<M>.pt
  - data_us/phase2/v3/probes/probe_<M>_train.json
  - data_us/phase2/v3/probes/probe_<M>_eval.json
plus shared:
  - data_us/phase2/v3/v3_embeddings_index.parquet
  - data_us/phase2/v3/leaderboard.json

Differences from v2:
  * Input: NAIP COG window-read (1m GSD, single-date, cloud-curated) — no STAC,
    no SCL masking, no multi-scene composite.
  * Unit: per-building crop = (fetch_xmin, fetch_ymin, fetch_xmax, fetch_ymax)
    from v3_scenes_index — a tight window around the building bbox + buffer.
  * Straddling buildings (n_tiles > 1) are mosaiced via rasterio's merge.
  * Backbones: dino_sat493m + dino_vitb only (no Prithvi, no ResNet).
  * Resume granularity: per-chunk (CHUNK_SIZE buildings/chunk), atomic-renamed
    .npy + .parquet per chunk.

Usage on EC2:
  python v3_train.py                # full pipeline
  python v3_train.py --skip-embed   # train probes on existing embeddings
  python v3_train.py --models dino_sat493m  # subset
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import resource
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
import pyproj
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from rasterio.merge import merge as rio_merge
from rasterio.session import AWSSession
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds, Window
from transformers import AutoModel

ROOT = Path(__file__).resolve().parents[3]
DATA_US = ROOT / "data_us"
MANIFEST = DATA_US / "phase2" / "v3_dataset_manifest.parquet"
SCENES_INDEX = DATA_US / "phase2" / "v3_scenes_index.parquet"

V3_DIR = DATA_US / "phase2" / "v3"
V3_DIR.mkdir(parents=True, exist_ok=True)
PROBES_DIR = V3_DIR / "probes"
PROBES_DIR.mkdir(parents=True, exist_ok=True)
EMB_IDX_OUT = V3_DIR / "v3_embeddings_index.parquet"
LEADERBOARD = V3_DIR / "leaderboard.json"
CHUNK_DIR = V3_DIR / "embed_chunks"
STATS_LOG = V3_DIR / "stats.jsonl"

N_CLASSES = 2
IMG_INPUT = 224
CHUNK_SIZE = int(os.environ.get("V3_CHUNK_SIZE", 4096))  # buildings per resume chunk
BATCH_SIZE = 64                 # GPU batch
# Env-overridable so local Mac runs can throttle concurrency without code edits.
IO_WORKERS   = int(os.environ.get("V3_IO_WORKERS",   1024))
PREP_WORKERS = int(os.environ.get("V3_PREP_WORKERS", 16))
PREFETCH_CHUNKS = 2             # chunks fetched ahead of GPU consumer

PREFLIGHT_N = int(os.environ.get("V3_PREFLIGHT_N", 256))   # buildings used to measure fetch rate
PREFLIGHT_MIN_RATE = float(os.environ.get("V3_PREFLIGHT_MIN_RATE", 30.0))  # b/s floor; abort below

# ru_maxrss is bytes on macOS, KB on Linux — normalize to GB.
_RSS_DIVISOR = 1024 ** 3 if platform.system() == "Darwin" else 1024 ** 2

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
DINO_SAT_MEAN = [0.430, 0.411, 0.296]
DINO_SAT_STD  = [0.213, 0.156, 0.143]


# ---------------------------------------------------------------------------
# Telemetry — thread-safe per-call timing records, drained per chunk.
# ---------------------------------------------------------------------------

_fetch_lock = threading.Lock()
_fetch_records: dict[int, list[tuple]] = {}   # chunk_id -> [(wall, n_uris, bytes, err)]
_prep_lock = threading.Lock()
_prep_records: dict[int, list[tuple]] = {}    # chunk_id -> [(wall, ok)]

# Sentinel for the standalone preflight path which has no chunk_id.
_PREFLIGHT_CHUNK_ID = -1


def _rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / _RSS_DIVISOR


def _record_fetch(chunk_id: int, wall: float, n_uris: int, bytes_read: int, err: str | None) -> None:
    with _fetch_lock:
        _fetch_records.setdefault(chunk_id, []).append((wall, n_uris, bytes_read, err))


def _record_prep(chunk_id: int, wall: float, ok: bool) -> None:
    with _prep_lock:
        _prep_records.setdefault(chunk_id, []).append((wall, ok))


def _drain_telemetry(chunk_id: int, window_wall_s: float | None = None) -> dict:
    """Aggregate per-chunk records into a stats dict. Pops state for chunk_id."""
    with _fetch_lock:
        fr = _fetch_records.pop(chunk_id, [])
    with _prep_lock:
        pr = _prep_records.pop(chunk_id, [])
    out: dict = {"fetch_n": len(fr), "prep_n": len(pr)}
    if fr:
        fw = np.array([r[0] for r in fr], dtype=np.float64)
        fu = np.array([r[1] for r in fr], dtype=np.int32)
        fb = np.array([r[2] for r in fr], dtype=np.float64)
        ok = np.array([r[3] is None for r in fr])
        out["fetch_p50_ms"] = round(float(np.percentile(fw, 50)) * 1000, 1)
        out["fetch_p95_ms"] = round(float(np.percentile(fw, 95)) * 1000, 1)
        out["fetch_p99_ms"] = round(float(np.percentile(fw, 99)) * 1000, 1)
        out["fetch_max_ms"] = round(float(fw.max()) * 1000, 1)
        out["fetch_mean_ms"] = round(float(fw.mean()) * 1000, 1)
        out["fetch_mb_total"] = round(float(fb.sum()) / 1024 / 1024, 2)
        out["fetch_ok"] = int(ok.sum())
        out["fetch_err"] = int((~ok).sum())
        out["n_single_tile"] = int((fu == 1).sum())
        out["n_multi_tile"] = int((fu > 1).sum())
        out["n_zero_tile"] = int((fu == 0).sum())
        err_types: dict = {}
        for r in fr:
            if r[3]:
                err_types[r[3]] = err_types.get(r[3], 0) + 1
        if err_types:
            out["fetch_err_types"] = err_types
        if window_wall_s and window_wall_s > 0:
            out["fetch_mb_per_s"] = round(float(fb.sum()) / 1024 / 1024 / window_wall_s, 2)
    if pr:
        pw = np.array([r[0] for r in pr], dtype=np.float64)
        pok = np.array([r[1] for r in pr])
        out["prep_p50_ms"] = round(float(np.percentile(pw, 50)) * 1000, 1)
        out["prep_p95_ms"] = round(float(np.percentile(pw, 95)) * 1000, 1)
        out["prep_mean_ms"] = round(float(pw.mean()) * 1000, 1)
        out["prep_ok"] = int(pok.sum())
        out["prep_err"] = int((~pok).sum())
    return out


def _stats_write(record: dict) -> None:
    STATS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with STATS_LOG.open("a") as f:
        f.write(json.dumps(record) + "\n")


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


def forward_dinov3(model, x):
    return model(x).last_hidden_state[:, 0, :]


MODEL_REGISTRY = [
    {"name": "dino_sat493m", "hf_id": "facebook/dinov3-vitl16-pretrain-sat493m",
     "norm_mean": DINO_SAT_MEAN, "norm_std": DINO_SAT_STD, "emb_dim": 1024,
     "load_fn": load_dinov3, "forward_fn": forward_dinov3},
    {"name": "dino_vitb",     "hf_id": "facebook/dinov3-vitb16-pretrain-lvd1689m",
     "norm_mean": IMAGENET_MEAN, "norm_std": IMAGENET_STD, "emb_dim": 768,
     "load_fn": load_dinov3, "forward_fn": forward_dinov3},
]


def load_models(device, names):
    loaded = {}
    for spec in MODEL_REGISTRY:
        if spec["name"] not in names:
            continue
        print(f"[v3-train] loading {spec['name']} ({spec['hf_id']})")
        try:
            m = spec["load_fn"](spec, device)
            loaded[spec["name"]] = (m, spec)
        except Exception as e:
            print(f"[v3-train]   {spec['name']} load failed: {e!r}; skipping")
    print(f"[v3-train] models active: {list(loaded.keys())}")
    return loaded


def build_norm_tensors(models, device):
    dt = torch.float16 if device.type == "cuda" else torch.float32
    out = {}
    for name, (_, spec) in models.items():
        mean = torch.tensor(spec["norm_mean"], dtype=dt, device=device).view(1, 3, 1, 1)
        std  = torch.tensor(spec["norm_std"],  dtype=dt, device=device).view(1, 3, 1, 1)
        out[name] = (mean, std)
    return out


# ---------------------------------------------------------------------------
# Rasterio + NAIP fetch
# ---------------------------------------------------------------------------

def setup_rasterio_env():
    # GDAL/curl knobs tuned for many-small-windowed-reads against requester-pays
    # NAIP COGs on S3. HTTP/2 multiplexing + skipped HEAD + larger ingest-at-open
    # cuts per-file round trips from 3 to 1. VSI cache is shared across the
    # process, so buildings that share a NAIP tile reuse the IFD and adjacent
    # byte ranges instead of refetching. See ScientificDaemon/COG-perf notes.
    return rasterio.Env(
        AWSSession(boto3.Session(), requester_pays=True),
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        GDAL_HTTP_MULTIPLEX="YES",
        GDAL_HTTP_VERSION="2",
        GDAL_HTTP_MAX_RETRY="5",
        GDAL_HTTP_RETRY_DELAY="0.5",
        GDAL_HTTP_TIMEOUT="20",
        CPL_VSIL_CURL_USE_HEAD="NO",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
        GDAL_INGESTED_BYTES_AT_OPEN="524288",
        VSI_CACHE="TRUE",
        VSI_CACHE_SIZE=2_147_483_648,       # 2 GB, shared across files
        CPL_VSIL_CURL_CHUNK_SIZE=1_048_576,
        AWS_REQUEST_PAYER="requester",
    )


def _fetch_tile_group(chunk_id: int, uri: str,
                      items: list[tuple[int, float, float, float, float]]
                      ) -> list[tuple[int, np.ndarray | None]]:
    """Open a NAIP COG once, perform window-reads for every crop sharing it.
    Avoids the 4×-per-tile redundant VSI open cost when sort+chunking gives
    avg ~4 buildings per primary tile.

    items: (row_idx, fxmin, fymin, fxmax, fymax) in EPSG:4326.
    """
    out: list[tuple[int, np.ndarray | None]] = []
    try:
        with rasterio.open(uri) as src:
            crs = src.crs
            for row_idx, fxmin, fymin, fxmax, fymax in items:
                t0 = time.perf_counter()
                arr: np.ndarray | None = None
                err: str | None = None
                try:
                    if crs is None:
                        raise RuntimeError("no_crs")
                    xmin, ymin, xmax, ymax = transform_bounds(
                        "EPSG:4326", crs, fxmin, fymin, fxmax, fymax,
                        densify_pts=21,
                    )
                    win = from_bounds(xmin, ymin, xmax, ymax, transform=src.transform)
                    col = max(0, int(round(win.col_off)))
                    row = max(0, int(round(win.row_off)))
                    col_end = min(src.width, int(round(win.col_off + win.width)))
                    row_end = min(src.height, int(round(win.row_off + win.height)))
                    if col >= col_end or row >= row_end:
                        raise RuntimeError("empty_window")
                    actual = Window(col, row, col_end - col, row_end - row)
                    arr = src.read([1, 2, 3], window=actual)
                except Exception as e:
                    err = type(e).__name__ or "Exception"
                bytes_read = int(arr.nbytes) if arr is not None else 0
                _record_fetch(chunk_id, time.perf_counter() - t0, 1, bytes_read, err)
                out.append((row_idx, arr))
    except Exception as e:
        err = type(e).__name__ or "Exception"
        for row_idx, *_ in items:
            _record_fetch(chunk_id, 0.0, 1, 0, err)
            out.append((row_idx, None))
    return out


def _fetch_crop(chunk_id: int, uris: list[str], fetch_xmin: float, fetch_ymin: float,
                fetch_xmax: float, fetch_ymax: float) -> np.ndarray | None:
    """Read RGB crop = (fetch_bbox) from the listed NAIP COG(s), returning a
    (3, H, W) uint8 numpy array. Mosaics multiple tiles when n_tiles > 1.

    NAIP COGs are 4-band (RGB + NIR); we keep only the first 3 (RGB).
    Self-instruments via _record_fetch so per-chunk telemetry can aggregate
    P50/95/99 latency, bytes throughput, and error breakdown by exception type.
    """
    t0 = time.perf_counter()
    arr: np.ndarray | None = None
    err: str | None = None
    n_uris = len(uris) if uris else 0
    try:
        if n_uris == 0:
            err = "no_uris"
        elif n_uris == 1:
            with rasterio.open(uris[0]) as src:
                if src.crs is None:
                    raise RuntimeError("no_crs")
                xmin, ymin, xmax, ymax = transform_bounds(
                    "EPSG:4326", src.crs,
                    fetch_xmin, fetch_ymin, fetch_xmax, fetch_ymax,
                    densify_pts=21,
                )
                win = from_bounds(xmin, ymin, xmax, ymax, transform=src.transform)
                col = max(0, int(round(win.col_off)))
                row = max(0, int(round(win.row_off)))
                col_end = min(src.width, int(round(win.col_off + win.width)))
                row_end = min(src.height, int(round(win.row_off + win.height)))
                if col >= col_end or row >= row_end:
                    raise RuntimeError("empty_window")
                actual = Window(col, row, col_end - col, row_end - row)
                arr = src.read([1, 2, 3], window=actual)
        else:
            srcs = [rasterio.open(u) for u in uris]
            try:
                crs = srcs[0].crs
                xmin, ymin, xmax, ymax = transform_bounds(
                    "EPSG:4326", crs,
                    fetch_xmin, fetch_ymin, fetch_xmax, fetch_ymax,
                    densify_pts=21,
                )
                mosaic, _ = rio_merge(srcs, bounds=(xmin, ymin, xmax, ymax),
                                      indexes=[1, 2, 3])
                arr = mosaic
            finally:
                for s in srcs:
                    try: s.close()
                    except Exception: pass
    except Exception as e:
        err = type(e).__name__ or "Exception"
    bytes_read = int(arr.nbytes) if arr is not None else 0
    _record_fetch(chunk_id, time.perf_counter() - t0, n_uris, bytes_read, err)
    return arr


def prep_crop(chunk_id: int, arr: np.ndarray) -> torch.Tensor | None:
    """RGB crop → 1–99 percentile stretch → square-pad → LANCZOS 224 → tensor.

    arr is (3, H, W) uint8 from rasterio. NAIP is already RGB so no channel
    swap needed. Self-instruments via _record_prep."""
    t0 = time.perf_counter()
    result: torch.Tensor | None = None
    try:
        if arr is None or arr.size == 0:
            return None
        if arr.ndim != 3 or arr.shape[0] < 3:
            return None
        rgb = arr[:3].astype(np.float32)
        # 1–99 percentile per channel for robust contrast (NAIP can vary by season).
        out = np.empty_like(rgb)
        for c in range(3):
            lo, hi = np.percentile(rgb[c], [1, 99])
            if hi - lo < 1.0:  # degenerate (all-water tile, all-snow, etc.)
                return None
            out[c] = np.clip((rgb[c] - lo) / max(hi - lo, 1e-6), 0, 1)
        # Square-pad to longest side so the resize doesn't distort aspect.
        _, h, w = out.shape
        side = max(h, w)
        pad = np.zeros((3, side, side), dtype=np.float32)
        y0 = (side - h) // 2
        x0 = (side - w) // 2
        pad[:, y0:y0+h, x0:x0+w] = out
        # LANCZOS resize 256-ish→224
        arr_u8 = (pad.transpose(1, 2, 0) * 255).astype(np.uint8)
        img = Image.fromarray(arr_u8).resize((IMG_INPUT, IMG_INPUT), Image.LANCZOS)
        result = torch.from_numpy(np.asarray(img).astype(np.float32).transpose(2, 0, 1) / 255.0)
        return result
    finally:
        _record_prep(chunk_id, time.perf_counter() - t0, result is not None)


# ---------------------------------------------------------------------------
# Per-chunk checkpoint helpers
# ---------------------------------------------------------------------------

def _chunk_paths(model_names, ci: int):
    return {
        "__idx__": CHUNK_DIR / "_index" / f"chunk_{ci:06d}.parquet",
        "__empty__": CHUNK_DIR / "_empty" / f"chunk_{ci:06d}.json",
        **{n: CHUNK_DIR / n / f"chunk_{ci:06d}.npy" for n in model_names},
    }


def _chunk_done(paths) -> bool:
    if paths["__empty__"].exists():
        return True
    return paths["__idx__"].exists() and all(
        p.exists() for k, p in paths.items() if k not in ("__idx__", "__empty__")
    )


def _atomic_save(write_fn, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    write_fn(tmp)
    tmp.replace(path)


def _atomic_save_npy(arr: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, arr)
    tmp.replace(path)


def _save_chunk(paths, model_names, embs, idx_rows):
    _atomic_save(lambda p: pd.DataFrame(idx_rows).to_parquet(p, index=False),
                 paths["__idx__"])
    for n in model_names:
        _atomic_save_npy(embs[n], paths[n])


def _concat_chunks(model_names):
    idx_dir = CHUNK_DIR / "_index"
    if not idx_dir.exists():
        return pd.DataFrame(), {n: np.zeros((0, 0), dtype=np.float16) for n in model_names}
    idx_files = sorted(idx_dir.glob("chunk_*.parquet"))
    if not idx_files:
        return pd.DataFrame(), {n: np.zeros((0, 0), dtype=np.float16) for n in model_names}
    idx_dfs = []
    embs_per_model = {n: [] for n in model_names}
    offset = 0
    for f in idx_files:
        key = f.stem
        df = pd.read_parquet(f).copy()
        df["row_idx"] = np.arange(offset, offset + len(df))
        idx_dfs.append(df)
        for n in model_names:
            embs_per_model[n].append(np.load(CHUNK_DIR / n / f"{key}.npy"))
        offset += len(df)
    idx_df = pd.concat(idx_dfs, ignore_index=True)
    embs = {n: np.concatenate(embs_per_model[n], axis=0) for n in model_names}
    return idx_df, embs


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

def run_preflight(scenes_full: pd.DataFrame, n: int,
                  io_pool: ThreadPoolExecutor, do_prep: bool = True) -> dict:
    """Fetch (and optionally prep) the first n rows of scenes_full to measure
    I/O performance before the full run. Returns aggregated telemetry dict.

    Drains _fetch_records / _prep_records and writes one stats.jsonl entry."""
    n = min(n, len(scenes_full))
    if n == 0:
        return {"phase": "preflight", "skipped": True}
    rows = scenes_full.iloc[:n]
    print(f"[v3-train] preflight: fetching {n} buildings...", flush=True)
    # Drain any stale records from a previous phase.
    _drain_telemetry(_PREFLIGHT_CHUNK_ID)
    t0 = time.perf_counter()
    fut = [io_pool.submit(_fetch_crop, _PREFLIGHT_CHUNK_ID, list(r.naip_uris),
                          float(r.fetch_xmin), float(r.fetch_ymin),
                          float(r.fetch_xmax), float(r.fetch_ymax))
           for r in rows.itertuples(index=False)]
    arrs = [f.result() for f in fut]
    fetch_wall = time.perf_counter() - t0
    if do_prep:
        t_p0 = time.perf_counter()
        prep_pool = ThreadPoolExecutor(max_workers=PREP_WORKERS)
        try:
            pf = [prep_pool.submit(prep_crop, _PREFLIGHT_CHUNK_ID, a) for a in arrs]
            _ = [f.result() for f in pf]
        finally:
            prep_pool.shutdown(wait=False)
        prep_wall = time.perf_counter() - t_p0
    else:
        prep_wall = 0.0
    wall = time.perf_counter() - t0
    tel = _drain_telemetry(_PREFLIGHT_CHUNK_ID, window_wall_s=wall)
    rate = n / max(wall, 1e-6)
    record = dict(
        phase="preflight",
        ts=round(time.time(), 1),
        n=n,
        io_workers=IO_WORKERS,
        prep_workers=PREP_WORKERS,
        fetch_wall_s=round(fetch_wall, 2),
        prep_wall_s=round(prep_wall, 2),
        total_wall_s=round(wall, 2),
        rate_b_per_s=round(rate, 1),
        threads_alive=threading.active_count(),
        rss_gb=round(_rss_gb(), 2),
        **tel,
    )
    _stats_write(record)
    err_str = ""
    if tel.get("fetch_err_types"):
        err_str = f"  err={tel['fetch_err_types']}"
    print(f"[v3-train] preflight: {tel.get('fetch_ok', 0)}/{n} ok  "
          f"fetch={fetch_wall:.1f}s prep={prep_wall:.1f}s wall={wall:.1f}s  "
          f"rate={rate:.1f}b/s  "
          f"p50={tel.get('fetch_p50_ms')}ms p95={tel.get('fetch_p95_ms')}ms "
          f"p99={tel.get('fetch_p99_ms')}ms max={tel.get('fetch_max_ms')}ms  "
          f"net={tel.get('fetch_mb_per_s', 0)}MB/s  "
          f"single={tel.get('n_single_tile', 0)} multi={tel.get('n_multi_tile', 0)}"
          f"{err_str}",
          flush=True)
    return record


def _log_config(device, requested_models) -> None:
    cfg = {
        "phase": "config",
        "ts": round(time.time(), 1),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "rasterio": rasterio.__version__,
        "io_workers": IO_WORKERS,
        "prep_workers": PREP_WORKERS,
        "chunk_size": CHUNK_SIZE,
        "batch_size": BATCH_SIZE,
        "preflight_n": PREFLIGHT_N,
        "preflight_min_rate": PREFLIGHT_MIN_RATE,
        "device": str(device),
        "models_requested": sorted(requested_models),
        # Only the IO-relevant env knobs — these are what we'd change to chase perf.
        "env": {k: os.environ.get(k, "") for k in [
            "GDAL_HTTP_VERSION", "GDAL_HTTP_MULTIPLEX", "GDAL_HTTP_MAX_RETRY",
            "GDAL_HTTP_TIMEOUT", "CPL_VSIL_CURL_USE_HEAD",
            "GDAL_INGESTED_BYTES_AT_OPEN", "VSI_CACHE", "VSI_CACHE_SIZE",
            "AWS_REQUEST_PAYER", "AWS_DEFAULT_REGION", "AWS_REGION",
        ]},
    }
    _stats_write(cfg)
    print(f"[v3-train] config: io_workers={IO_WORKERS} prep_workers={PREP_WORKERS} "
          f"chunk={CHUNK_SIZE} batch={BATCH_SIZE} device={device}", flush=True)


# ---------------------------------------------------------------------------
# Embedding pass
# ---------------------------------------------------------------------------

def fetch_and_embed(scenes: pd.DataFrame, manifest: pd.DataFrame,
                    models: dict, device) -> tuple[pd.DataFrame, dict]:
    norms = build_norm_tensors(models, device)
    model_names = list(models.keys())
    # Join scenes ⨝ manifest on building_id for labels + metadata
    meta_cols = ["building_id", "class_id", "source", "weight", "split", "site_id",
                 "ovt_class", "ovt_subtype", "ovt_name", "ovt_id"]
    scenes_full = scenes.merge(manifest[meta_cols], on="building_id", how="left")
    # Skip buildings with no NAIP coverage upfront.
    scenes_full = scenes_full[scenes_full["n_tiles"] > 0].reset_index(drop=True)
    n_total = len(scenes_full)
    print(f"[v3-train] buildings to embed: {n_total:,}  ({len(scenes) - n_total} skipped for no NAIP)")

    # Sort by primary NAIP tile URI so adjacent buildings share open datasets
    # and reuse the VSI cache. 80% of buildings are single-tile; sorting cuts
    # IFD fetches by roughly the average building-per-tile multiplicity.
    scenes_full["__sort_uri__"] = scenes_full["naip_uris"].map(
        lambda u: u[0] if isinstance(u, (list, np.ndarray)) and len(u) > 0 else "")
    scenes_full = (scenes_full
                   .sort_values("__sort_uri__", kind="stable")
                   .drop(columns="__sort_uri__")
                   .reset_index(drop=True))
    n_unique_primary_tiles = scenes_full["naip_uris"].map(
        lambda u: u[0] if isinstance(u, (list, np.ndarray)) and len(u) > 0 else "").nunique()
    print(f"[v3-train] sorted by primary NAIP tile URI "
          f"(avg buildings/tile = {n_total / max(n_unique_primary_tiles, 1):.1f})")

    # Chunk planning. Each chunk = CHUNK_SIZE consecutive buildings.
    n_chunks = (n_total + CHUNK_SIZE - 1) // CHUNK_SIZE
    print(f"[v3-train] chunks: {n_chunks} × {CHUNK_SIZE}")

    # Resume scan
    todo: list[int] = []
    n_resumed_chunks = 0
    n_resumed_rows = 0
    for ci in range(n_chunks):
        p = _chunk_paths(model_names, ci)
        if _chunk_done(p):
            n_resumed_chunks += 1
            if not p["__empty__"].exists():
                try:
                    n_resumed_rows += len(pd.read_parquet(p["__idx__"]))
                except Exception:
                    pass
        else:
            todo.append(ci)
    print(f"[v3-train] resume: {n_resumed_chunks}/{n_chunks} chunks done (+{n_resumed_rows} rows); "
          f"{len(todo)} chunks remaining")

    io_pool = ThreadPoolExecutor(max_workers=IO_WORKERS)
    n_filled_this_run = 0
    n_failed = 0
    t0 = time.time()

    try:
        with setup_rasterio_env():
            for chunk_idx_pos, ci in enumerate(todo):
                t_chunk_start = time.perf_counter()
                cs = ci * CHUNK_SIZE
                ce = min(cs + CHUNK_SIZE, n_total)
                chunk_rows = scenes_full.iloc[cs:ce].reset_index(drop=True)
                paths = _chunk_paths(model_names, ci)

                # Partition rows by tile-open profile: single-tile crops share
                # one rasterio.open() per primary tile URI (avg ~4 crops/tile,
                # so this cuts open overhead ~4×). Multi-tile (mosaic) crops
                # remain per-crop via _fetch_crop.
                single_groups: dict[str, list[tuple]] = {}
                multi_items: list[tuple[int, list[str], float, float, float, float]] = []
                for i, r in enumerate(chunk_rows.itertuples(index=False)):
                    uris = list(r.naip_uris) if r.naip_uris is not None else []
                    coords = (i, float(r.fetch_xmin), float(r.fetch_ymin),
                              float(r.fetch_xmax), float(r.fetch_ymax))
                    if len(uris) == 1:
                        single_groups.setdefault(uris[0], []).append(coords)
                    elif len(uris) > 1:
                        multi_items.append((i, uris, coords[1], coords[2],
                                            coords[3], coords[4]))
                n_tile_opens = len(single_groups) + len(multi_items)

                crops_arr: list[np.ndarray | None] = [None] * len(chunk_rows)
                group_futs = [io_pool.submit(_fetch_tile_group, ci, uri, items)
                              for uri, items in single_groups.items()]
                multi_futs = [io_pool.submit(_fetch_crop, ci, uris, x0, y0, x1, y1)
                              for (_, uris, x0, y0, x1, y1) in multi_items]
                for fu in group_futs:
                    for ri, a in fu.result():
                        crops_arr[ri] = a
                for (ri, *_), fu in zip(multi_items, multi_futs):
                    crops_arr[ri] = fu.result()

                prep_pool = ThreadPoolExecutor(max_workers=PREP_WORKERS)
                prep_futs = [prep_pool.submit(prep_crop, ci, a) for a in crops_arr]
                crops = [(r, pf.result())
                         for r, pf in zip(chunk_rows.itertuples(index=False), prep_futs)]
                prep_pool.shutdown(wait=False)

                t_gpu = 0.0
                n_gpu_batches = 0
                t_gpu_per_model = {n: 0.0 for n in model_names}

                # GPU forward
                batch_tensors = []
                batch_meta: list = []
                chunk_embs = {n: [] for n in model_names}
                chunk_idx_rows: list = []

                def _flush():
                    nonlocal n_filled_this_run, t_gpu, n_gpu_batches
                    if not batch_meta:
                        return
                    t_g0 = time.perf_counter()
                    x = torch.stack(batch_tensors).to(device)
                    if device.type == "cuda":
                        x = x.half()
                    with torch.inference_mode():
                        for name, (model, spec) in models.items():
                            t_m0 = time.perf_counter()
                            mean, std = norms[name]
                            xn = (x - mean) / std
                            emb = spec["forward_fn"](model, xn)
                            if device.type == "cuda":
                                torch.cuda.synchronize()
                            emb_np = emb.float().cpu().numpy().astype(np.float16)
                            t_gpu_per_model[name] += time.perf_counter() - t_m0
                            for k in range(emb_np.shape[0]):
                                chunk_embs[name].append(emb_np[k])
                    for r in batch_meta:
                        chunk_idx_rows.append(dict(
                            building_id=r.building_id, class_id=int(r.class_id),
                            source=r.source, weight=float(r.weight), split=r.split,
                            site_id=r.site_id, lat=float(r.lat), lon=float(r.lon),
                            ovt_id=r.ovt_id,
                        ))
                    n_filled_this_run += len(batch_meta)
                    t_gpu += time.perf_counter() - t_g0
                    n_gpu_batches += 1
                    batch_tensors.clear(); batch_meta.clear()

                n_failed_chunk = 0
                for r, t in crops:
                    if t is None:
                        n_failed += 1
                        n_failed_chunk += 1
                        continue
                    batch_tensors.append(t)
                    batch_meta.append(r)
                    if len(batch_meta) >= BATCH_SIZE:
                        _flush()
                _flush()

                if chunk_idx_rows:
                    _save_chunk(paths, model_names,
                                {n: np.stack(chunk_embs[n]).astype(np.float16) for n in model_names},
                                chunk_idx_rows)
                else:
                    _atomic_save(lambda p: p.write_text(json.dumps({"all_skipped": True})),
                                 paths["__empty__"])

                # ---- Per-chunk telemetry ---------------------------------
                wall = time.perf_counter() - t_chunk_start
                tel = _drain_telemetry(ci, window_wall_s=wall)
                # How many distinct primary tiles back this chunk? High reuse
                # ratio = sort + VSI cache effective.
                primary_uris = chunk_rows["naip_uris"].map(
                    lambda u: u[0] if isinstance(u, (list, np.ndarray)) and len(u) > 0 else "")
                unique_uris = int(primary_uris.nunique())
                _stats_write(dict(
                    phase="chunk",
                    ts=round(time.time(), 1),
                    chunk=ci,
                    chunk_pos=chunk_idx_pos,
                    n_rows=len(chunk_idx_rows),
                    chunk_size=len(chunk_rows),
                    wall_s=round(wall, 3),
                    gpu_s=round(t_gpu, 3),
                    gpu_batches=n_gpu_batches,
                    gpu_s_per_model={n: round(v, 3) for n, v in t_gpu_per_model.items()},
                    tile_opens=n_tile_opens,
                    rss_gb=round(_rss_gb(), 2),
                    threads_alive=threading.active_count(),
                    unique_primary_tiles=unique_uris,
                    tile_reuse_ratio=round(len(chunk_rows) / max(unique_uris, 1), 2),
                    n_failed_chunk=n_failed_chunk,
                    **tel,
                ))

                if (chunk_idx_pos + 1) % 10 == 0 or chunk_idx_pos == len(todo) - 1:
                    elapsed = time.time() - t0
                    rate = (chunk_idx_pos + 1) / max(elapsed, 1e-6)
                    eta = (len(todo) - chunk_idx_pos - 1) / max(rate, 1e-6) / 60
                    print(f"[v3-train] chunk {chunk_idx_pos+1}/{len(todo)} "
                          f"wall={wall:.1f}s gpu={t_gpu:.2f}s({n_gpu_batches}b) "
                          f"opens={n_tile_opens}  "
                          f"fetch p50={tel.get('fetch_p50_ms')}ms "
                          f"p99={tel.get('fetch_p99_ms')}ms  "
                          f"net={tel.get('fetch_mb_per_s', 0)}MB/s  "
                          f"fail={tel.get('fetch_err', 0)}/{tel.get('fetch_n', 0)}  "
                          f"reuse={round(len(chunk_rows) / max(unique_uris, 1), 1)}x  "
                          f"rss={_rss_gb():.1f}GB  "
                          f"ETA={eta:.0f}min",
                          flush=True)
                if (chunk_idx_pos + 1) % 50 == 0:
                    gc.collect()
    finally:
        io_pool.shutdown(wait=False)

    print(f"[v3-train] embed pass done: {n_filled_this_run} new, {n_resumed_rows} resumed, "
          f"{n_failed} failed")
    print("[v3-train] concatenating chunks...")
    idx_df, embs = _concat_chunks(model_names)
    print(f"[v3-train] concatenated: {len(idx_df)} index rows across {len(model_names)} encoders")
    return idx_df, embs


# ---------------------------------------------------------------------------
# Probe training (lifted from v2_train.py)
# ---------------------------------------------------------------------------

def train_probe(name: str, emb_dim: int, emb: np.ndarray, idx: pd.DataFrame, device) -> dict:
    idx = idx[idx["class_id"].isin([0, 2])].reset_index(drop=True)
    train = idx[idx["split"] == "train"].reset_index(drop=True)
    test  = idx[idx["split"] == "test"].reset_index(drop=True)

    def _to_label(cid: np.ndarray) -> np.ndarray:
        return (cid == 2).astype(np.int64)

    X_tr = torch.from_numpy(emb[train["row_idx"].values]).float()
    y_tr = torch.from_numpy(_to_label(train["class_id"].values))
    w_tr = torch.from_numpy(train["weight"].values).float()
    X_te = torch.from_numpy(emb[test["row_idx"].values]).float()
    y_te = torch.from_numpy(_to_label(test["class_id"].values))

    print(f"[v3-train] [{name}] train per-class: "
          f"{pd.Series(y_tr.numpy()).value_counts().to_dict()}")

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
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += float(loss) * len(sel)
        sched.step()
        ep_loss /= n
        head.eval()
        with torch.inference_mode():
            te_pred = head(X_te.to(device)).argmax(dim=1).cpu()
            acc = float((te_pred == y_te).float().mean())
        history.append(dict(epoch=ep, train_loss=ep_loss, test_acc=acc))
        if (ep + 1) % 10 == 0:
            print(f"[v3-train] [{name}] epoch {ep+1}: loss={ep_loss:.4f} test_acc={acc:.4f}")

    torch.save({"state_dict": head.state_dict(),
                "n_classes": N_CLASSES, "emb_dim": emb_dim},
               PROBES_DIR / f"probe_{name}.pt")
    return dict(history=history, final=history[-1])


def eval_probe(name: str, emb_dim: int, emb: np.ndarray, idx: pd.DataFrame, device) -> dict:
    p = PROBES_DIR / f"probe_{name}.pt"
    if not p.exists():
        return {}
    ckpt = torch.load(p, map_location=device, weights_only=False)
    head = nn.Linear(emb_dim, N_CLASSES).to(device)
    head.load_state_dict(ckpt["state_dict"]); head.eval()
    with torch.inference_mode():
        probs = F.softmax(head(torch.from_numpy(emb).float().to(device)), dim=1).cpu().numpy()
    idx = idx.copy()
    idx["p_industrial"] = probs[idx["row_idx"].values, 1]
    idx = idx[idx["class_id"].isin([0, 2])]
    test = idx[idx["split"] == "test"]
    out = {"model": name, "test_n": int(len(test))}
    y_bin = (test["class_id"] == 2).astype(int).values
    p_ind = test["p_industrial"].values
    out["non_n"] = int((y_bin == 0).sum())
    out["industrial_n"] = int((y_bin == 1).sum())
    if len(test):
        try:
            from sklearn.metrics import roc_auc_score, average_precision_score
            out["auroc"] = float(roc_auc_score(y_bin, p_ind)) if y_bin.any() and (1 - y_bin).any() else None
            out["ap"] = float(average_precision_score(y_bin, p_ind)) if y_bin.any() else None
        except Exception:
            out["auroc"] = None
        out["recall_p>=0.7"] = float((p_ind[y_bin == 1] >= 0.7).mean()) if (y_bin == 1).any() else None
        out["recall_p>=0.95"] = float((p_ind[y_bin == 1] >= 0.95).mean()) if (y_bin == 1).any() else None
        out["fpr_p>=0.5"] = float((p_ind[y_bin == 0] >= 0.5).mean()) if (y_bin == 0).any() else None
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-embed", action="store_true")
    ap.add_argument("--models", default=",".join(s["name"] for s in MODEL_REGISTRY))
    ap.add_argument("--preflight-only", type=int, default=None, metavar="N",
                    help="Fetch+prep N buildings, log telemetry, and exit. "
                         "Skips model loading and the embed pass — fast I/O probe.")
    args = ap.parse_args()
    requested = set(args.models.split(","))

    manifest = pd.read_parquet(MANIFEST)
    scenes   = pd.read_parquet(SCENES_INDEX)
    print(f"[v3-train] manifest: {len(manifest):,}; scenes: {len(scenes):,}")
    print(f"[v3-train] requested models: {sorted(requested)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log_config(device, requested)

    # ----- Standalone preflight: I/O-only probe, no models, no GPU --------
    if args.preflight_only is not None:
        meta_cols = ["building_id"]
        sf = scenes.merge(manifest[meta_cols], on="building_id", how="left")
        sf = sf[sf["n_tiles"] > 0].reset_index(drop=True)
        # Same sort as the full embed pass so cache reuse is comparable.
        sf["__sort_uri__"] = sf["naip_uris"].map(
            lambda u: u[0] if isinstance(u, (list, np.ndarray)) and len(u) > 0 else "")
        sf = sf.sort_values("__sort_uri__", kind="stable").drop(columns="__sort_uri__").reset_index(drop=True)
        io_pool = ThreadPoolExecutor(max_workers=IO_WORKERS)
        try:
            with setup_rasterio_env():
                run_preflight(sf, args.preflight_only, io_pool, do_prep=True)
        finally:
            io_pool.shutdown(wait=False)
        print(f"[v3-train] preflight-only complete; stats at {STATS_LOG}")
        return

    emb_paths = {s["name"]: V3_DIR / f"emb_{s['name']}.npy"
                 for s in MODEL_REGISTRY if s["name"] in requested}

    if EMB_IDX_OUT.exists() and all(p.exists() for p in emb_paths.values()):
        idx_df = pd.read_parquet(EMB_IDX_OUT)
        embs = {n: np.load(p) for n, p in emb_paths.items()}
        print(f"[v3-train] reusing existing embeddings (N={len(idx_df)})")
    elif args.skip_embed:
        print("[v3-train] --skip-embed set but artifacts missing; aborting")
        sys.exit(1)
    else:
        models = load_models(device, requested)
        if not models:
            sys.exit(1)
        idx_df, embs = fetch_and_embed(scenes, manifest, models, device)
        if len(idx_df) == 0:
            sys.exit(1)
        idx_df.to_parquet(EMB_IDX_OUT, index=False)
        print(f"[v3-train] wrote index -> {EMB_IDX_OUT}")
        for n, arr in embs.items():
            if n in emb_paths:
                np.save(emb_paths[n], arr)
                print(f"[v3-train] wrote emb {n} {arr.shape} -> {emb_paths[n]}")

    leaderboard = {}
    for spec in MODEL_REGISTRY:
        name = spec["name"]
        if name not in embs:
            continue
        try:
            train_rep = train_probe(name, spec["emb_dim"], embs[name], idx_df, device)
            eval_rep  = eval_probe(name, spec["emb_dim"], embs[name], idx_df, device)
        except Exception as e:
            print(f"[v3-train] [{name}] probe failed: {e!r}")
            continue
        (PROBES_DIR / f"probe_{name}_train.json").write_text(json.dumps(train_rep, indent=2))
        (PROBES_DIR / f"probe_{name}_eval.json").write_text(json.dumps(eval_rep, indent=2))
        leaderboard[name] = eval_rep

    LEADERBOARD.write_text(json.dumps(leaderboard, indent=2))
    print(f"[v3-train] leaderboard -> {LEADERBOARD}")
    print(json.dumps(leaderboard, indent=2))


if __name__ == "__main__":
    main()
