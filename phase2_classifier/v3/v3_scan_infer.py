"""CONUS scan inference (rewritten from scratch).

Simple architecture: torch DataLoader with multiprocess workers fetches NAIP
crops; main process does GPU forward + probe scoring; per-chunk parquet writes
enable resume.

Why multiprocess: each worker is its own Python process — no GIL, no rasterio
thread-local Env race, no semaphore juggling. Prefetch queue keeps GPU fed.

Why no AWSSession: GDAL picks AWS_ACCESS_KEY_ID/SECRET/REQUEST_PAYER from env
vars directly. AWSSession's lazy credential resolution wedged worker threads
in Env.start() on EC2 (deterministic on chunk 126 across 3 runs).

Reads:
  data_us/phase2/v3_scan_manifest.parquet
  data_us/phase2/v3_scan_scenes_index.parquet
  data_us/phase2/v3/probes/probe_<model>.pt

Writes:
  data_us/phase2/v3/scan_chunks/_scores/chunk_*.parquet
  data_us/phase2/v3/scan_results.parquet (final concat)

Env knobs:
  V3_SCAN_SAMPLE_N      — random subsample (validation runs); omit for full
  V3_SCAN_SEED          — sample seed (default 7)
  V3_SCAN_MODELS        — comma-separated model names (default dino_sat493m)
  V3_SCAN_ONLY_CHUNKS   — comma-separated chunk indices for probe mode
  V3_SCAN_MAX_CHUNKS    — cap todo length (validation)
  V3_NUM_WORKERS        — DataLoader workers (default 16)
  V3_PREFETCH_FACTOR    — DataLoader prefetch (default 4)
  V3_BATCH_SIZE         — GPU batch (default 64)
"""
from __future__ import annotations

import faulthandler
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path

# Periodic thread-stack dump for hang diagnosis (writes every 30s).
_STACKS_PATH = "/tmp/v3_stacks.txt"
faulthandler.enable()
def _stack_dumper() -> None:
    while True:
        try:
            with open(_STACKS_PATH, "w") as f:
                f.write(f"=== dump {time.time():.0f} ===\n")
                for tid, frame in sys._current_frames().items():
                    f.write(f"\n--- thread {tid} ---\n")
                    f.write("".join(traceback.format_stack(frame)))
        except Exception:
            pass
        time.sleep(30)
_stack_thread = threading.Thread(target=_stack_dumper, name="stack-dumper", daemon=True)
_stack_thread.start()

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

# Load .env into os.environ BEFORE any rasterio/boto import so GDAL picks up
# AWS credentials from env vars (no AWSSession needed).
ENV_PATH = ROOT / "sites_us" / ".env"
if ENV_PATH.exists():
    for line in ENV_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
os.environ.setdefault("AWS_REQUEST_PAYER", "requester")

import numpy as np
import pandas as pd
import rasterio
from rasterio.merge import merge as rio_merge
from rasterio.windows import Window, from_bounds
from rasterio.warp import transform_bounds

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from PIL import Image

from sites_us.phase2_classifier.v3.v3_train import (
    load_models, build_norm_tensors, MODEL_REGISTRY,
    N_CLASSES, IMG_INPUT,
)

DATA_US = ROOT / "data_us"
MANIFEST = DATA_US / "phase2" / "v3_scan_manifest.parquet"
SCENES   = DATA_US / "phase2" / "v3_scan_scenes_index.parquet"
PROBES_DIR = DATA_US / "phase2" / "v3" / "probes"
V3_DIR = DATA_US / "phase2" / "v3"
SCAN_CHUNK_DIR = V3_DIR / "scan_chunks" / "_scores"
SCAN_RESULTS = V3_DIR / "scan_results.parquet"

CHUNK_SIZE = 2048
BATCH_SIZE = int(os.environ.get("V3_BATCH_SIZE", 64))
NUM_WORKERS = int(os.environ.get("V3_NUM_WORKERS", 16))
PREFETCH_FACTOR = int(os.environ.get("V3_PREFETCH_FACTOR", 4))

SAMPLE_N = int(os.environ.get("V3_SCAN_SAMPLE_N", 0)) or None
SAMPLE_SEED = int(os.environ.get("V3_SCAN_SEED", 7))
SCAN_MODELS = [s.strip() for s in os.environ.get("V3_SCAN_MODELS", "dino_sat493m").split(",") if s.strip()]
MAX_CHUNKS = int(os.environ.get("V3_SCAN_MAX_CHUNKS", 0)) or None
ONLY_CHUNKS_RAW = os.environ.get("V3_SCAN_ONLY_CHUNKS", "").strip()
ONLY_CHUNKS = sorted({int(x) for x in ONLY_CHUNKS_RAW.split(",") if x.strip()}) if ONLY_CHUNKS_RAW else None

MODEL_REGISTRY_BY_NAME = {m["name"]: m for m in MODEL_REGISTRY}

# GDAL knobs that make COG reads cheap. NO AWSSession — env vars only.
_GDAL_KNOBS = dict(
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    GDAL_HTTP_MULTIPLEX="YES",
    GDAL_HTTP_VERSION="2",
    GDAL_HTTP_TIMEOUT="20",
    GDAL_HTTP_MAX_RETRY="5",
    GDAL_HTTP_RETRY_DELAY="0.5",
    CPL_VSIL_CURL_USE_HEAD="NO",
    CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
    GDAL_INGESTED_BYTES_AT_OPEN="524288",
    VSI_CACHE="TRUE",
    VSI_CACHE_SIZE=1_073_741_824,  # 1 GB per worker process
    CPL_VSIL_CURL_CHUNK_SIZE=1_048_576,
    AWS_REQUEST_PAYER="requester",
)


# ---------------------------------------------------------------------------
# Fetch + prep (runs in worker processes)
# ---------------------------------------------------------------------------

def _fetch_window(uris: list, bbox: tuple) -> np.ndarray | None:
    """Read RGB window from one or more NAIP COGs. bbox is (xmin,ymin,xmax,ymax)
    in EPSG:4326. Multi-URI → rio_merge. Returns (3, H, W) uint8 or None."""
    if not uris:
        return None
    try:
        if len(uris) == 1:
            with rasterio.open(uris[0]) as src:
                if src.crs is None:
                    return None
                xmin, ymin, xmax, ymax = transform_bounds(
                    "EPSG:4326", src.crs, *bbox, densify_pts=21)
                win = from_bounds(xmin, ymin, xmax, ymax, transform=src.transform)
                col = max(0, int(round(win.col_off)))
                row = max(0, int(round(win.row_off)))
                col_end = min(src.width, int(round(win.col_off + win.width)))
                row_end = min(src.height, int(round(win.row_off + win.height)))
                if col >= col_end or row >= row_end:
                    return None
                return src.read([1, 2, 3], window=Window(col, row, col_end - col, row_end - row))
        # multi-tile mosaic
        srcs = [rasterio.open(u) for u in uris]
        try:
            xmin, ymin, xmax, ymax = transform_bounds(
                "EPSG:4326", srcs[0].crs, *bbox, densify_pts=21)
            arr, _ = rio_merge(srcs, bounds=(xmin, ymin, xmax, ymax), indexes=[1, 2, 3])
            return arr
        finally:
            for s in srcs:
                try: s.close()
                except Exception: pass
    except Exception:
        return None


def _prep_crop(arr: np.ndarray) -> torch.Tensor | None:
    """RGB uint8 (3,H,W) → 1-99% stretch → square-pad → LANCZOS 224 → float tensor."""
    if arr is None or arr.size == 0 or arr.ndim != 3 or arr.shape[0] < 3:
        return None
    rgb = arr[:3].astype(np.float32)
    out = np.empty_like(rgb)
    for c in range(3):
        lo, hi = np.percentile(rgb[c], [1, 99])
        if hi - lo < 1.0:
            return None
        out[c] = np.clip((rgb[c] - lo) / max(hi - lo, 1e-6), 0, 1)
    _, h, w = out.shape
    side = max(h, w)
    pad = np.zeros((3, side, side), dtype=np.float32)
    y0 = (side - h) // 2
    x0 = (side - w) // 2
    pad[:, y0:y0+h, x0:x0+w] = out
    arr_u8 = (pad.transpose(1, 2, 0) * 255).astype(np.uint8)
    img = Image.fromarray(arr_u8).resize((IMG_INPUT, IMG_INPUT), Image.LANCZOS)
    return torch.from_numpy(np.asarray(img).astype(np.float32).transpose(2, 0, 1) / 255.0)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

# Sentinel tensor for failed/skipped rows. Collate stacks fixed-shape tensors;
# returning None breaks default_collate, so we emit a zeros tensor + a `valid`
# flag the main loop filters on. Tiny constant memory cost vs the alternative
# of a custom collate_fn.
_DEAD_TENSOR = torch.zeros(3, IMG_INPUT, IMG_INPUT)


class NAIPCrops(IterableDataset):
    """Yields per-building dicts with prepped crop tensors. Shards across
    DataLoader workers via worker_info.id."""
    def __init__(self, rows: pd.DataFrame):
        # Store as records (lighter to pickle than DataFrame).
        cols = ["chunk_id", "building_id", "ovt_id", "lat", "lon",
                "approx_area_m2", "ovt_class", "naip_uris",
                "fetch_xmin", "fetch_ymin", "fetch_xmax", "fetch_ymax"]
        self.records = rows[cols].to_dict("records")

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        # Open a single rasterio.Env for this worker process. Env covers all
        # rasterio.open() calls in this process; no per-call Env nesting.
        env = rasterio.Env(**_GDAL_KNOBS)
        env.__enter__()
        try:
            if info is None:
                sub = self.records
            else:
                sub = self.records[info.id::info.num_workers]
            for r in sub:
                uris = list(r["naip_uris"]) if r["naip_uris"] is not None else []
                bbox = (float(r["fetch_xmin"]), float(r["fetch_ymin"]),
                        float(r["fetch_xmax"]), float(r["fetch_ymax"]))
                arr = _fetch_window(uris, bbox)
                t = _prep_crop(arr)
                if t is None:
                    yield {
                        "valid": 0,
                        "tensor": _DEAD_TENSOR,
                        "chunk_id": int(r["chunk_id"]),
                        "building_id": str(r["building_id"]),
                        "ovt_id": str(r["ovt_id"]) if r["ovt_id"] is not None else "",
                        "lat": float(r["lat"]),
                        "lon": float(r["lon"]),
                        "approx_area_m2": float(r["approx_area_m2"]),
                        "ovt_class": str(r["ovt_class"]) if r["ovt_class"] is not None else "",
                    }
                    continue
                yield {
                    "valid": 1,
                    "tensor": t,
                    "chunk_id": int(r["chunk_id"]),
                    "building_id": str(r["building_id"]),
                    "ovt_id": str(r["ovt_id"]) if r["ovt_id"] is not None else "",
                    "lat": float(r["lat"]),
                    "lon": float(r["lon"]),
                    "approx_area_m2": float(r["approx_area_m2"]),
                    "ovt_class": str(r["ovt_class"]) if r["ovt_class"] is not None else "",
                }
        finally:
            try: env.__exit__(None, None, None)
            except Exception: pass


# ---------------------------------------------------------------------------
# Output + probe loading
# ---------------------------------------------------------------------------

def _chunk_score_path(ci: int) -> Path:
    return SCAN_CHUNK_DIR / f"chunk_{ci:06d}.parquet"


def _save_chunk(ci: int, rows: list) -> None:
    """Atomic parquet write — tmp file + rename."""
    if not rows:
        return
    df = pd.DataFrame(rows)
    p = _chunk_score_path(ci)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp.parquet")
    df.to_parquet(tmp, index=False)
    tmp.replace(p)


def _load_probes(model_names, device) -> dict[str, nn.Module]:
    probes = {}
    for name in model_names:
        p = PROBES_DIR / f"probe_{name}.pt"
        if not p.exists():
            print(f"[scan] WARN: probe missing for {name} ({p})", flush=True)
            continue
        ckpt = torch.load(p, map_location=device, weights_only=False)
        head = nn.Linear(ckpt["emb_dim"], ckpt.get("n_classes", N_CLASSES)).to(device)
        head.load_state_dict(ckpt["state_dict"]); head.eval()
        probes[name] = head
        print(f"[scan] probe loaded: {name} (emb_dim={ckpt['emb_dim']})")
    return probes


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _concat() -> None:
    files = sorted(SCAN_CHUNK_DIR.glob("chunk_*.parquet"))
    if not files:
        print("[scan] no chunk files to concat"); return
    dfs = [pd.read_parquet(f) for f in files]
    out = pd.concat(dfs, ignore_index=True)
    out.to_parquet(SCAN_RESULTS, index=False)
    print(f"[scan] concat: {len(out):,} rows → {SCAN_RESULTS}")


def main() -> None:
    device = _device()
    print(f"[scan] device={device} batch={BATCH_SIZE} workers={NUM_WORKERS} prefetch={PREFETCH_FACTOR}", flush=True)

    manifest = pd.read_parquet(MANIFEST)
    scenes = pd.read_parquet(SCENES)
    print(f"[scan] manifest={len(manifest):,}  scenes={len(scenes):,}", flush=True)

    merged = manifest.merge(
        scenes[["building_id", "naip_uris", "fetch_xmin", "fetch_ymin",
                "fetch_xmax", "fetch_ymax", "n_tiles"]],
        on="building_id", how="left",
    )
    merged = merged[merged["n_tiles"].fillna(0) > 0].reset_index(drop=True)
    print(f"[scan] with NAIP coverage: {len(merged):,}", flush=True)

    if SAMPLE_N:
        merged = merged.sample(n=min(SAMPLE_N, len(merged)),
                               random_state=SAMPLE_SEED).reset_index(drop=True)
        print(f"[scan] sampled: {len(merged):,} (V3_SCAN_SAMPLE_N={SAMPLE_N})")

    # Stable order by primary URI gives the DataLoader good cache locality
    # (adjacent buildings likely share a NAIP tile, so worker VSI_CACHE reuses).
    merged["primary_uri"] = merged["naip_uris"].map(
        lambda u: u[0] if isinstance(u, (list, np.ndarray)) and len(u) > 0 else "")
    merged = merged.sort_values("primary_uri").reset_index(drop=True)
    merged = merged.drop(columns="primary_uri")

    n_total = len(merged)
    n_chunks = (n_total + CHUNK_SIZE - 1) // CHUNK_SIZE
    print(f"[scan] chunks: {n_chunks} × {CHUNK_SIZE}", flush=True)

    SCAN_CHUNK_DIR.mkdir(parents=True, exist_ok=True)

    # Resume: which chunks are already done?
    done = {int(p.stem.split("_")[-1]) for p in SCAN_CHUNK_DIR.glob("chunk_*.parquet")}
    todo = [ci for ci in range(n_chunks) if ci not in done]
    print(f"[scan] resume: {len(done)}/{n_chunks} chunks done; {len(todo)} remaining", flush=True)

    if ONLY_CHUNKS is not None:
        todo = [ci for ci in ONLY_CHUNKS if 0 <= ci < n_chunks]
        print(f"[scan] V3_SCAN_ONLY_CHUNKS={ONLY_CHUNKS_RAW} → probe mode, todo={todo}", flush=True)
    if MAX_CHUNKS:
        todo = todo[:MAX_CHUNKS]
        print(f"[scan] V3_SCAN_MAX_CHUNKS={MAX_CHUNKS} → {len(todo)} chunks", flush=True)

    if not todo:
        print("[scan] nothing to do — concat & exit", flush=True)
        _concat()
        return

    # Slice merged to just the todo rows. Attach chunk_id column so workers can
    # emit it back with each item, letting the main loop bucket scores per-chunk
    # without re-deriving from index.
    todo_set = set(todo)
    chunk_id_col = np.arange(n_total) // CHUNK_SIZE
    keep_mask = np.isin(chunk_id_col, list(todo_set))
    todo_rows = merged.iloc[keep_mask].copy()
    todo_rows["chunk_id"] = chunk_id_col[keep_mask]
    expected_per_chunk = todo_rows.groupby("chunk_id").size().to_dict()
    print(f"[scan] todo rows: {len(todo_rows):,} across {len(todo)} chunks", flush=True)

    # Load model + probe in main process (CUDA).
    print(f"[scan] loading models: {SCAN_MODELS}", flush=True)
    models = load_models(device, SCAN_MODELS)
    norms = build_norm_tensors(models, device)
    probes = _load_probes(list(models.keys()), device)
    active = [n for n in models if n in probes]
    print(f"[scan] active models with probes: {active}", flush=True)
    if not active:
        print("[scan] FATAL: no active models", flush=True); sys.exit(1)

    # DataLoader
    ds = NAIPCrops(todo_rows)
    dl = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
        persistent_workers=False,  # one-shot iteration; not worth process keep-alive
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    # Per-chunk score buffers; flush when count == expected.
    buffers: dict[int, list] = {ci: [] for ci in todo}
    n_processed = 0
    n_failed = 0
    n_chunks_done = 0
    t0 = time.time()
    t_last_log = t0

    print(f"[scan] starting forward pass...", flush=True)
    for batch in dl:
        # batch is a dict of: tensor (B,3,H,W), valid (B,), chunk_id (B,),
        # building_id (list), ovt_id (list), lat (B,), lon (B,),
        # approx_area_m2 (B,), ovt_class (list)
        valid_mask = batch["valid"].numpy().astype(bool)
        n_failed += int((~valid_mask).sum())
        n_processed += len(valid_mask)

        if valid_mask.any():
            x = batch["tensor"][valid_mask].to(device, non_blocking=True)
            if device.type == "cuda":
                x = x.half()
            chunk_ids = batch["chunk_id"][valid_mask].numpy()
            bids = [batch["building_id"][i] for i in range(len(valid_mask)) if valid_mask[i]]
            ovt_ids = [batch["ovt_id"][i] for i in range(len(valid_mask)) if valid_mask[i]]
            ovt_classes = [batch["ovt_class"][i] for i in range(len(valid_mask)) if valid_mask[i]]
            lats = batch["lat"][valid_mask].numpy()
            lons = batch["lon"][valid_mask].numpy()
            areas = batch["approx_area_m2"][valid_mask].numpy()

            with torch.inference_mode():
                scores_by_model = {}
                for name in active:
                    model, _spec = models[name]
                    mean, std = norms[name]
                    emb = MODEL_REGISTRY_BY_NAME[name]["forward_fn"](model, (x - mean) / std)
                    logits = probes[name](emb.float())
                    scores_by_model[name] = F.softmax(logits, dim=1)[:, 1].cpu().numpy()

            for i in range(len(bids)):
                ci = int(chunk_ids[i])
                row = {
                    "building_id": bids[i], "ovt_id": ovt_ids[i],
                    "lat": float(lats[i]), "lon": float(lons[i]),
                    "approx_area_m2": float(areas[i]), "ovt_class": ovt_classes[i],
                }
                for name in active:
                    row[f"p_{name}"] = float(scores_by_model[name][i])
                if len(active) >= 2:
                    row["p_mean"] = float(np.mean([scores_by_model[n][i] for n in active]))
                buffers[ci].append(row)

        # Flush completed chunks. A chunk is complete when its buffer count
        # (valid + invalid) equals expected. Track invalid via expected-vs-actual
        # but simpler: also count invalid rows toward the chunk's count.
        # Re-scan buffer keys: flush any chunk where total accounted ≥ expected.
        # We track accounted = valid_in_buffer + invalid_seen_for_chunk via a
        # parallel counter:
        # ↓ simpler: count INVALID into expected too
        # We'll add invalids as no-score rows to the buffer.
        for i in range(len(valid_mask)):
            if valid_mask[i]:
                continue
            ci = int(batch["chunk_id"][i])
            row = {
                "building_id": batch["building_id"][i],
                "ovt_id": batch["ovt_id"][i],
                "lat": float(batch["lat"][i]),
                "lon": float(batch["lon"][i]),
                "approx_area_m2": float(batch["approx_area_m2"][i]),
                "ovt_class": batch["ovt_class"][i],
            }
            for name in active:
                row[f"p_{name}"] = float("nan")
            if len(active) >= 2:
                row["p_mean"] = float("nan")
            buffers[ci].append(row)

        # Flush any chunks that hit expected count.
        for ci in list(buffers.keys()):
            if len(buffers[ci]) >= expected_per_chunk.get(ci, 0):
                _save_chunk(ci, buffers[ci])
                del buffers[ci]
                n_chunks_done += 1

        # Periodic log (every 5s)
        now = time.time()
        if now - t_last_log >= 5:
            elapsed = now - t0
            rate = n_processed / max(elapsed, 1e-6)
            remaining = len(todo_rows) - n_processed
            eta_min = remaining / max(rate, 1e-6) / 60
            print(f"[scan] {n_processed:,}/{len(todo_rows):,} buildings "
                  f"({rate:.0f}/s) | chunks_done={n_chunks_done}/{len(todo)} | "
                  f"failed={n_failed} | ETA={eta_min:.1f}min",
                  flush=True)
            t_last_log = now

    # Flush remaining (incomplete) chunks
    for ci, rows in buffers.items():
        if rows:
            _save_chunk(ci, rows)
            n_chunks_done += 1

    elapsed = time.time() - t0
    print(f"[scan] DONE: {n_processed:,} buildings, {n_failed} failed, "
          f"{n_chunks_done} chunks in {elapsed:.0f}s "
          f"({n_processed / max(elapsed, 1e-6):.0f}/s)", flush=True)
    _concat()


if __name__ == "__main__":
    main()
