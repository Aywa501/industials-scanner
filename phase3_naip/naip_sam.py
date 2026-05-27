"""Step 4 worker: NAIP fetch + SAM 3 text-prompted segmentation per cluster.

Reads the per-cluster NAIP manifest, fetches NAIP COGs windowed to each
cluster's fetch bbox (in-region us-west-2 reads, requester-pays NAIP bucket),
and runs SAM 3 (Meta's Promptable Concept Segmentation model, `facebook/sam3`
via HuggingFace transformers) against a fixed list of text prompts (building,
warehouse, parking lot, tank, silo, vegetation, road, ...). Each returned mask
is tagged with the prompt that produced it.

Output schema (parquet, S3-cached): one row per mask with `label` (the prompt),
`label_score`, polygon + bounds + per-band radiometry + shape stats. The
labelled output replaces the SAM 1 AMG path; downstream Step 5 reasons over
labelled masks instead of having to re-derive material categories.

Reads:
  data_us/phase3_naip/naip_manifest.parquet     (one row per cluster_id)
Writes (S3):
  s3://{BUCKET}/{S3_PREFIX}/{cluster_id}/masks.parquet

Auth: requester-pays for NAIP reads (rasterio AWSSession requester_pays=True);
instance-role or .env IAM for the output bucket.

Note: SAM 3 first shipped in transformers main after 4.56.0. The project's
4.56.0 pin (for PT 2.4.1 compat) needs an audit at EC2 launch time — see
the [[pin_transformers]] memory.

Usage:
  python -m phase3_naip.naip_sam --shard 0/8                 # 1/8 of work
  python -m phase3_naip.naip_sam --limit 20                  # smoke test
  python -m phase3_naip.naip_sam --inspect --limit 4         # NAIP only, no SAM
  python -m phase3_naip.naip_sam --cluster-ids X,Y           # targeted
"""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import threading
import time
import traceback
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
import pyproj
import rasterio
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from rasterio.features import shapes
from rasterio.merge import merge as rio_merge
from rasterio.session import AWSSession
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling
from shapely import Polygon
from shapely.geometry import shape as shapely_shape
from shapely.ops import transform as shapely_transform

SITES_US = Path(__file__).resolve().parents[1]
load_dotenv(SITES_US / ".env")
DATA_US = SITES_US.parent / "data_us"
MANIFEST_PATH = DATA_US / "phase3_naip" / "naip_manifest.parquet"

# Output S3 location
OUTPUT_BUCKET = os.environ.get("PHASE3_NAIP_BUCKET", "industrials-scanner-us-west-2")
OUTPUT_PREFIX = os.environ.get("PHASE3_NAIP_PREFIX", "phase3-naip-sam3")
AWS_REGION = os.environ.get("AWS_REGION", "us-west-2")

# SAM 3 (Meta Promptable Concept Segmentation, HF transformers)
SAM3_MODEL_ID = os.environ.get("SAM3_MODEL_ID", "facebook/sam3")
SAM3_THRESHOLD = float(os.environ.get("SAM3_THRESHOLD", "0.5"))      # box score gate
SAM3_MASK_THRESHOLD = float(os.environ.get("SAM3_MASK_THRESHOLD", "0.5"))
SAM3_MIN_MASK_AREA_PX = int(os.environ.get("SAM3_MIN_MASK_AREA_PX", "200"))

# Prompt set is a calibration item (see NAIP_STAGE_NOTES.md). Each prompt produces
# its own labelled instance masks; downstream reasoning consumes the labels.
SAM3_PROMPTS: tuple[str, ...] = tuple(
    p.strip() for p in os.environ.get(
        "SAM3_PROMPTS",
        "industrial building,warehouse,parking lot,storage tank,silo"
    ).split(",") if p.strip()
)

# Reproject everything to CONUS Albers Equal Area; native res taken from source NAIP.
TARGET_CRS = "EPSG:5070"
LOCAL_TMP_RGB = Path(os.environ.get("PHASE3_NAIP_TMP", "/tmp/phase3_naip"))

# Telemetry — feeds post-run resize math. Per-cluster timings to STATS_LOG;
# system-wide CPU/RAM/GPU/disk/net sampled in a background thread to SYSTEM_LOG.
TELEMETRY_DIR = Path(os.environ.get(
    "PHASE3_NAIP_TELEMETRY_DIR",
    str(DATA_US / "phase3_naip" / "telemetry")))
STATS_LOG = TELEMETRY_DIR / "stats.jsonl"
SYSTEM_LOG = TELEMETRY_DIR / "system.jsonl"
SYSTEM_SAMPLE_INTERVAL_S = float(os.environ.get(
    "PHASE3_NAIP_SAMPLE_INTERVAL_S", "30"))
PROGRESS_WINDOW = int(os.environ.get("PHASE3_NAIP_PROGRESS_WINDOW", "50"))

# Batch tunables. The GPU is fed one N-image x M-prompt SAM forward per step.
# Per-cluster sync overhead — the cause of <16% GPU util in the prior per-prompt
# loop — collapses because we run one decoder call per batch instead of N*M.
# Reads run in parallel via a threadpool; writes (postproc + S3 upload) likewise.
# Max input dim caps the per-image pixel count fed to SAM so mosaic outliers
# (p90=2.3 MP, max=9.3 MP in prior telemetry) can't blow VRAM at batch.
#
# BATCH_N is auto-tuned at startup by probing N=1 and N=2 peak VRAM, then
# fitting how many clusters fit in GPU_BUDGET_BYTES. PHASE3_NAIP_BATCH_N (if
# set non-zero) overrides the probe with a fixed value. MAX_BATCH_N caps the
# auto-detected value as a safety net.
BATCH_N_OVERRIDE = int(os.environ.get("PHASE3_NAIP_BATCH_N", "0"))
MAX_BATCH_N = int(os.environ.get("PHASE3_NAIP_MAX_BATCH_N", "32"))
GPU_BUDGET_BYTES = int(os.environ.get("PHASE3_NAIP_GPU_BUDGET_BYTES",
                                      str(42 * 1024**3)))
READ_WORKERS = int(os.environ.get("PHASE3_NAIP_READ_WORKERS", "32"))
WRITE_WORKERS = int(os.environ.get("PHASE3_NAIP_WRITE_WORKERS", "8"))
MAX_INPUT_DIM = int(os.environ.get("PHASE3_NAIP_MAX_INPUT_DIM", "1024"))


# ---------- telemetry ------------------------------------------------------

try:
    import psutil  # type: ignore
    _HAS_PSUTIL = True
except ImportError:
    psutil = None  # type: ignore
    _HAS_PSUTIL = False


def _nvidia_smi() -> dict | None:
    """Returns dict with gpu_util_pct, gpu_mem_used_mb, gpu_mem_total_mb,
    gpu_temp_c, or None on systems with no nvidia-smi / no GPU."""
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2, check=True)
        # Single-GPU instance — take first line.
        line = out.stdout.strip().splitlines()[0]
        util, mu, mt, t = (float(x.strip()) for x in line.split(","))
        return {"gpu_util_pct": util, "gpu_mem_used_mb": mu,
                "gpu_mem_total_mb": mt, "gpu_temp_c": t}
    except (FileNotFoundError, subprocess.CalledProcessError,
            subprocess.TimeoutExpired, ValueError):
        return None


class SystemSampler(threading.Thread):
    """Background thread sampling host CPU/RAM/GPU/disk/net into SYSTEM_LOG.
    Also retains a rolling GPU util window for the progress lines."""

    def __init__(self, interval_s: float, log_path: Path):
        super().__init__(daemon=True)
        self.interval_s = interval_s
        self.log_path = log_path
        self._stop_evt = threading.Event()
        self._gpu_util_window: deque[float] = deque(maxlen=240)  # 240 * interval = 2 hr
        self._current_cluster: str | None = None
        self._lock = threading.Lock()
        self._prev_disk = None
        self._prev_net = None
        self._prev_ts = None

    def set_cluster(self, cluster_id: str | None):
        with self._lock:
            self._current_cluster = cluster_id

    def stop(self):
        self._stop_evt.set()

    def recent_gpu_util(self) -> float | None:
        """Mean GPU util % over the rolling window (None if no samples)."""
        if not self._gpu_util_window:
            return None
        return sum(self._gpu_util_window) / len(self._gpu_util_window)

    def run(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        f = self.log_path.open("a", buffering=1)
        try:
            while not self._stop_evt.wait(self.interval_s):
                self._sample(f)
        finally:
            f.close()

    def _sample(self, f):
        sample = {"ts": time.time()}
        if _HAS_PSUTIL:
            try:
                vm = psutil.virtual_memory()
                sample["cpu_pct"] = psutil.cpu_percent(interval=None)
                sample["ram_used_gb"] = vm.used / (1024 ** 3)
                sample["ram_pct"] = vm.percent
                # Per-interval disk + net rates
                d = psutil.disk_io_counters()
                n = psutil.net_io_counters()
                now = time.time()
                if self._prev_disk is not None and self._prev_ts is not None:
                    dt = max(now - self._prev_ts, 1e-3)
                    sample["disk_read_mb_s"] = (d.read_bytes - self._prev_disk.read_bytes) / dt / 1e6
                    sample["disk_write_mb_s"] = (d.write_bytes - self._prev_disk.write_bytes) / dt / 1e6
                    sample["net_recv_mb_s"] = (n.bytes_recv - self._prev_net.bytes_recv) / dt / 1e6
                    sample["net_send_mb_s"] = (n.bytes_sent - self._prev_net.bytes_sent) / dt / 1e6
                self._prev_disk = d
                self._prev_net = n
                self._prev_ts = now
            except Exception as e:
                sample["psutil_err"] = repr(e)
        gpu = _nvidia_smi()
        if gpu is not None:
            sample.update(gpu)
            self._gpu_util_window.append(gpu["gpu_util_pct"])
        with self._lock:
            if self._current_cluster:
                sample["in_flight_cluster_id"] = self._current_cluster
        f.write(json.dumps(sample) + "\n")


def append_stats_jsonl(row: dict, path: Path = STATS_LOG):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


def _pct(vals: list[float], q: float) -> float:
    if not vals:
        return 0.0
    return float(np.percentile(vals, q))


# ---------- AWS env --------------------------------------------------------

def _rasterio_env() -> rasterio.Env:
    """rasterio env wired for NAIP requester-pays reads, in-region us-west-2."""
    return rasterio.Env(
        AWSSession(boto3.Session(), requester_pays=True),
        AWS_REQUEST_PAYER="requester",
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        GDAL_HTTP_MULTIPLEX="YES",
        GDAL_HTTP_VERSION="2",
        VSI_CACHE="TRUE",
        VSI_CACHE_SIZE=200_000_000,
        CPL_VSIL_CURL_CHUNK_SIZE=1_048_576,
        CPL_VSIL_CURL_CACHE_SIZE=200_000_000,
    )


_s3_out_client = None


def s3_out_client():
    global _s3_out_client
    if _s3_out_client is None:
        _s3_out_client = boto3.client("s3", region_name=AWS_REGION)
    return _s3_out_client


def output_key(cluster_id: str) -> str:
    return f"{OUTPUT_PREFIX}/{cluster_id}/masks.parquet"


def output_exists(cluster_id: str) -> bool:
    try:
        s3_out_client().head_object(Bucket=OUTPUT_BUCKET,
                                    Key=output_key(cluster_id))
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] in {"404", "NoSuchKey", "Not Found"}:
            return False
        raise


def upload_parquet(df: pd.DataFrame, cluster_id: str) -> str:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    key = output_key(cluster_id)
    s3_out_client().put_object(Bucket=OUTPUT_BUCKET, Key=key, Body=buf.getvalue())
    return f"s3://{OUTPUT_BUCKET}/{key}"


# ---------- SAM 3 lazy load ------------------------------------------------

_sam3 = None  # (model, processor, device)


def get_sam3():
    """Lazy-load SAM 3 and cache (a) the model + processor + device,
    (b) the tokenized prompt tensors on device (M is constant across clusters)."""
    global _sam3
    if _sam3 is not None:
        return _sam3
    import torch
    from transformers import Sam3Model, Sam3Processor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[naip-sam3] loading {SAM3_MODEL_ID} on {device}", flush=True)
    t0 = time.time()
    model = Sam3Model.from_pretrained(SAM3_MODEL_ID).to(device).eval()
    processor = Sam3Processor.from_pretrained(SAM3_MODEL_ID)
    # Tokenize the M prompts once. input_ids: [M, L], attn_mask: [M, L].
    text_inputs = processor(text=list(SAM3_PROMPTS), return_tensors="pt",
                            padding=True).to(device)
    _sam3 = (model, processor, device, text_inputs)
    print(f"[naip-sam3] SAM 3 ready ({time.time()-t0:.1f}s); "
          f"prompts cached on {device} (M={len(SAM3_PROMPTS)}, "
          f"L={text_inputs.input_ids.shape[1]})", flush=True)
    return _sam3


def run_sam3_batch(rgb_images: list[np.ndarray]) -> list[list[dict]]:
    """Run SAM 3 over a batch of N RGB images against all M prompts in ONE
    forward pass through the decoder. Returns a list of N lists of mask dicts
    (one inner list per input image, flattened across prompts).

    Batch construction: vision_embeds repeated M times along batch dim;
    cached prompt tokens tiled N times. Combined batch dim is N*M; row k
    corresponds to (image_idx = k // M, prompt_idx = k % M).
    """
    import torch
    from PIL import Image
    from transformers.models.sam3.modeling_sam3 import Sam3VisionEncoderOutput
    model, processor, device, text_tok = get_sam3()
    N = len(rgb_images)
    M = len(SAM3_PROMPTS)
    if N == 0:
        return []

    pils = [Image.fromarray(img).convert("RGB") for img in rgb_images]
    sizes = [(int(img.shape[0]), int(img.shape[1])) for img in rgb_images]

    img_inputs = processor(images=pils, return_tensors="pt").to(device)
    with torch.no_grad():
        ve = model.get_vision_features(pixel_values=img_inputs.pixel_values)
    fpn_rep = [t.repeat_interleave(M, dim=0) for t in ve.fpn_hidden_states]
    fpn_pos_rep = [t.repeat_interleave(M, dim=0) for t in ve.fpn_position_encoding]
    ve_rep = Sam3VisionEncoderOutput(
        fpn_hidden_states=fpn_rep, fpn_position_encoding=fpn_pos_rep)
    input_ids = text_tok.input_ids.repeat(N, 1)
    attn = text_tok.attention_mask.repeat(N, 1)
    target_sizes = [sz for sz in sizes for _ in range(M)]  # repeat each N*M

    with torch.no_grad():
        out = model(vision_embeds=ve_rep, input_ids=input_ids, attention_mask=attn)
    results = processor.post_process_instance_segmentation(
        out, threshold=SAM3_THRESHOLD, mask_threshold=SAM3_MASK_THRESHOLD,
        target_sizes=target_sizes,
    )

    per_image: list[list[dict]] = [[] for _ in range(N)]
    for n in range(N):
        for j, prompt in enumerate(SAM3_PROMPTS):
            r = results[n * M + j]
            masks = r["masks"]
            scores = r["scores"]
            boxes = r.get("boxes")
            for i in range(len(scores)):
                m = masks[i].detach().cpu().numpy().astype(bool)
                if m.sum() < SAM3_MIN_MASK_AREA_PX:
                    continue
                box = (boxes[i].detach().cpu().numpy().tolist()
                       if boxes is not None else None)
                per_image[n].append({"label": prompt,
                                     "score": float(scores[i].item()),
                                     "mask_bool": m, "box": box})
    return per_image


def probe_batch_n() -> int:
    """Probe peak VRAM at N=1 and N=2; fit how many clusters fit GPU_BUDGET_BYTES.
    Returns the chosen effective batch size, clamped to [1, MAX_BATCH_N]."""
    if BATCH_N_OVERRIDE > 0:
        print(f"[naip-sam3] batch_n override = {BATCH_N_OVERRIDE}", flush=True)
        return BATCH_N_OVERRIDE
    import torch
    if not torch.cuda.is_available():
        return 1
    _, _, device, _ = get_sam3()
    # Synthetic RGB at the worst-case mosaic dim — same shape SAM 3 processor
    # will resize to internally (1008x1008), so probe cost ≈ runtime cost.
    dummy = (np.random.rand(MAX_INPUT_DIM, MAX_INPUT_DIM, 3) * 255).astype(np.uint8)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    run_sam3_batch([dummy])
    peak1 = torch.cuda.max_memory_allocated(device)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    run_sam3_batch([dummy, dummy])
    peak2 = torch.cuda.max_memory_allocated(device)

    baseline = max(0, 2 * peak1 - peak2)  # extrapolate intercept
    marginal = max(1, peak2 - peak1)
    fit_n = (GPU_BUDGET_BYTES - baseline) // marginal
    chosen = max(1, min(MAX_BATCH_N, int(fit_n)))
    print(f"[naip-sam3] vram probe: peak@N=1={peak1/1e9:.2f}GB "
          f"peak@N=2={peak2/1e9:.2f}GB marginal={marginal/1e9:.2f}GB/cluster "
          f"budget={GPU_BUDGET_BYTES/1e9:.1f}GB -> N={chosen}", flush=True)
    torch.cuda.empty_cache()
    return chosen


# ---------- NAIP read + mosaic ---------------------------------------------

def _bbox_4326_to_5070(lon_min, lat_min, lon_max, lat_max):
    tr = pyproj.Transformer.from_crs(4326, 5070, always_xy=True).transform
    xs, ys = tr([lon_min, lon_max, lon_max, lon_min],
                [lat_min, lat_min, lat_max, lat_max])
    return min(xs), min(ys), max(xs), max(ys)


def read_naip_mosaic(naip_uris: list[str],
                     fetch_lon_min: float, fetch_lat_min: float,
                     fetch_lon_max: float, fetch_lat_max: float
                     ) -> tuple[np.ndarray, rasterio.Affine] | None:
    """Open each NAIP COG, warp to EPSG:5070, mosaic to one (4, H, W) array.

    Returns (array, transform) in EPSG:5070, or None if no valid data.
    """
    xmin, ymin, xmax, ymax = _bbox_4326_to_5070(
        fetch_lon_min, fetch_lat_min, fetch_lon_max, fetch_lat_max)
    target_bounds = (xmin, ymin, xmax, ymax)

    srcs = []
    vrts = []
    try:
        for uri in naip_uris:
            try:
                src = rasterio.open(uri)
            except Exception as e:
                print(f"[naip-sam]   open failed {uri}: {e!r}", flush=True)
                continue
            srcs.append(src)
            native_res_x = abs(src.transform.a)
            vrt = WarpedVRT(src, crs=TARGET_CRS, resampling=Resampling.bilinear,
                            src_nodata=src.nodata)
            vrts.append((vrt, native_res_x))
        if not vrts:
            return None

        # Native res of the first source (per state-year all NAIP same res).
        # Cap output dim at MAX_INPUT_DIM: if a cluster span at native res would
        # exceed the cap, downsample. SAM 3 internally resizes to ~1008 anyway;
        # super-high-res inputs only inflate postproc memory without adding info.
        native_res = vrts[0][1]
        span_x_m = xmax - xmin
        span_y_m = ymax - ymin
        min_res = max(span_x_m, span_y_m) / MAX_INPUT_DIM
        res = max(native_res, min_res)
        arr, transform = rio_merge([v for v, _ in vrts], bounds=target_bounds,
                                   res=res, method="first", nodata=0)
        return arr, transform
    finally:
        for v, _ in vrts:
            try:
                v.close()
            except Exception:
                pass
        for s in srcs:
            try:
                s.close()
            except Exception:
                pass


def to_uint8_rgb(arr_chw: np.ndarray) -> np.ndarray:
    """RGB (4-band -> first 3), per-band 1-99 percentile stretch -> uint8 HWC."""
    rgb = arr_chw[:3].astype(np.float32)
    out = np.empty_like(rgb, dtype=np.uint8)
    for b in range(3):
        v = rgb[b]
        m = v > 0
        if m.sum() == 0:
            out[b] = 0
            continue
        lo, hi = np.percentile(v[m], (1, 99))
        if hi <= lo:
            hi = lo + 1
        out[b] = np.clip((v - lo) / (hi - lo), 0, 1).__mul__(255).astype(np.uint8)
    return np.transpose(out, (1, 2, 0))


# ---------- mask -> polygons + features ------------------------------------

def _polygon_shape_features(poly: Polygon) -> dict:
    if poly.is_empty or not poly.is_valid:
        return {"rectangularity": 0.0, "aspect_ratio": 0.0, "elongation": 0.0}
    minx, miny, maxx, maxy = poly.bounds
    bbox_w = max(maxx - minx, 1e-6)
    bbox_h = max(maxy - miny, 1e-6)
    bbox_area = bbox_w * bbox_h
    rectangularity = poly.area / bbox_area
    aspect = max(bbox_w, bbox_h) / min(bbox_w, bbox_h)
    # Elongation via minimum rotated rectangle
    try:
        mrr = poly.minimum_rotated_rectangle
        mb = list(mrr.exterior.coords)
        d1 = np.hypot(mb[1][0] - mb[0][0], mb[1][1] - mb[0][1])
        d2 = np.hypot(mb[2][0] - mb[1][0], mb[2][1] - mb[1][1])
        elong = max(d1, d2) / max(min(d1, d2), 1e-6)
    except Exception:
        elong = aspect
    return {"rectangularity": float(rectangularity),
            "aspect_ratio": float(aspect),
            "elongation": float(elong)}


def masks_to_features(masks: list[dict],
                      mosaic_5070: np.ndarray,
                      transform_5070: rasterio.Affine,
                      ) -> pd.DataFrame:
    """Convert SAM 3 labelled masks (label, score, mask_bool, box) to polygons
    (in EPSG:5070) with per-mask features. Returns one row per mask."""
    rgb_nir = mosaic_5070  # (4, H, W), original radiometry
    to_4326 = pyproj.Transformer.from_crs(5070, 4326, always_xy=True).transform
    rows = []
    for i, m in enumerate(masks):
        seg = m["mask_bool"].astype(np.uint8)
        polys_5070 = []
        for geom, _val in shapes(seg, mask=seg.astype(bool),
                                 transform=transform_5070):
            p = shapely_shape(geom)
            if p.is_valid and not p.is_empty:
                polys_5070.append(p)
        if not polys_5070:
            continue
        poly_5070 = max(polys_5070, key=lambda p: p.area)
        shape_feats = _polygon_shape_features(poly_5070)

        mask_bool = m["mask_bool"]
        n_px = int(mask_bool.sum())
        if n_px == 0:
            mean_r = mean_g = mean_b = mean_nir = 0.0
        else:
            mean_r = float(rgb_nir[0][mask_bool].mean())
            mean_g = float(rgb_nir[1][mask_bool].mean())
            mean_b = float(rgb_nir[2][mask_bool].mean())
            mean_nir = (float(rgb_nir[3][mask_bool].mean())
                        if rgb_nir.shape[0] >= 4 else 0.0)

        poly_4326 = shapely_transform(lambda x, y: to_4326(x, y), poly_5070)
        b = poly_4326.bounds

        rows.append(dict(
            mask_id=i,
            label=m["label"],
            label_score=float(m["score"]),
            area_m2=float(poly_5070.area),
            area_px=n_px,
            lon_min=float(b[0]), lat_min=float(b[1]),
            lon_max=float(b[2]), lat_max=float(b[3]),
            mean_r=mean_r, mean_g=mean_g, mean_b=mean_b, mean_nir=mean_nir,
            rectangularity=shape_feats["rectangularity"],
            aspect_ratio=shape_feats["aspect_ratio"],
            elongation=shape_feats["elongation"],
            geom_wkt=poly_4326.wkt,
        ))
    return pd.DataFrame(rows)


# ---------- write worker (process-pool target) -----------------------------
# Module-level so it pickles. The batched main loop submits these into a
# spawn'd ProcessPoolExecutor — masks_to_features is GIL-bound (shapely,
# pyproj, _polygon_shape_features) and threads do not parallelize it
# (empirical: 8 threads = 1.07x serial; 8 processes = >3x).

def write_one(cid: str, masks: list[dict], arr: np.ndarray,
              tr: rasterio.Affine, h: int, w: int, row_meta: dict,
              read_s: float, sam_s: float) -> dict:
    try:
        t2 = time.time()
        df_out = masks_to_features(masks, arr, tr)
        postproc_s = time.time() - t2
        t3 = time.time()
        upload_parquet(df_out, cid)
        upload_s = time.time() - t3
        return {
            "cluster_id": cid,
            "candidate_id": row_meta.get("candidate_id"),
            "n_buildings": int(row_meta.get("n_buildings", 0) or 0),
            "n_tiles": int(row_meta.get("n_tiles", 0) or 0),
            "span_m": int(row_meta.get("span_m", 0) or 0),
            "h": h, "w": w, "mosaic_mp": h * w / 1e6,
            "read_s": read_s, "sam_s": sam_s,
            "postproc_s": postproc_s, "upload_s": upload_s,
            "total_s": read_s + sam_s + postproc_s + upload_s,
            "n_masks": len(df_out), "status": "ok",
        }
    except Exception as e:
        traceback.print_exc()
        return {"cluster_id": cid, "status": "post_exception",
                "error": repr(e), "read_s": read_s}


# ---------- main worker ----------------------------------------------------

def process_one(row: pd.Series, run_sam: bool = True,
                inspect_dir: Path | None = None) -> dict:
    """Process a single cluster. Returns a structured status dict with
    per-stage timings + mosaic dims (for the telemetry log)."""
    cid = row.cluster_id
    naip_uris = list(row.naip_uris)
    base = {"cluster_id": cid, "candidate_id": getattr(row, "candidate_id", None),
            "n_buildings": int(getattr(row, "n_buildings", 0) or 0),
            "n_tiles": int(len(naip_uris)),
            "span_m": int(getattr(row, "span_m", 0) or 0),
            "ts_start": time.time()}
    if not naip_uris:
        return {**base, "status": "no_naip_tiles"}

    t0 = time.time()
    mosaic = read_naip_mosaic(
        naip_uris,
        float(row.fetch_lon_min), float(row.fetch_lat_min),
        float(row.fetch_lon_max), float(row.fetch_lat_max))
    if mosaic is None:
        return {**base, "status": "read_failed",
                "read_s": time.time() - t0}
    arr, tr = mosaic
    rgb_hwc = to_uint8_rgb(arr)
    read_s = time.time() - t0
    h, w = rgb_hwc.shape[:2]
    base.update({"h": h, "w": w, "mosaic_mp": h * w / 1e6, "read_s": read_s})

    if inspect_dir is not None:
        from PIL import Image
        inspect_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb_hwc).save(inspect_dir / f"{cid}.png")
        return {**base, "status": "inspected"}

    if not run_sam:
        return {**base, "status": "skipped_sam"}

    t1 = time.time()
    masks = run_sam3_batch([rgb_hwc])[0]
    sam_s = time.time() - t1
    t2 = time.time()
    df = masks_to_features(masks, arr, tr)
    postproc_s = time.time() - t2
    t3 = time.time()
    upload_parquet(df, cid)
    upload_s = time.time() - t3
    return {**base, "status": "ok", "n_masks": len(df),
            "sam_s": sam_s, "postproc_s": postproc_s, "upload_s": upload_s,
            "total_s": read_s + sam_s + postproc_s + upload_s}


def _run_serial(df: pd.DataFrame, args, inspect_dir: Path | None) -> int:
    """Serial fallback for --inspect / --no-sam diagnostic modes. The 3-stage
    pipeline is not needed when we're skipping SAM."""
    n_ok = n_fail = 0
    with _rasterio_env():
        for i, row in df.iterrows():
            cid = row.cluster_id
            try:
                r = process_one(row, run_sam=not args.no_sam,
                                inspect_dir=inspect_dir)
                append_stats_jsonl(r)
                if r["status"] in {"inspected", "skipped_sam"}:
                    n_ok += 1
                    print(f"[naip-sam3] {i+1}/{len(df)} {cid}: "
                          f"{r['status']} h={r.get('h')} w={r.get('w')} "
                          f"read={r.get('read_s', 0):.1f}s", flush=True)
                else:
                    n_fail += 1
                    print(f"[naip-sam3] {i+1}/{len(df)} {cid}: {r['status']}",
                          flush=True)
            except Exception as e:
                n_fail += 1
                append_stats_jsonl({"cluster_id": cid, "status": "exception",
                                    "error": repr(e), "ts_start": time.time()})
                print(f"[naip-sam3] {i+1}/{len(df)} {cid} FAILED: {e!r}",
                      flush=True)
                traceback.print_exc()
    print(f"[naip-sam3] serial done: {n_ok} ok / {n_fail} failed")
    return 0 if n_fail == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=str, default="0/1",
                    help="N/M: process the N-th of M equal slices of the manifest")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap on sub-clusters processed (after sharding)")
    ap.add_argument("--cluster-ids", type=str, default=None,
                    help="comma-separated cluster_ids to process (overrides shard)")
    ap.add_argument("--inspect", action="store_true",
                    help="read NAIP and save RGB PNG locally; skip SAM and upload")
    ap.add_argument("--inspect-dir", type=str, default=str(LOCAL_TMP_RGB / "inspect"))
    ap.add_argument("--no-sam", action="store_true",
                    help="read NAIP only; skip SAM (for testing the read path)")
    ap.add_argument("--manifest", type=str, default=str(MANIFEST_PATH))
    args = ap.parse_args()

    df = pd.read_parquet(args.manifest)
    df = df[df.naip_uris.apply(len) > 0].reset_index(drop=True)

    if args.cluster_ids:
        ids = set(args.cluster_ids.split(","))
        df = df[df.cluster_id.isin(ids)].reset_index(drop=True)
    else:
        n, m = (int(x) for x in args.shard.split("/"))
        if not 0 <= n < m:
            raise SystemExit(f"--shard {args.shard} invalid")
        df = df.iloc[n::m].reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)
    print(f"[naip-sam3] {len(df):,} clusters to process "
          f"(prompts: {SAM3_PROMPTS})", flush=True)

    inspect_dir = Path(args.inspect_dir) if args.inspect else None
    if args.inspect or args.no_sam:
        # Single-threaded path for the diagnostic modes: no need to wire up the
        # 3-stage pipeline just to dump RGB PNGs or test the read path.
        return _run_serial(df, args, inspect_dir)

    t_start = time.time()
    sampler = SystemSampler(SYSTEM_SAMPLE_INTERVAL_S, SYSTEM_LOG)
    sampler.start()
    print(f"[naip-sam3] telemetry -> {STATS_LOG} | {SYSTEM_LOG} "
          f"(sample every {SYSTEM_SAMPLE_INTERVAL_S:.0f}s)", flush=True)
    print(f"[naip-sam3] read_workers={READ_WORKERS} "
          f"write_workers={WRITE_WORKERS} max_input_dim={MAX_INPUT_DIM} "
          f"gpu_budget={GPU_BUDGET_BYTES/1e9:.1f}GB", flush=True)

    get_sam3()  # eager-load on main thread before the first batch
    batch_n = probe_batch_n()

    n_rows = len(df)
    rows = list(df.itertuples(index=False))
    n_done = n_ok = n_skip = n_fail = 0
    all_read: list[float] = []
    all_sam: list[float] = []          # amortized: batch_wall / batch_size
    all_postproc: list[float] = []
    all_upload: list[float] = []
    all_total: list[float] = []
    all_mosaic_mp: list[float] = []
    window_read: deque[float] = deque(maxlen=PROGRESS_WINDOW)
    window_sam: deque[float] = deque(maxlen=PROGRESS_WINDOW)
    window_total: deque[float] = deque(maxlen=PROGRESS_WINDOW)

    def read_one(row):
        cid = row.cluster_id
        if output_exists(cid):
            return {"row": row, "cid": cid, "status": "skip"}
        t0 = time.time()
        try:
            with _rasterio_env():
                mos = read_naip_mosaic(
                    list(row.naip_uris),
                    float(row.fetch_lon_min), float(row.fetch_lat_min),
                    float(row.fetch_lon_max), float(row.fetch_lat_max))
        except Exception as e:
            return {"row": row, "cid": cid, "status": "read_exception",
                    "error": repr(e), "read_s": time.time() - t0}
        if mos is None:
            return {"row": row, "cid": cid, "status": "read_failed",
                    "read_s": time.time() - t0}
        arr, tr = mos
        rgb = to_uint8_rgb(arr)
        return {"row": row, "cid": cid, "status": "ok",
                "arr": arr, "tr": tr, "rgb": rgb,
                "read_s": time.time() - t0}

    def record(rec):
        nonlocal n_ok, n_fail, n_done
        append_stats_jsonl(rec)
        n_done += 1
        if rec.get("status") == "ok":
            n_ok += 1
            all_read.append(rec["read_s"])
            all_sam.append(rec["sam_s"])
            all_postproc.append(rec["postproc_s"])
            all_upload.append(rec["upload_s"])
            all_total.append(rec["total_s"])
            all_mosaic_mp.append(rec["mosaic_mp"])
            window_read.append(rec["read_s"])
            window_sam.append(rec["sam_s"])
            window_total.append(rec["total_s"])
        else:
            n_fail += 1
        if n_done % PROGRESS_WINDOW == 0 or n_done == n_rows:
            elapsed = max(time.time() - t_start, 1e-3)
            rate = n_done / elapsed
            remaining = (n_rows - n_done) / max(rate, 1e-6)
            gpu = sampler.recent_gpu_util()
            gpu_str = f"gpu={gpu:.0f}%" if gpu is not None else "gpu=n/a"
            print(f"[naip-sam3] {n_done}/{n_rows} {rec.get('cluster_id','?')}: "
                  f"{rec.get('n_masks', 0)} masks "
                  f"read p50/p90={_pct(list(window_read),50):.1f}/"
                  f"{_pct(list(window_read),90):.1f}s "
                  f"sam p50/p90={_pct(list(window_sam),50):.2f}/"
                  f"{_pct(list(window_sam),90):.2f}s "
                  f"total p50={_pct(list(window_total),50):.1f}s "
                  f"{gpu_str} ({rate:.2f} cl/s, ETA {remaining/60:.0f} min)",
                  flush=True)

    read_pool = ThreadPoolExecutor(max_workers=READ_WORKERS,
                                   thread_name_prefix="read")
    # Process pool for postproc: masks_to_features is GIL-bound (Python +
    # shapely + pyproj per-vertex transforms). Threads serialize on the GIL;
    # processes don't. Spawn context (not fork) so children don't inherit
    # CUDA state from the SAM-loaded parent.
    write_pool = ProcessPoolExecutor(max_workers=WRITE_WORKERS,
                                     mp_context=get_context("spawn"))
    read_futs: deque = deque()
    write_futs: deque = deque()
    idx = 0

    def submit_more_reads():
        nonlocal idx
        while idx < n_rows and len(read_futs) < 2 * batch_n:
            read_futs.append(read_pool.submit(read_one, rows[idx]))
            idx += 1

    try:
        submit_more_reads()
        while read_futs:
            # Pull next batch_n completed reads; handle skips/failures inline.
            batch = []
            while read_futs and len(batch) < batch_n:
                it = read_futs.popleft().result()
                if it["status"] == "skip":
                    n_skip += 1; n_done += 1
                elif it["status"] != "ok":
                    record({"cluster_id": it["cid"], "status": it["status"],
                            "error": it.get("error"),
                            "read_s": it.get("read_s", 0)})
                    print(f"[naip-sam3] {it['cid']}: {it['status']}", flush=True)
                else:
                    batch.append(it)
            submit_more_reads()
            if not batch:
                continue

            t_sam = time.time()
            try:
                per_image_masks = run_sam3_batch([it["rgb"] for it in batch])
            except Exception as e:
                traceback.print_exc()
                for it in batch:
                    record({"cluster_id": it["cid"], "status": "sam_exception",
                            "error": repr(e), "read_s": it["read_s"]})
                print(f"[naip-sam3] batch (N={len(batch)}): sam_exception {e!r}",
                      flush=True)
                continue
            sam_per = (time.time() - t_sam) / max(1, len(batch))

            for it, m in zip(batch, per_image_masks):
                row = it["row"]
                h, w = it["rgb"].shape[:2]
                row_meta = {
                    "candidate_id": getattr(row, "candidate_id", None),
                    "n_buildings": getattr(row, "n_buildings", 0) or 0,
                    "n_tiles": len(row.naip_uris),
                    "span_m": getattr(row, "span_m", 0) or 0,
                }
                write_futs.append(write_pool.submit(
                    write_one, it["cid"], m, it["arr"], it["tr"],
                    h, w, row_meta, it["read_s"], sam_per))

            # Bound pending writes so memory stays predictable.
            while len(write_futs) > 4 * batch_n:
                record(write_futs.popleft().result())

        # Drain remaining writes
        while write_futs:
            record(write_futs.popleft().result())
    finally:
        read_pool.shutdown(wait=True)
        write_pool.shutdown(wait=True)
        sampler.stop()
        sampler.join(timeout=SYSTEM_SAMPLE_INTERVAL_S + 5)

    elapsed = time.time() - t_start
    print(f"\n[naip-sam3] done: {n_ok} ok / {n_skip} skipped (cached) / "
          f"{n_fail} failed in {elapsed:.1f}s")

    if all_total:
        sum_read = sum(all_read); sum_sam = sum(all_sam)
        sum_postproc = sum(all_postproc); sum_upload = sum(all_upload)
        stage_sum = sum_read + sum_sam + sum_postproc + sum_upload
        budget = (("read", sum_read), ("sam (amortized)", sum_sam),
                  ("postproc", sum_postproc), ("upload", sum_upload))
        print("[naip-sam3] === per-cluster timing ===")
        print(f"  total_s p50/p90/p99/max = "
              f"{_pct(all_total,50):.1f}/{_pct(all_total,90):.1f}/"
              f"{_pct(all_total,99):.1f}/{max(all_total):.1f}")
        print(f"  read_s  p50/p90/max     = "
              f"{_pct(all_read,50):.1f}/{_pct(all_read,90):.1f}/{max(all_read):.1f}")
        print(f"  sam_s amortized p50/p90/max = "
              f"{_pct(all_sam,50):.2f}/{_pct(all_sam,90):.2f}/"
              f"{max(all_sam):.2f}")
        print(f"  mosaic megapixels p50/p90/max = "
              f"{_pct(all_mosaic_mp,50):.1f}/{_pct(all_mosaic_mp,90):.1f}/"
              f"{max(all_mosaic_mp):.1f}")
        print("[naip-sam3] === stage time (summed across clusters) ===")
        for name, t in budget:
            pct = 100.0 * t / stage_sum if stage_sum > 0 else 0
            print(f"  {name:<16} {t:>8.1f}s ({pct:>5.1f}% of stage time)")
        overlap = stage_sum / elapsed if elapsed > 0 else 1.0
        print(f"[naip-sam3] elapsed wall = {elapsed:.1f}s | summed stages = "
              f"{stage_sum:.1f}s | overlap factor = {overlap:.2f}x")
        gpu = sampler.recent_gpu_util()
        if gpu is not None:
            print(f"[naip-sam3] mean GPU util (recent window): {gpu:.1f}%")

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
