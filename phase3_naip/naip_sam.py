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
        "industrial building,warehouse,office building,parking lot,"
        "loading dock,storage tank,silo,vegetation,road"
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
    _sam3 = (model, processor, device)
    print(f"[naip-sam3] SAM 3 ready ({time.time()-t0:.1f}s)", flush=True)
    return _sam3


def run_sam3(image_hwc: np.ndarray) -> list[dict]:
    """Run SAM 3 over the full SAM3_PROMPTS list against `image_hwc` (uint8 HWC).

    Returns flat list of {label, score, mask_bool [H,W], box [x1,y1,x2,y2]}.
    """
    import torch
    from PIL import Image
    model, processor, device = get_sam3()
    pil = Image.fromarray(image_hwc).convert("RGB")
    h, w = image_hwc.shape[:2]
    out: list[dict] = []
    for prompt in SAM3_PROMPTS:
        inputs = processor(images=pil, text=prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            res = model(**inputs)
        results = processor.post_process_instance_segmentation(
            res, threshold=SAM3_THRESHOLD, mask_threshold=SAM3_MASK_THRESHOLD,
            target_sizes=[(h, w)],
        )[0]
        masks = results["masks"]
        scores = results["scores"]
        boxes = results.get("boxes")
        for i in range(len(scores)):
            m = masks[i].detach().cpu().numpy().astype(bool)
            if m.sum() < SAM3_MIN_MASK_AREA_PX:
                continue
            box = boxes[i].detach().cpu().numpy().tolist() if boxes is not None else None
            out.append({"label": prompt, "score": float(scores[i].item()),
                        "mask_bool": m, "box": box})
    return out


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

        # Use the native res of the first source (per state-year all NAIP same res).
        res = vrts[0][1]
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
    masks = run_sam3(rgb_hwc)
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
    n_ok = n_skip = n_fail = 0
    t_start = time.time()

    sampler = SystemSampler(SYSTEM_SAMPLE_INTERVAL_S, SYSTEM_LOG)
    sampler.start()
    print(f"[naip-sam3] telemetry -> {STATS_LOG} | {SYSTEM_LOG} "
          f"(sample every {SYSTEM_SAMPLE_INTERVAL_S:.0f}s)", flush=True)

    # Rolling window for progress lines + end summary.
    window_read: deque[float] = deque(maxlen=PROGRESS_WINDOW)
    window_sam: deque[float] = deque(maxlen=PROGRESS_WINDOW)
    window_total: deque[float] = deque(maxlen=PROGRESS_WINDOW)
    all_read: list[float] = []
    all_sam: list[float] = []
    all_postproc: list[float] = []
    all_upload: list[float] = []
    all_total: list[float] = []
    all_mosaic_mp: list[float] = []
    sum_read = sum_sam = sum_postproc = sum_upload = 0.0

    try:
        with _rasterio_env():
            for i, row in df.iterrows():
                cid = row.cluster_id
                if not args.inspect and not args.no_sam:
                    if output_exists(cid):
                        n_skip += 1
                        continue
                sampler.set_cluster(cid)
                try:
                    r = process_one(row, run_sam=not args.no_sam,
                                    inspect_dir=inspect_dir)
                    append_stats_jsonl(r)
                    if r["status"] == "ok":
                        n_ok += 1
                        window_read.append(r["read_s"])
                        window_sam.append(r["sam_s"])
                        window_total.append(r["total_s"])
                        all_read.append(r["read_s"])
                        all_sam.append(r["sam_s"])
                        all_postproc.append(r["postproc_s"])
                        all_upload.append(r["upload_s"])
                        all_total.append(r["total_s"])
                        all_mosaic_mp.append(r["mosaic_mp"])
                        sum_read += r["read_s"]
                        sum_sam += r["sam_s"]
                        sum_postproc += r["postproc_s"]
                        sum_upload += r["upload_s"]
                        done = n_ok + n_skip
                        if done % PROGRESS_WINDOW == 0 or n_ok <= 5:
                            elapsed = max(time.time() - t_start, 1e-3)
                            rate = done / elapsed
                            remaining = (len(df) - done) / max(rate, 1e-6)
                            gpu = sampler.recent_gpu_util()
                            gpu_str = f"gpu={gpu:.0f}%" if gpu is not None else "gpu=n/a"
                            print(f"[naip-sam3] {i+1}/{len(df)} {cid}: "
                                  f"{r['n_masks']} masks "
                                  f"read p50/p90={_pct(list(window_read),50):.1f}/"
                                  f"{_pct(list(window_read),90):.1f}s "
                                  f"sam p50/p90={_pct(list(window_sam),50):.1f}/"
                                  f"{_pct(list(window_sam),90):.1f}s "
                                  f"total p50={_pct(list(window_total),50):.1f}s "
                                  f"{gpu_str} "
                                  f"({rate:.2f} cl/s, ETA {remaining/60:.0f} min)",
                                  flush=True)
                    elif r["status"] in {"inspected", "skipped_sam"}:
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
                    err_row = {"cluster_id": cid, "status": "exception",
                               "error": repr(e), "ts_start": time.time()}
                    append_stats_jsonl(err_row)
                    print(f"[naip-sam3] {i+1}/{len(df)} {cid} FAILED: {e!r}",
                          flush=True)
                    traceback.print_exc()
                finally:
                    sampler.set_cluster(None)
    finally:
        sampler.stop()
        sampler.join(timeout=SYSTEM_SAMPLE_INTERVAL_S + 5)

    elapsed = time.time() - t_start
    print(f"\n[naip-sam3] done: {n_ok} ok / {n_skip} skipped (cached) / "
          f"{n_fail} failed in {elapsed:.1f}s")

    if all_total:
        total_wall = sum(all_total)
        budget = (("read", sum_read), ("sam", sum_sam),
                  ("postproc", sum_postproc), ("upload", sum_upload))
        print("[naip-sam3] === per-cluster timing ===")
        print(f"  total_s p50/p90/p99/max = "
              f"{_pct(all_total,50):.1f}/{_pct(all_total,90):.1f}/"
              f"{_pct(all_total,99):.1f}/{max(all_total):.1f}")
        print(f"  read_s  p50/p90/max     = "
              f"{_pct(all_read,50):.1f}/{_pct(all_read,90):.1f}/{max(all_read):.1f}")
        print(f"  sam_s   p50/p90/max     = "
              f"{_pct(all_sam,50):.1f}/{_pct(all_sam,90):.1f}/{max(all_sam):.1f}")
        print(f"  mosaic megapixels p50/p90/max = "
              f"{_pct(all_mosaic_mp,50):.1f}/{_pct(all_mosaic_mp,90):.1f}/"
              f"{max(all_mosaic_mp):.1f}")
        print("[naip-sam3] === time-budget split ===")
        for name, t in budget:
            pct = 100.0 * t / total_wall if total_wall > 0 else 0
            print(f"  {name:<10} {t:>8.1f}s ({pct:>5.1f}%)")
        idle = max(0.0, elapsed - total_wall)
        print(f"  {'idle':<10} {idle:>8.1f}s "
              f"({100.0 * idle / max(elapsed, 1e-3):>5.1f}%)")

        gpu = sampler.recent_gpu_util()
        if gpu is not None:
            print(f"[naip-sam3] mean GPU util (recent window): {gpu:.1f}%")

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
