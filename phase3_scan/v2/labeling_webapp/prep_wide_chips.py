"""Pre-render wider context chips (5120m / 2x zoom-out) for each queued candidate.

Reads existing queue.json (so it uses the same scene_id per candidate). Saves to
chips_wide/{tile_id}.png. Skips if file already exists.
"""
from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyproj
import rasterio
from PIL import Image
from rasterio.windows import from_bounds

os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")

ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = ROOT / ".artifacts" / "labeling_v2"
QUEUE_PATH = ARTIFACTS / "queue.json"
WIDE_DIR = ARTIFACTS / "chips_wide"

DATA_US = ROOT.parent / "data_us"
SCENES_PATH = DATA_US / "phase3_scenes.parquet"

IMG_NATIVE = 512        # 2x the regular chip → 5120 m on a side
GSD_M = 10.0
HALF_M = (IMG_NATIVE / 2) * GSD_M
RENDER_PX = 512
WORKERS = 8


def _utm_epsg(mgrs_tile: str) -> int:
    zone = int(mgrs_tile[:-3])
    return 32600 + zone


def render_chip(b_paths, lat, lon, epsg, out_path):
    if out_path.exists():
        return True, None
    try:
        to_utm = pyproj.Transformer.from_crs(4326, epsg, always_xy=True).transform
        x, y = to_utm(lon, lat)
        xmin, ymin, xmax, ymax = x - HALF_M, y - HALF_M, x + HALF_M, y + HALF_M
        bands = []
        for path in b_paths:
            with rasterio.open(path) as src:
                arr = src.read(
                    1,
                    window=from_bounds(xmin, ymin, xmax, ymax, transform=src.transform),
                    out_shape=(IMG_NATIVE, IMG_NATIVE),
                    boundless=True, fill_value=0,
                )
            bands.append(arr.astype(np.float32))
        rgb = np.stack(bands, axis=0)
        lo, hi = np.percentile(rgb, (1, 99))
        if hi <= lo:
            hi = lo + 1
        arr = np.clip((rgb - lo) / (hi - lo), 0, 1) * 255
        arr = arr.astype(np.uint8).transpose(1, 2, 0)
        img = Image.fromarray(arr).resize((RENDER_PX, RENDER_PX), Image.LANCZOS)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path, "PNG", optimize=False)
        return True, None
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def main() -> int:
    queue = json.loads(QUEUE_PATH.read_text())
    print(f"queue: {len(queue)} candidates")

    scenes = pd.read_parquet(SCENES_PATH).set_index("scene_id")
    WIDE_DIR.mkdir(parents=True, exist_ok=True)

    jobs = []
    for c in queue:
        sid = c.get("scene_id")
        if not sid or sid not in scenes.index:
            continue
        sc = scenes.loc[sid]
        if isinstance(sc, pd.DataFrame):
            sc = sc.iloc[0]
        b_paths = [sc["b04_s3"], sc["b03_s3"], sc["b02_s3"]]
        epsg = _utm_epsg(c["mgrs_tile"])
        out = WIDE_DIR / f"{c['tile_id']}.png"
        jobs.append((c["tile_id"], b_paths, c["lat"], c["lon"], epsg, out))

    print(f"rendering {len(jobs)} wide chips with {WORKERS} workers")
    failures = []
    done = 0
    with rasterio.Env(
        AWS_NO_SIGN_REQUEST="YES",
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.TIF",
        GDAL_HTTP_TIMEOUT="20",
        GDAL_HTTP_CONNECTTIMEOUT="8",
        GDAL_HTTP_LOW_SPEED_TIME="20",
        GDAL_HTTP_LOW_SPEED_LIMIT="1024",
        GDAL_HTTP_MAX_RETRY="2",
        VSI_CACHE="TRUE",
    ):
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futs = {ex.submit(render_chip, paths, lat, lon, epsg, out): tid
                    for tid, paths, lat, lon, epsg, out in jobs}
            for f in as_completed(futs):
                tid = futs[f]
                ok, err = f.result()
                done += 1
                if not ok:
                    failures.append((tid, err))
                if done % 50 == 0 or done == len(jobs):
                    print(f"  {done}/{len(jobs)} (failures: {len(failures)})", flush=True)

    if failures:
        print(f"first 5 failures of {len(failures)}:")
        for tid, err in failures[:5]:
            print(f"  {tid}: {err}")
    print(f"done → {WIDE_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
