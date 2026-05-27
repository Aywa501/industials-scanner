"""Re-render chips that came out empty (std=0) by trying alternate scenes.

For each chip whose chip_std==0, walk through next-lowest-cloud scenes for its
MGRS and re-render until a non-empty chip is produced (std >= 5). Update
queue.json with the new scene_id/date.
"""
from __future__ import annotations

import json
import os
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
CHIPS_DIR = ARTIFACTS / "chips"
WIDE_DIR = ARTIFACTS / "chips_wide"

DATA_US = ROOT.parent / "data_us"
SCENES_PATH = DATA_US / "phase3_scan" / "phase3_scenes.parquet"

IMG_NATIVE = 256
WIDE_NATIVE = 512
GSD_M = 10.0
RENDER_PX = 512
MIN_STD = 5.0
WORKERS = 8


def _utm_epsg(mgrs_tile: str) -> int:
    return 32600 + int(mgrs_tile[:-3])


def render_one(b_paths, lat, lon, epsg, native, out_path):
    to_utm = pyproj.Transformer.from_crs(4326, epsg, always_xy=True).transform
    x, y = to_utm(lon, lat)
    half = (native / 2) * GSD_M
    xmin, ymin, xmax, ymax = x - half, y - half, x + half, y + half
    bands = []
    for p in b_paths:
        with rasterio.open(p) as src:
            arr = src.read(1,
                           window=from_bounds(xmin, ymin, xmax, ymax, transform=src.transform),
                           out_shape=(native, native), boundless=True, fill_value=0)
        bands.append(arr.astype(np.float32))
    rgb = np.stack(bands, axis=0)
    if rgb.std() < 0.5:
        return None  # truly empty — let caller try another scene
    lo, hi = np.percentile(rgb, (1, 99))
    if hi <= lo:
        hi = lo + 1
    arr = np.clip((rgb - lo) / (hi - lo), 0, 1) * 255
    arr = arr.astype(np.uint8).transpose(1, 2, 0)
    img = Image.fromarray(arr).resize((RENDER_PX, RENDER_PX), Image.LANCZOS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, "PNG", optimize=False)
    return float(arr.std())


def try_refetch(cand, scenes_by_mgrs):
    mgrs = cand["mgrs_tile"]
    scenes = scenes_by_mgrs.get(mgrs)
    if scenes is None:
        return cand["tile_id"], None
    epsg = _utm_epsg(mgrs)
    for sc in scenes:
        if sc["scene_id"] == cand.get("scene_id"):
            continue
        b_paths = [sc["b04_s3"], sc["b03_s3"], sc["b02_s3"]]
        out = CHIPS_DIR / f"{cand['tile_id']}.png"
        try:
            s = render_one(b_paths, cand["lat"], cand["lon"], epsg, IMG_NATIVE, out)
        except Exception:
            continue
        if s is None or s < MIN_STD:
            continue
        # also render wide version
        wide_out = WIDE_DIR / f"{cand['tile_id']}.png"
        try:
            render_one(b_paths, cand["lat"], cand["lon"], epsg, WIDE_NATIVE, wide_out)
        except Exception:
            pass
        return cand["tile_id"], {"scene_id": sc["scene_id"],
                                  "scene_date": str(sc["datetime"])[:10],
                                  "chip_std": round(s, 1)}
    return cand["tile_id"], None


def main() -> int:
    queue = json.loads(QUEUE_PATH.read_text())
    bad = [c for c in queue if c.get("chip_std", 99) == 0.0]
    print(f"refetching {len(bad)} empty chips")

    scenes = pd.read_parquet(SCENES_PATH).sort_values(["mgrs_tile", "cloud_cover"])
    scenes_by_mgrs: dict[str, list[dict]] = {}
    for r in scenes.itertuples(index=False):
        scenes_by_mgrs.setdefault(r.mgrs_tile, []).append({
            "scene_id": r.scene_id, "datetime": r.datetime,
            "b04_s3": r.b04_s3, "b03_s3": r.b03_s3, "b02_s3": r.b02_s3,
        })

    updates = {}
    with rasterio.Env(
        AWS_NO_SIGN_REQUEST="YES",
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.TIF",
        GDAL_HTTP_TIMEOUT="20",
        GDAL_HTTP_CONNECTTIMEOUT="8",
        GDAL_HTTP_MAX_RETRY="2",
        VSI_CACHE="TRUE",
    ):
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futs = {ex.submit(try_refetch, c, scenes_by_mgrs): c["tile_id"] for c in bad}
            done = 0
            for f in as_completed(futs):
                tid, upd = f.result()
                if upd:
                    updates[tid] = upd
                done += 1
                if done % 25 == 0 or done == len(bad):
                    print(f"  {done}/{len(bad)} (recovered: {len(updates)})", flush=True)

    for c in queue:
        if c["tile_id"] in updates:
            c.update(updates[c["tile_id"]])
            c["chip_quality"] = "ok"
    QUEUE_PATH.write_text(json.dumps(queue, indent=2))
    print(f"recovered {len(updates)}/{len(bad)} chips, queue updated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
