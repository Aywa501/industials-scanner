"""Prep offline-labeling queue for v2 phase3 candidates.

1. Read all per-shard result parquets from /tmp/v2_results (or RESULTS_DIR env).
2. Stratified sample N_TOTAL candidates across prob buckets.
3. For each candidate, read a 256-px (2.56km) S2 chip from the lowest-cloud
   scene for its MGRS shard, render B04/B03/B02 to PNG with 1-99 stretch.
4. Write queue.json describing each candidate.

Outputs under sites_us/.artifacts/labeling_v2/:
    chips/{tile_id}.png
    queue.json
"""

from __future__ import annotations

import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyproj
import rasterio
from PIL import Image
from rasterio.session import AWSSession
from rasterio.windows import from_bounds

os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")

ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = ROOT / ".artifacts" / "labeling_v2"
CHIPS_DIR = ARTIFACTS / "chips"
QUEUE_PATH = ARTIFACTS / "queue.json"

DATA_US = ROOT.parent / "data_us"
SCENES_PATH = DATA_US / "phase3_scan" / "phase3_scenes.parquet"

RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "/tmp/v2_results"))

# Stratified buckets: (lo, hi, target). Total = 1500.
BUCKETS = [
    (0.00, 0.50, 100),
    (0.50, 0.70, 250),
    (0.70, 0.80, 250),
    (0.80, 0.90, 300),
    (0.90, 0.95, 250),
    (0.95, 0.99, 200),
    (0.99, 1.01, 150),
]
N_TOTAL = sum(b[2] for b in BUCKETS)
SEED = 42

IMG_NATIVE = 256
GSD_M = 10.0
HALF_M = (IMG_NATIVE / 2) * GSD_M  # 1280 m
RENDER_PX = 512
WORKERS = 8


def _utm_epsg(mgrs_tile: str) -> int:
    zone = int(mgrs_tile[:-3])
    return 32600 + zone


def load_candidates() -> pd.DataFrame:
    files = sorted(p for p in RESULTS_DIR.glob("*.parquet") if not p.stem.endswith("_emb"))
    if not files:
        sys.exit(f"no parquets under {RESULTS_DIR}")
    frames = []
    for p in files:
        df = pd.read_parquet(p)
        if df.empty:
            continue
        df["mgrs_tile"] = p.stem
        frames.append(df)
    cand = pd.concat(frames, ignore_index=True)
    cand["prob"] = pd.to_numeric(cand["prob"], errors="coerce")
    cand = cand.dropna(subset=["prob", "lat", "lon"]).reset_index(drop=True)
    return cand


def stratified_sample(cand: pd.DataFrame, rng: random.Random) -> pd.DataFrame:
    picks = []
    for lo, hi, n in BUCKETS:
        pool = cand[(cand.prob >= lo) & (cand.prob < hi)]
        if pool.empty:
            print(f"  [{lo:.2f}-{hi:.2f}): 0 available, skipping {n}")
            continue
        take = min(n, len(pool))
        idxs = rng.sample(pool.index.tolist(), take)
        picks.append(pool.loc[idxs])
        print(f"  [{lo:.2f}-{hi:.2f}): picked {take}/{n} from pool of {len(pool):,}")
    out = pd.concat(picks).sort_values("prob", ascending=False).reset_index(drop=True)
    return out


def pick_scene_per_mgrs(scenes: pd.DataFrame, mgrs_tiles: list[str]) -> dict[str, dict]:
    """Lowest-cloud scene per MGRS shard."""
    sub = scenes[scenes.mgrs_tile.isin(mgrs_tiles)].copy()
    best = (sub.sort_values(["mgrs_tile", "cloud_cover"])
              .drop_duplicates("mgrs_tile", keep="first"))
    return best.set_index("mgrs_tile").to_dict("index")


def render_chip(b_paths: list[str], lat: float, lon: float, epsg: int, out_path: Path) -> tuple[bool, str | None]:
    if out_path.exists():
        return True, None
    try:
        to_utm = pyproj.Transformer.from_crs(4326, epsg, always_xy=True).transform
        x, y = to_utm(lon, lat)
        xmin, ymin, xmax, ymax = x - HALF_M, y - HALF_M, x + HALF_M, y + HALF_M
        bands = []
        for path in b_paths:
            with rasterio.open(path) as src:
                win = from_bounds(xmin, ymin, xmax, ymax, transform=src.transform)
                arr = src.read(1, window=win, out_shape=(IMG_NATIVE, IMG_NATIVE),
                               boundless=True, fill_value=0)
            bands.append(arr.astype(np.float32))
        rgb = np.stack(bands, axis=0)  # (3, H, W) order: R=B04, G=B03, B=B02
        lo, hi = np.percentile(rgb, (1, 99))
        if hi <= lo:
            hi = lo + 1
        arr = np.clip((rgb - lo) / (hi - lo), 0, 1) * 255
        arr = arr.astype(np.uint8).transpose(1, 2, 0)  # CHW->HWC
        img = Image.fromarray(arr).resize((RENDER_PX, RENDER_PX), Image.LANCZOS)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path, "PNG", optimize=False)
        return True, None
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def main() -> int:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    CHIPS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"loading candidates from {RESULTS_DIR}")
    cand = load_candidates()
    print(f"  total: {len(cand):,} candidates from {cand.mgrs_tile.nunique()} shards")
    print(f"  prob: min={cand.prob.min():.3f} max={cand.prob.max():.3f} median={cand.prob.median():.3f}")

    rng = random.Random(SEED)
    print(f"stratified sample target={N_TOTAL}:")
    sample = stratified_sample(cand, rng)
    print(f"  sampled {len(sample):,} candidates ({sample.mgrs_tile.nunique()} shards)")

    print(f"loading scenes from {SCENES_PATH}")
    scenes = pd.read_parquet(SCENES_PATH)
    needed = sample.mgrs_tile.unique().tolist()
    scene_by_mgrs = pick_scene_per_mgrs(scenes, needed)
    missing = set(needed) - set(scene_by_mgrs)
    if missing:
        print(f"  WARN: {len(missing)} MGRS have no scene; their candidates will be skipped")
        sample = sample[~sample.mgrs_tile.isin(missing)]

    jobs = []
    for r in sample.itertuples(index=False):
        sc = scene_by_mgrs[r.mgrs_tile]
        b_paths = [sc["b04_s3"], sc["b03_s3"], sc["b02_s3"]]
        epsg = _utm_epsg(r.mgrs_tile)
        out = CHIPS_DIR / f"{r.tile_id}.png"
        jobs.append((r.tile_id, b_paths, float(r.lat), float(r.lon), epsg, out))

    print(f"rendering {len(jobs)} chips with {WORKERS} workers...")
    failures = []
    done = 0
    # Anonymous S3 — env var set at module-load; rasterio reads from env.
    with rasterio.Env(
        AWS_NO_SIGN_REQUEST="YES",
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.TIF",
        GDAL_HTTP_TIMEOUT="20",
        GDAL_HTTP_CONNECTTIMEOUT="8",
        GDAL_HTTP_LOW_SPEED_TIME="20",
        GDAL_HTTP_LOW_SPEED_LIMIT="1024",
        GDAL_HTTP_MAX_RETRY="2",
        GDAL_HTTP_RETRY_DELAY="2",
        VSI_CACHE="TRUE",
        CPL_VSIL_CURL_NON_CACHED="",
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

    # Build queue.json — keep only candidates whose chip rendered.
    queue = []
    for r in sample.itertuples(index=False):
        chip = CHIPS_DIR / f"{r.tile_id}.png"
        if not chip.exists():
            continue
        sc = scene_by_mgrs[r.mgrs_tile]
        queue.append({
            "tile_id": r.tile_id,
            "mgrs_tile": r.mgrs_tile,
            "lat": float(r.lat),
            "lon": float(r.lon),
            "prob": float(r.prob),
            "scene_id": sc.get("scene_id"),
            "scene_date": str(sc.get("datetime"))[:10] if sc.get("datetime") else None,
        })
    # Sort by prob desc — labelers usually want to start with high-conf
    queue.sort(key=lambda x: -x["prob"])

    QUEUE_PATH.write_text(json.dumps(queue, indent=2))
    print(f"wrote queue: {len(queue)} candidates → {QUEUE_PATH}")
    print(f"chips → {CHIPS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
