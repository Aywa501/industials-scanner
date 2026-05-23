"""STAC-search Sentinel-2 L2A COGs for each MGRS tile in the CONUS grid.

For every distinct MGRS tile in `phase3_grid.parquet`, query the Element84
STAC catalog for 2025-04-01..2025-10-31 acquisitions and keep the N least
cloudy. We deliberately do NOT pre-filter on scene cloud cover — the worker
medians pixel-wise across scenes and uses per-pixel SCL masks to drop cloudy
pixels, so a 60%-cloud scene still contributes its 40% clear pixels.

Two passes:
  1. Primary pass: query by mgrs:utm_zone + latitude_band + grid_square.
     This catches every MGRS name that S2 actually publishes under.
  2. Fallback pass: for tiles that came back empty, the `mgrs` python lib
     returned a name S2 doesn't use (UTM zone-boundary edge cases). We do a
     bbox query at the empty tile's centroid, learn the *real* tile name from
     the returned items' `mgrs:utm_zone/...` properties, add scenes under the
     real name, and remap the grid in place so downstream code is consistent.

Run locally once after make_grid:
    cd sites_us
    python -m phase3_scan.find_s2_scenes

Output:
  data_us/phase3_scenes.parquet
    cols: mgrs_tile, scene_id, datetime, cloud_cover,
          b04_s3, b03_s3, b02_s3, scl_s3
  data_us/phase3_grid.parquet  (mgrs_tile column patched in place)
"""

from __future__ import annotations

import os
import random
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
from dotenv import load_dotenv
from pystac_client import Client

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
DATA_US = ROOT.parent / "data_us"
GRID_PATH = DATA_US / "phase3_grid.parquet"
SCENES_PATH = DATA_US / "phase3_scenes.parquet"

STAC_URL = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-l2a"
DATE_RANGE = "2025-04-01/2025-10-31"
SCENES_PER_TILE = 16
N_WORKERS = int(os.environ.get("SCENES_WORKERS", "12"))
FALLBACK_BBOX_HALF = 0.05  # ~5km half-width on lon/lat
MAX_RETRIES = 5            # Element84 rate-limits with 403s; back off and retry


def _https_to_s3(href: str) -> str:
    p = urlparse(href)
    bucket = p.netloc.split(".")[0]
    return f"s3://{bucket}{p.path}"


def _split_mgrs(t: str) -> tuple[int, str, str]:
    return int(t[:-3]), t[-3], t[-2:]


def _true_mgrs_name(item) -> str | None:
    p = item.properties
    z = p.get("mgrs:utm_zone")
    b = p.get("mgrs:latitude_band")
    s = p.get("mgrs:grid_square")
    if z is None or not b or not s:
        return None
    return f"{int(z)}{b}{s}"


def _scene_row(mgrs_tile: str, item) -> dict | None:
    try:
        return {
            "mgrs_tile": mgrs_tile,
            "scene_id": item.id,
            "datetime": item.properties.get("datetime"),
            "cloud_cover": item.properties.get("eo:cloud_cover"),
            "b04_s3": _https_to_s3(item.assets["red"].href),
            "b03_s3": _https_to_s3(item.assets["green"].href),
            "b02_s3": _https_to_s3(item.assets["blue"].href),
            "scl_s3": _https_to_s3(item.assets["scl"].href),
        }
    except KeyError:
        return None


def _retry(fn, label: str):
    """Run fn() with exponential backoff. Element84 answers rate-limited
    requests with 403s; a short backoff clears transient throttling. A hard
    block still raises after MAX_RETRIES — the caller records it as failed and
    a later resume run picks the tile back up."""
    last = None
    for attempt in range(MAX_RETRIES):
        try:
            return fn()
        except Exception as e:  # STAC client raises assorted exception types
            last = e
            if attempt < MAX_RETRIES - 1:
                time.sleep(min(2 ** attempt, 30) + random.random())
    raise last


def _query_one(client: Client, mgrs_tile: str) -> list[dict]:
    zone, band, sq = _split_mgrs(mgrs_tile)

    def _do():
        search = client.search(
            collections=[COLLECTION],
            datetime=DATE_RANGE,
            query={
                "mgrs:utm_zone": {"eq": zone},
                "mgrs:latitude_band": {"eq": band},
                "mgrs:grid_square": {"eq": sq},
            },
            max_items=200,
        )
        return list(search.items())

    items = _retry(_do, mgrs_tile)
    items.sort(key=lambda it: it.properties.get("eo:cloud_cover", 100))
    out = []
    for it in items[:SCENES_PER_TILE]:
        row = _scene_row(mgrs_tile, it)
        if row is not None:
            out.append(row)
    return out


def _fallback_bbox(client: Client, lon: float, lat: float
                   ) -> tuple[str | None, list[dict]]:
    """Bbox-search around (lon, lat) and return (dominant_real_tile, scene_rows)."""
    bbox = [lon - FALLBACK_BBOX_HALF, lat - FALLBACK_BBOX_HALF,
            lon + FALLBACK_BBOX_HALF, lat + FALLBACK_BBOX_HALF]
    items = _retry(
        lambda: list(client.search(
            collections=[COLLECTION], datetime=DATE_RANGE, bbox=bbox,
            max_items=300,
        ).items()),
        f"fallback@{lon:.2f},{lat:.2f}",
    )
    by_tile: dict[str, list] = defaultdict(list)
    for it in items:
        name = _true_mgrs_name(it)
        if name:
            by_tile[name].append(it)
    if not by_tile:
        return None, []
    dominant = max(by_tile.items(), key=lambda kv: len(kv[1]))[0]
    chosen = sorted(
        by_tile[dominant],
        key=lambda it: it.properties.get("eo:cloud_cover", 100),
    )[:SCENES_PER_TILE]
    rows = [r for r in (_scene_row(dominant, it) for it in chosen) if r is not None]
    return dominant, rows


def _primary_pass(client: Client, mgrs_tiles: list[str]
                  ) -> tuple[list[dict], list[str], list[tuple[str, str]]]:
    rows: list[dict] = []
    empty: list[str] = []
    failed: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(_query_one, client, t): t for t in mgrs_tiles}
        for i, fut in enumerate(as_completed(futs), 1):
            t = futs[fut]
            try:
                got = fut.result()
            except Exception as e:
                failed.append((t, repr(e)))
                continue
            if got:
                rows.extend(got)
            else:
                empty.append(t)
            if i % 100 == 0:
                print(f"[scenes]   primary {i}/{len(mgrs_tiles)} done "
                      f"(rows={len(rows)}, empty={len(empty)}, "
                      f"failed={len(failed)})")
    return rows, empty, failed


def _fallback_pass(client: Client, grid: pd.DataFrame, empty_tiles: list[str]
                   ) -> tuple[list[dict], dict[str, str], list[str]]:
    centroids = (
        grid[grid.mgrs_tile.isin(empty_tiles)]
        .groupby("mgrs_tile")[["lon", "lat"]]
        .median()
    )
    rows: list[dict] = []
    remap: dict[str, str] = {}
    still_empty: list[str] = []

    def _job(wrong_tile: str):
        c = centroids.loc[wrong_tile]
        return wrong_tile, _fallback_bbox(client, float(c.lon), float(c.lat))

    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(_job, t): t for t in empty_tiles}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                wrong_tile, (true_tile, scene_rows) = fut.result()
            except Exception as e:
                wrong_tile = futs[fut]
                still_empty.append(wrong_tile)
                print(f"[scenes]   fallback {wrong_tile} errored: {e!r}")
                continue
            if true_tile is None or not scene_rows:
                still_empty.append(wrong_tile)
                continue
            remap[wrong_tile] = true_tile
            rows.extend(scene_rows)
            if i % 25 == 0:
                print(f"[scenes]   fallback {i}/{len(empty_tiles)} done "
                      f"(remapped={len(remap)}, still_empty={len(still_empty)})")
    return rows, remap, still_empty


def _write_index(*frames: pd.DataFrame) -> pd.DataFrame:
    """Merge frames, dedup, and atomically overwrite SCENES_PATH (write-temp +
    rename) so a kill mid-write can never corrupt the on-disk index."""
    parts = [f for f in frames if f is not None and not f.empty]
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if not df.empty:
        df = df.drop_duplicates(subset=["mgrs_tile", "scene_id"]).reset_index(drop=True)
    DATA_US.mkdir(parents=True, exist_ok=True)
    tmp = SCENES_PATH.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(SCENES_PATH)
    return df


def main() -> None:
    grid = pd.read_parquet(GRID_PATH)
    all_tiles = sorted(grid.mgrs_tile.unique().tolist())

    # Resume: a rate-limited run still writes whatever it got, so re-running
    # picks up only the tiles still missing — no lost progress, no redo.
    existing = pd.DataFrame()
    if SCENES_PATH.exists():
        existing = pd.read_parquet(SCENES_PATH)
        done = set(existing.mgrs_tile)
        todo = [t for t in all_tiles if t not in done]
        print(f"[scenes] resume: {len(done)}/{len(all_tiles)} tiles already in "
              f"{SCENES_PATH.name}; {len(todo)} still to query")
    else:
        todo = all_tiles
    if not todo:
        print("[scenes] index already complete — nothing to query")
        return

    print(f"[scenes] primary pass over {len(todo)} MGRS tiles "
          f"({DATE_RANGE}, top {SCENES_PER_TILE} by lowest cloud_cover, "
          f"{N_WORKERS} workers)")
    client = Client.open(STAC_URL)
    primary_rows, empty, failed = _primary_pass(client, todo)
    print(f"[scenes] primary done: {len(primary_rows)} rows, "
          f"{len(empty)} empty, {len(failed)} failed")

    # Checkpoint after the primary pass so a rate-limit during fallback (or a
    # kill) cannot lose the primary results.
    df = _write_index(existing, pd.DataFrame(primary_rows))
    print(f"[scenes] checkpoint: {df.mgrs_tile.nunique()} tiles on disk")

    remap: dict[str, str] = {}
    if empty:
        print(f"[scenes] fallback bbox pass over {len(empty)} empty tiles")
        fallback_rows, remap, still_empty = _fallback_pass(client, grid, empty)
        print(f"[scenes] fallback done: {len(fallback_rows)} rows, "
              f"{len(remap)} remapped, {len(still_empty)} still empty")
        df = _write_index(df, pd.DataFrame(fallback_rows))

    if remap:
        grid["mgrs_tile"] = grid["mgrs_tile"].replace(remap)
        grid.to_parquet(GRID_PATH, index=False)
        print(f"[scenes] patched {len(remap)} tile names in {GRID_PATH}")

    covered = df.mgrs_tile.nunique()
    missing = len(all_tiles) - covered
    print(f"\n[scenes] {len(df):,} scene rows for {covered}/{len(all_tiles)} "
          f"MGRS tiles → {SCENES_PATH}")
    if missing:
        print(f"[scenes] {missing} tiles still missing (likely rate-limited). "
              f"Wait for the limit to cool down and re-run — it resumes from here.")
        if failed[:5]:
            for t, e in failed[:5]:
                print(f"[scenes]   failed: {t}: {e}")
    else:
        print(f"[scenes] complete. median scenes/tile: "
              f"{int(df.groupby('mgrs_tile').size().median())} | "
              f"cloud_cover p50={df.cloud_cover.quantile(0.50):.2f}")


if __name__ == "__main__":
    main()
