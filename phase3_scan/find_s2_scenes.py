"""STAC-search Sentinel-2 L2A COGs for each MGRS tile in the CONUS grid.

For every distinct MGRS tile in `phase3_grid.parquet`, query the Element84
STAC catalog for 2025-06-01..2025-08-31 acquisitions and keep the N least
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
DATE_RANGE = "2025-06-01/2025-08-31"
SCENES_PER_TILE = 8
N_WORKERS = 12
FALLBACK_BBOX_HALF = 0.05  # ~5km half-width on lon/lat


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


def _query_one(client: Client, mgrs_tile: str) -> list[dict]:
    zone, band, sq = _split_mgrs(mgrs_tile)
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
    items = list(search.items())
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
    search = client.search(
        collections=[COLLECTION],
        datetime=DATE_RANGE,
        bbox=bbox,
        max_items=300,
    )
    items = list(search.items())
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


def main() -> None:
    grid = pd.read_parquet(GRID_PATH)
    mgrs_tiles = sorted(grid.mgrs_tile.unique().tolist())
    print(f"[scenes] primary pass over {len(mgrs_tiles)} MGRS tiles "
          f"({DATE_RANGE}, top {SCENES_PER_TILE} by lowest cloud_cover)")

    client = Client.open(STAC_URL)
    primary_rows, empty, failed = _primary_pass(client, mgrs_tiles)
    print(f"[scenes] primary done: {len(primary_rows)} rows, "
          f"{len(empty)} empty, {len(failed)} failed")

    fallback_rows: list[dict] = []
    remap: dict[str, str] = {}
    still_empty: list[str] = []
    if empty:
        print(f"[scenes] fallback bbox pass over {len(empty)} empty tiles")
        fallback_rows, remap, still_empty = _fallback_pass(client, grid, empty)
        print(f"[scenes] fallback done: {len(fallback_rows)} rows, "
              f"{len(remap)} remapped, {len(still_empty)} still empty")

    df = pd.DataFrame(primary_rows + fallback_rows)
    df = df.drop_duplicates(subset=["mgrs_tile", "scene_id"]).reset_index(drop=True)
    DATA_US.mkdir(parents=True, exist_ok=True)
    df.to_parquet(SCENES_PATH, index=False)

    if remap:
        grid["mgrs_tile"] = grid["mgrs_tile"].replace(remap)
        grid.to_parquet(GRID_PATH, index=False)
        print(f"[scenes] patched {len(remap)} tile names in {GRID_PATH}")
        sample = list(remap.items())[:5]
        for w, r in sample:
            print(f"[scenes]   {w} → {r}")

    print(f"\n[scenes] wrote {len(df):,} scene rows for "
          f"{df.mgrs_tile.nunique()} MGRS tiles → {SCENES_PATH}")
    print(f"[scenes] tiles with no scene after fallback: {len(still_empty)}")
    if still_empty[:10]:
        print(f"[scenes]   sample: {still_empty[:10]}")
    if failed[:5]:
        for t, e in failed[:5]:
            print(f"[scenes]   primary failed: {t}: {e}")
    print(f"[scenes] median scenes/tile: "
          f"{int(df.groupby('mgrs_tile').size().median())}")
    print(f"[scenes] cloud_cover quantiles: "
          f"p25={df.cloud_cover.quantile(0.25):.1f} "
          f"p50={df.cloud_cover.quantile(0.50):.1f} "
          f"p75={df.cloud_cover.quantile(0.75):.1f}")


if __name__ == "__main__":
    main()
