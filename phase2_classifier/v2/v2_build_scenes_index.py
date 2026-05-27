"""Pre-compute the (mgrs_tile, year) → top-N S2 scenes index for v2_train.

Mirrors `find_s2_scenes.py` but:
  * keyed on (mgrs_tile, target_year) since v2 manifests carry per-row years
  * 6 bands (B02 B03 B04 B8A B11 B12) + SCL — Prithvi needs SWIR/NIR, not just RGB
  * driven by `data_us/phase2/v2_dataset_manifest.parquet` (MGRS computed by mgrs lib
    inside this script — manifest only has lat/lon)

Run locally once:
    cd sites_us
    python -m phase2_classifier.v2.v2_build_scenes_index

Output:
    data_us/phase2/v2_scenes_index.parquet
      cols: mgrs_tile, year, scene_id, datetime, cloud_cover,
            B02_s3, B03_s3, B04_s3, B8A_s3, B11_s3, B12_s3, scl_s3
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

import mgrs as mgrs_lib
import numpy as np
import pandas as pd
from pystac_client import Client

ROOT = Path(__file__).resolve().parents[3]
DATA_US = ROOT / "data_us"
MANIFEST = DATA_US / "phase2" / "v2_dataset_manifest.parquet"
OUT_PATH = DATA_US / "phase2" / "v2_scenes_index.parquet"

STAC_URL = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-l2a"
SUMMER_RANGE = "{y}-05-01/{y}-09-30"
SCENES_PER_GROUP = 8
N_WORKERS = 8  # STAC API rate-limits aggressively above ~10-12 concurrent

BAND_ASSETS = {
    "B02": "blue", "B03": "green", "B04": "red",
    "B8A": "nir08", "B11": "swir16", "B12": "swir22",
}


def _https_to_s3(href: str) -> str:
    p = urlparse(href)
    return f"s3://{p.netloc.split('.')[0]}{p.path}"


def _split_mgrs(t: str) -> tuple[int, str, str]:
    return int(t[:-3]), t[-3], t[-2:]


def _scene_row(mgrs_tile: str, year: int, item) -> dict | None:
    try:
        row = {
            "mgrs_tile": mgrs_tile,
            "year": year,
            "scene_id": item.id,
            "datetime": item.properties.get("datetime"),
            "cloud_cover": item.properties.get("eo:cloud_cover"),
            "scl_s3": _https_to_s3(item.assets["scl"].href),
        }
        for b, asset in BAND_ASSETS.items():
            row[f"{b}_s3"] = _https_to_s3(item.assets[asset].href)
        return row
    except KeyError:
        return None


def _query_one(client: Client, mgrs_tile: str, year: int,
               max_attempts: int = 5) -> list[dict]:
    """STAC search one (mgrs, year) with exponential backoff for 429/5xx/Forbidden."""
    import random
    zone, band, sq = _split_mgrs(mgrs_tile)
    last_err: Exception | None = None
    for attempt in range(max_attempts):
        try:
            search = client.search(
                collections=[COLLECTION],
                datetime=SUMMER_RANGE.format(y=year),
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
            for it in items[:SCENES_PER_GROUP]:
                row = _scene_row(mgrs_tile, year, it)
                if row is not None:
                    out.append(row)
            return out
        except Exception as e:
            last_err = e
            # exponential backoff with jitter, 1s..16s
            time.sleep(min(16.0, 2 ** attempt) + random.uniform(0, 1))
    raise last_err  # type: ignore[misc]


def _compute_mgrs(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    m = mgrs_lib.MGRS()
    out = np.empty(len(lats), dtype=object)
    for i, (la, lo) in enumerate(zip(lats, lons)):
        try:
            out[i] = m.toMGRS(float(la), float(lo), MGRSPrecision=0)[:5]
        except Exception:
            out[i] = None
    return out


def main() -> None:
    print(f"[v2-scenes] reading manifest {MANIFEST}")
    manifest = pd.read_parquet(MANIFEST)
    print(f"[v2-scenes] {len(manifest):,} rows")

    print("[v2-scenes] computing MGRS...")
    manifest = manifest.assign(mgrs_tile=_compute_mgrs(
        manifest["lat"].to_numpy(), manifest["lon"].to_numpy())
    ).dropna(subset=["mgrs_tile"]).reset_index(drop=True)

    groups = manifest[["mgrs_tile", "target_year"]].drop_duplicates().reset_index(drop=True)
    groups["target_year"] = groups["target_year"].astype(int)
    print(f"[v2-scenes] {len(groups)} unique (mgrs_tile, year) groups to query")

    client = Client.open(STAC_URL)
    rows: list[dict] = []
    empty: list[tuple[str, int]] = []
    failed: list[tuple[str, int, str]] = []
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(_query_one, client, r.mgrs_tile, int(r.target_year)): (r.mgrs_tile, int(r.target_year))
                for r in groups.itertuples(index=False)}
        for i, fut in enumerate(as_completed(futs), 1):
            key = futs[fut]
            try:
                got = fut.result()
            except Exception as e:
                failed.append((*key, repr(e)))
                if len(failed) <= 5 or len(failed) % 500 == 0:
                    print(f"[v2-scenes]   FAIL {key}: {e!r}")
                continue
            if got:
                rows.extend(got)
            else:
                empty.append(key)
            if i % 200 == 0:
                print(f"[v2-scenes]   {i}/{len(groups)} done "
                      f"(rows={len(rows)} empty={len(empty)} failed={len(failed)})")

    df = pd.DataFrame(rows)
    df.to_parquet(OUT_PATH, index=False)
    print(f"[v2-scenes] wrote {len(df):,} scene rows for "
          f"{df.groupby(['mgrs_tile','year']).ngroups} groups → {OUT_PATH}")
    print(f"[v2-scenes] empty groups: {len(empty)}, failed: {len(failed)}")


if __name__ == "__main__":
    main()
