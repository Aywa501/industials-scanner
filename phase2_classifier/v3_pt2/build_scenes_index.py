"""Build Landsat scenes index for Stage 2b — one STAC query per (1-deg cell, year).

Spatially bucket candidates into a 1-degree grid (~600 cells over CONUS). For
each (grid_cell, year) pair, query landsatlook STAC for matching pan scenes
(Jun-Aug, <40% cloud). Output reused by change_scan.py without any in-worker
STAC calls — keeps the EC2 IP off the rate-limit blacklist.

Output: data_us/phase2/v3/landsat_scenes_index.parquet
  columns: grid_lat, grid_lon, year, scene_id, platform, s3_href, cloud_cover, datetime
"""
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from pystac_client import Client

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CANDS = os.path.join(ROOT, "..", "data_us/phase2/v3/stage2_candidates.parquet")
OUT = os.path.join(ROOT, "..", "data_us/phase2/v3/landsat_scenes_index.parquet")

STAC_URL = "https://landsatlook.usgs.gov/stac-server"
GRID_DEG = 1.0
MAX_SCENES = 24
YEARS = {2008: "LANDSAT_7", 2022: "LANDSAT_8"}
STAC_WORKERS = 4   # residential IP — keep low


def _s3_href(asset):
    alt = asset.extra_fields.get("alternate", {}).get("s3", {})
    return alt.get("href") or asset.href


def query_cell(cell, year, platform, retries=6):
    lat_idx, lon_idx = cell
    bbox = [
        lon_idx * GRID_DEG, lat_idx * GRID_DEG,
        (lon_idx + 1) * GRID_DEG, (lat_idx + 1) * GRID_DEG,
    ]
    last_err = None
    for attempt in range(retries):
        try:
            items = list(Client.open(STAC_URL).search(
                collections=["landsat-c2l1"],
                bbox=bbox,
                datetime=f"{year}-06-01/{year}-08-31",
                query={"platform": {"eq": platform}, "eo:cloud_cover": {"lt": 40}},
                max_items=MAX_SCENES,
            ).items())
            rows = []
            for it in items:
                pan = it.assets.get("pan")
                if not pan:
                    continue
                rows.append({
                    "grid_lat": lat_idx, "grid_lon": lon_idx, "year": year,
                    "scene_id": it.id, "platform": platform,
                    "s3_href": _s3_href(pan),
                    "cloud_cover": float(it.properties.get("eo:cloud_cover", -1)),
                    "datetime": it.properties.get("datetime"),
                })
            return rows
        except Exception as e:
            last_err = e
            time.sleep(min(60, 2 ** attempt + 0.5 * attempt))
    print(f"  GIVE UP cell={cell} year={year}: {last_err}")
    return []


def main():
    df = pd.read_parquet(CANDS)
    lat_idx = np.floor(df["lat"].values / GRID_DEG).astype(int)
    lon_idx = np.floor(df["lon"].values / GRID_DEG).astype(int)
    unique_cells = sorted({(int(la), int(lo)) for la, lo in zip(lat_idx, lon_idx)})
    print(f"candidates: {len(df):,}   unique {GRID_DEG}-deg cells: {len(unique_cells):,}")
    print(f"STAC queries to make: {len(unique_cells) * len(YEARS):,}")

    all_rows = []
    n_queries = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=STAC_WORKERS) as pool:
        futs = {}
        for cell in unique_cells:
            for year, platform in YEARS.items():
                futs[pool.submit(query_cell, cell, year, platform)] = (cell, year)
        for fut in as_completed(futs):
            rows = fut.result()
            all_rows.extend(rows)
            n_queries += 1
            if n_queries % 50 == 0 or n_queries == len(futs):
                elapsed = time.time() - t0
                rate = n_queries / max(elapsed, 1)
                eta = (len(futs) - n_queries) / max(rate, 0.01) / 60
                print(f"  STAC {n_queries:,}/{len(futs):,}  scenes={len(all_rows):,}  "
                      f"rate={rate:.1f}/s  eta={eta:.1f}min")

    out = pd.DataFrame(all_rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"\nwrote {OUT}: {len(out):,} scene rows")

    print(f"\nscenes per (cell,year) — describe:")
    print(out.groupby(["grid_lat", "grid_lon", "year"]).size().describe())
    cells_with_zero = (out.groupby(["grid_lat", "grid_lon", "year"]).size() == 0).sum()
    expected = len(unique_cells) * len(YEARS)
    got = out.groupby(["grid_lat", "grid_lon", "year"]).ngroups
    print(f"\ncoverage: {got}/{expected} cell-years with >=1 scene")


if __name__ == "__main__":
    main()
