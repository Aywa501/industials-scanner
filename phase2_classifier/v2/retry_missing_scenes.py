"""Retry STAC search for (mgrs_tile, year) groups missing from v2_scenes_index.

When `v2_build_scenes_index.py` runs locally, Element84 throttles the IP and a
fraction of groups fail with 403 Forbidden. This script identifies the gap by
comparing manifest groups vs scenes-index groups, then re-queries those groups
with lower concurrency (2 workers) and longer backoff to dodge the rate limit.

Successful retries are appended to v2_scenes_index.parquet (idempotent: dedups
on (mgrs_tile, year, scene_id)). Groups that still fail or that genuinely have
no scenes (Element84 returns []) are reported at the end.

Run locally:
    cd sites_us
    python -m phase2_classifier.v2.retry_missing_scenes
"""

from __future__ import annotations

import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import mgrs as mgrs_lib
import numpy as np
import pandas as pd
from pystac_client import Client

from phase2_classifier.v2.v2_build_scenes_index import (
    BAND_ASSETS, COLLECTION, SCENES_PER_GROUP, STAC_URL, SUMMER_RANGE,
    _https_to_s3, _scene_row, _split_mgrs,
)

ROOT = Path(__file__).resolve().parents[3]
DATA_US = ROOT / "data_us"
MANIFEST = DATA_US / "phase2" / "v2_dataset_manifest.parquet"
SCENES_PATH = DATA_US / "phase2" / "v2_scenes_index.parquet"

# Conservative settings vs. the main script (which uses 8 workers + 5 retries
# at 1-16s backoff). Throttled local IPs need even gentler patterns.
N_WORKERS = 2
MAX_ATTEMPTS = 8
BASE_BACKOFF_S = 4.0   # 4, 8, 16, 32, 64, 90, 90, 90 (capped)


def _query_one(client: Client, mgrs_tile: str, year: int) -> list[dict]:
    zone, band, sq = _split_mgrs(mgrs_tile)
    last_err: Exception | None = None
    for attempt in range(MAX_ATTEMPTS):
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
            sleep_s = min(90.0, BASE_BACKOFF_S * (2 ** attempt)) + random.uniform(0, 2)
            time.sleep(sleep_s)
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
    print(f"[retry-scenes] reading manifest {MANIFEST}")
    manifest = pd.read_parquet(MANIFEST)
    manifest = manifest.assign(mgrs_tile=_compute_mgrs(
        manifest["lat"].to_numpy(), manifest["lon"].to_numpy())
    ).dropna(subset=["mgrs_tile"])

    all_groups = (
        manifest[["mgrs_tile", "target_year"]]
        .drop_duplicates()
        .assign(target_year=lambda d: d["target_year"].astype(int))
    )
    print(f"[retry-scenes] manifest groups: {len(all_groups)}")

    print(f"[retry-scenes] reading existing scenes index {SCENES_PATH}")
    scenes = pd.read_parquet(SCENES_PATH)
    covered = set(zip(scenes["mgrs_tile"], scenes["year"].astype(int)))
    print(f"[retry-scenes] covered groups: {len(covered)}")

    missing = [
        (r.mgrs_tile, int(r.target_year))
        for r in all_groups.itertuples(index=False)
        if (r.mgrs_tile, int(r.target_year)) not in covered
    ]
    print(f"[retry-scenes] missing groups to retry: {len(missing)}")
    if not missing:
        print("[retry-scenes] nothing to do")
        return

    client = Client.open(STAC_URL)
    new_rows: list[dict] = []
    still_failed: list[tuple[str, int, str]] = []
    still_empty: list[tuple[str, int]] = []

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(_query_one, client, mt, yr): (mt, yr) for mt, yr in missing}
        for i, fut in enumerate(as_completed(futs), 1):
            key = futs[fut]
            try:
                got = fut.result()
            except Exception as e:
                still_failed.append((*key, repr(e)))
                continue
            if got:
                new_rows.extend(got)
            else:
                still_empty.append(key)
            if i % 25 == 0 or i == len(missing):
                el = time.time() - t0
                print(f"[retry-scenes]   {i}/{len(missing)} done in {el:.0f}s "
                      f"(new_rows={len(new_rows)} still_empty={len(still_empty)} "
                      f"still_failed={len(still_failed)})")

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        merged = pd.concat([scenes, new_df], ignore_index=True).drop_duplicates(
            subset=["mgrs_tile", "year", "scene_id"]
        )
        merged.to_parquet(SCENES_PATH, index=False)
        print()
        print(f"[retry-scenes] appended {len(new_df)} new scene rows")
        print(f"[retry-scenes] scenes_index now: {len(merged):,} rows / "
              f"{merged.groupby(['mgrs_tile','year']).ngroups} groups")

    print()
    print(f"[retry-scenes] still empty (no STAC items): {len(still_empty)}")
    print(f"[retry-scenes] still failed (after backoff): {len(still_failed)}")
    if still_failed:
        for mt, yr, err in still_failed[:5]:
            print(f"  fail: ({mt}, {yr}) {err}")


if __name__ == "__main__":
    main()
