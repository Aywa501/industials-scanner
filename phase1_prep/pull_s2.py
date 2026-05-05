"""Phase 1 anchor S2 chip fetcher via ee.data.computePixels (high-volume EE endpoint).

Reads s2_chip_manifest.parquet, finds (site_id, year) pairs not yet COMPLETED,
fetches chips synchronously, uploads to GCS, and updates the manifest at end.
The high-volume endpoint bypasses GEE's batch-export task queue entirely.
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import ee
import pandas as pd
from dotenv import load_dotenv
from google.cloud import storage
from pyproj import Transformer

load_dotenv(Path(__file__).parent.parent / ".env")

GCP_PROJECT = os.getenv("GCP_PROJECT", "")
GCS_BUCKET = os.getenv("GCS_BUCKET", "")
S2_AOI_BUFFER_M = int(os.getenv("S2_AOI_BUFFER_M", "1280"))
S2_TILE_SIZE_PX = int(os.getenv("S2_TILE_SIZE_PX", "256"))
S2_CLOUD_PCT = float(os.getenv("S2_CLOUD_PCT", "20"))
S2_YEAR_START = int(os.getenv("S2_YEAR_START", "2017"))
S2_YEAR_END = int(os.getenv("S2_YEAR_END", "2025"))
S2_YEARS = list(range(S2_YEAR_START, S2_YEAR_END + 1))
S2_BANDS = ["B4", "B3", "B2", "B8"]
S2_SEASON_START = os.getenv("S2_SEASON_START", "06-01")
S2_SEASON_END = os.getenv("S2_SEASON_END", "08-31")

MAX_WORKERS = int(os.getenv("S2_FETCH_WORKERS", "20"))
MANIFEST_URI = f"gs://{GCS_BUCKET}/manifest/s2_chip_manifest.parquet"

MANIFEST_COLUMNS = [
    "site_id", "site_type", "canonical_project_name", "state",
    "lat", "lng", "announcement_date", "year",
    "aoi_buffer_m", "tile_size_px", "tile_uri",
    "export_task_id", "export_status",
    "created_at", "updated_at",
]

HIGHVOL_URL = "https://earthengine-highvolume.googleapis.com"


def init_ee_highvol() -> None:
    sa = os.getenv("GEE_SERVICE_ACCOUNT") or ""
    key = os.getenv("GEE_KEY_FILE") or ""
    if sa and key:
        ee.Initialize(ee.ServiceAccountCredentials(sa, key_file=key),
                      project=GCP_PROJECT, opt_url=HIGHVOL_URL)
    else:
        ee.Initialize(project=GCP_PROJECT, opt_url=HIGHVOL_URL)


def utm_epsg(lat: float, lng: float) -> int:
    zone = int((lng + 180) / 6) + 1
    return 32600 + zone if lat >= 0 else 32700 + zone


def chip_grid(lat: float, lng: float, buffer_m: int, dim: int) -> dict:
    epsg = utm_epsg(lat, lng)
    tr = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    cx, cy = tr.transform(lng, lat)
    pixel = (2 * buffer_m) / dim
    return {
        "dimensions": {"width": dim, "height": dim},
        "affineTransform": {
            "scaleX": pixel,
            "shearX": 0,
            "translateX": cx - buffer_m,
            "shearY": 0,
            "scaleY": -pixel,
            "translateY": cy + buffer_m,
        },
        "crsCode": f"EPSG:{epsg}",
    }


def yearly_composite(aoi: ee.Geometry, year: int) -> ee.Image:
    coll = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(aoi)
            .filterDate(f"{year}-{S2_SEASON_START}", f"{year}-{S2_SEASON_END}")
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", S2_CLOUD_PCT)))
    return coll.median().select(S2_BANDS)


def fetch_one(site_id: str, lat: float, lng: float, year: int,
              gcs: storage.Client) -> str:
    aoi = ee.Geometry.Point([lng, lat]).buffer(S2_AOI_BUFFER_M).bounds()
    img = yearly_composite(aoi, year)
    grid = chip_grid(lat, lng, S2_AOI_BUFFER_M, S2_TILE_SIZE_PX)
    payload = ee.data.computePixels({
        "expression": img,
        "fileFormat": "GEO_TIFF",
        "grid": grid,
    })
    blob_path = f"s2/{site_id}/{year}.tif"
    gcs.bucket(GCS_BUCKET).blob(blob_path).upload_from_string(
        payload, content_type="image/tiff"
    )
    return f"gs://{GCS_BUCKET}/{blob_path}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                        help="cap chips this run (default: all missing)")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS,
                        help=f"parallel workers (default {MAX_WORKERS})")
    parser.add_argument("--force", action="store_true",
                        help="re-fetch even chips already marked COMPLETED")
    args = parser.parse_args()

    if not (GCP_PROJECT and GCS_BUCKET):
        parser.error("GCP_PROJECT and GCS_BUCKET must be set")

    init_ee_highvol()
    gcs = storage.Client(project=GCP_PROJECT)

    manifest = pd.read_parquet(MANIFEST_URI)
    print(f"manifest: {len(manifest)} rows")

    site_info = (manifest.drop_duplicates("site_id")
                          .set_index("site_id")
                          [["site_type", "canonical_project_name", "state",
                            "lat", "lng", "announcement_date"]]
                          .to_dict("index"))

    if args.force:
        completed = set()
    else:
        completed = set(zip(manifest[manifest.export_status == "COMPLETED"].site_id,
                             manifest[manifest.export_status == "COMPLETED"].year))
    target = {(sid, y) for sid in site_info for y in S2_YEARS}
    missing = sorted(target - completed)
    if args.limit:
        missing = missing[:args.limit]
    print(f"missing: {len(missing)} chips; fetching with {args.workers} workers")

    successes: list[dict] = []
    failures: list[tuple] = []
    now = datetime.now(timezone.utc).isoformat()

    def worker(pair):
        sid, year = pair
        info = site_info[sid]
        try:
            uri = fetch_one(sid, float(info["lat"]), float(info["lng"]), year, gcs)
            return ("ok", sid, year, uri, None)
        except Exception as e:
            return ("fail", sid, year, None, f"{type(e).__name__}: {e}")

    t0 = datetime.now()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(worker, p) for p in missing]
        for i, fut in enumerate(as_completed(futures), 1):
            status, sid, year, uri, err = fut.result()
            if status == "ok":
                info = site_info[sid]
                successes.append({
                    "site_id": sid, "site_type": info["site_type"],
                    "canonical_project_name": info.get("canonical_project_name"),
                    "state": info.get("state"),
                    "lat": float(info["lat"]), "lng": float(info["lng"]),
                    "announcement_date": info.get("announcement_date"),
                    "year": int(year),
                    "aoi_buffer_m": S2_AOI_BUFFER_M,
                    "tile_size_px": S2_TILE_SIZE_PX,
                    "tile_uri": uri,
                    "export_task_id": "computePixels",
                    "export_status": "COMPLETED",
                    "created_at": now, "updated_at": now,
                })
            else:
                failures.append((sid, year, err))
            if i % 100 == 0 or i == len(missing):
                rate = i / max(1, (datetime.now() - t0).total_seconds())
                print(f"  {i}/{len(missing)}  ok={len(successes)} fail={len(failures)} ({rate:.1f}/s)")

    if failures:
        print(f"first 5 failures of {len(failures)}:")
        for sid, year, err in failures[:5]:
            print(f"  {sid} {year}: {err}")

    if successes:
        ok_keys = {(r["site_id"], r["year"]) for r in successes}
        keep_mask = manifest.apply(
            lambda r: (r["site_id"], r["year"]) not in ok_keys,
            axis=1,
        )
        manifest = manifest[keep_mask].reset_index(drop=True)
        manifest = pd.concat(
            [manifest, pd.DataFrame(successes, columns=MANIFEST_COLUMNS)],
            ignore_index=True,
        )
        manifest.to_parquet(MANIFEST_URI, index=False)
        print(f"manifest now {len(manifest)} rows; "
              + " ".join(f"{k}={v}" for k, v in manifest['export_status'].value_counts().to_dict().items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
