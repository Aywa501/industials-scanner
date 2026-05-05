"""Stage 1: NAIP tile puller.

Reads the geocoded manufacturing CSV, builds a per-site AOI, queries NAIP imagery
on Earth Engine, kicks off exports to GCS, and maintains a Parquet manifest in GCS.

The manifest is the source of truth: re-runs schedule only missing exports and
refresh statuses for in-flight tasks.

Usage:
    python stage1_pull_tiles.py             # schedule new exports + poll once
    python stage1_pull_tiles.py --resume    # poll only; do not schedule
    python stage1_pull_tiles.py --dry-run   # print plan; do not call EE
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import ee
import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

GCP_PROJECT = os.getenv("GCP_PROJECT", "")
GCS_BUCKET = os.getenv("GCS_BUCKET", "")
INPUT_CSV = os.getenv(
    "INPUT_CSV",
    str(Path(__file__).parent.parent / "data_us" / "manufacturing_announcements_geocoded.csv"),
)
AOI_BUFFER_M = int(os.getenv("AOI_BUFFER_M", "307"))
TILE_SIZE_PX = int(os.getenv("TILE_SIZE_PX", "1024"))
NAIP_START = os.getenv("NAIP_DATE_START", "2015-01-01")
NAIP_END = os.getenv("NAIP_DATE_END", "2025-12-31")

MANIFEST_URI = f"gs://{GCS_BUCKET}/manifest/tile_manifest.parquet"

MANIFEST_COLUMNS = [
    "project_id", "canonical_project_name", "state", "lat", "lng",
    "coordinate_precision", "announcement_date", "naip_acquisition_date",
    "naip_year", "relative_to_announcement", "image_id", "tile_uri",
    "aoi_buffer_m", "tile_size_px", "export_task_id", "export_status",
    "created_at", "updated_at",
]


def init_ee() -> None:
    sa = os.getenv("GEE_SERVICE_ACCOUNT") or ""
    key = os.getenv("GEE_KEY_FILE") or ""
    if sa and key:
        ee.Initialize(ee.ServiceAccountCredentials(sa, key_file=key), project=GCP_PROJECT)
    else:
        ee.Initialize(project=GCP_PROJECT)


def project_id_for(row: pd.Series) -> str:
    key = f"{row['canonical_project_name']}|{row['state']}|{row['lat']:.6f}|{row['lng']:.6f}"
    return hashlib.sha1(key.encode()).hexdigest()[:12]


def coord_precision(lat: float, lng: float) -> str:
    def dp(x: float) -> int:
        s = f"{x:.10f}".rstrip("0")
        return len(s.split(".")[1]) if "." in s else 0
    p = min(dp(lat), dp(lng))
    if p >= 5:
        return "parcel"
    if p >= 3:
        return "city"
    return "approximate"


def load_manifest() -> pd.DataFrame:
    try:
        return pd.read_parquet(MANIFEST_URI)
    except (FileNotFoundError, OSError):
        return pd.DataFrame(columns=MANIFEST_COLUMNS)


def save_manifest(df: pd.DataFrame) -> None:
    df.to_parquet(MANIFEST_URI, index=False)


def schedule_for_row(row: pd.Series, existing_keys: set[tuple[str, str]]) -> list[dict]:
    pid = project_id_for(row)
    aoi = ee.Geometry.Point([row["lng"], row["lat"]]).buffer(AOI_BUFFER_M).bounds()
    coll = (
        ee.ImageCollection("USDA/NAIP/DOQQ")
        .filterBounds(aoi)
        .filterDate(NAIP_START, NAIP_END)
        .sort("system:time_start")
    )
    info = coll.getInfo()
    features = info.get("features", []) if info else []
    if not features:
        print(f"[{pid}] no NAIP imagery in window")
        return []

    precision = coord_precision(row["lat"], row["lng"])
    ann_dt = pd.to_datetime(row.get("announcement_date"), errors="coerce", utc=True)

    rows: list[dict] = []
    now = datetime.now(timezone.utc).isoformat()
    for feat in features:
        image_id = feat["id"].split("/")[-1]
        if (pid, image_id) in existing_keys:
            continue

        ts_ms = feat["properties"].get("system:time_start")
        if ts_ms is None:
            continue
        naip_dt = pd.Timestamp(ts_ms, unit="ms", tz="UTC")
        naip_date_iso = naip_dt.date().isoformat()
        relative = "unknown"
        if pd.notna(ann_dt):
            relative = "before" if naip_dt < ann_dt else "after"

        prefix = f"tiles/{pid}/{naip_dt.year}_{image_id}"
        tile_uri = f"gs://{GCS_BUCKET}/{prefix}.tif"

        export = ee.batch.Export.image.toCloudStorage(
            image=ee.Image(feat["id"]).select(["R", "G", "B", "N"]).clip(aoi),
            description=f"naip_{pid}_{naip_dt.year}_{image_id}"[:100],
            bucket=GCS_BUCKET,
            fileNamePrefix=prefix,
            region=aoi,
            dimensions=f"{TILE_SIZE_PX}x{TILE_SIZE_PX}",
            fileFormat="GeoTIFF",
            maxPixels=int(1e9),
        )
        export.start()

        rows.append({
            "project_id": pid,
            "canonical_project_name": row["canonical_project_name"],
            "state": row["state"],
            "lat": float(row["lat"]),
            "lng": float(row["lng"]),
            "coordinate_precision": precision,
            "announcement_date": ann_dt.date().isoformat() if pd.notna(ann_dt) else None,
            "naip_acquisition_date": naip_date_iso,
            "naip_year": naip_dt.year,
            "relative_to_announcement": relative,
            "image_id": image_id,
            "tile_uri": tile_uri,
            "aoi_buffer_m": AOI_BUFFER_M,
            "tile_size_px": TILE_SIZE_PX,
            "export_task_id": export.id,
            "export_status": "PENDING",
            "created_at": now,
            "updated_at": now,
        })
        print(f"[{pid}] scheduled {naip_date_iso} → {tile_uri}")
    return rows


def schedule_all(df_input: pd.DataFrame, manifest: pd.DataFrame, dry_run: bool) -> pd.DataFrame:
    existing = set(zip(manifest["project_id"], manifest["image_id"])) if len(manifest) else set()
    new_rows: list[dict] = []
    skipped = 0
    for _, row in df_input.iterrows():
        if pd.isna(row.get("lat")) or pd.isna(row.get("lng")) or pd.isna(row.get("state")):
            skipped += 1
            continue
        if dry_run:
            pid = project_id_for(row)
            precision = coord_precision(row["lat"], row["lng"])
            print(f"[dry-run] {pid} {row['state']} {row['lat']:.5f},{row['lng']:.5f} precision={precision}")
            continue
        new_rows.extend(schedule_for_row(row, existing))
    print(f"scheduled {len(new_rows)} new exports, skipped {skipped} rows for missing lat/lng/state")
    if new_rows:
        manifest = pd.concat([manifest, pd.DataFrame(new_rows, columns=MANIFEST_COLUMNS)], ignore_index=True)
    return manifest


def poll_once(manifest: pd.DataFrame) -> pd.DataFrame:
    if not len(manifest):
        return manifest
    pending_mask = manifest["export_status"].isin(["PENDING", "RUNNING"])
    if not pending_mask.any():
        print("no pending exports")
        return manifest

    operations = None
    for attempt in range(6):
        try:
            operations = ee.data.listOperations()
            break
        except (ConnectionError, OSError) as e:
            wait = min(60, 5 * (2 ** attempt))
            print(f"listOperations failed ({type(e).__name__}); retry in {wait}s")
            time.sleep(wait)
    if operations is None:
        print("listOperations failed 6× in a row; skipping this poll")
        return manifest
    state_by_id: dict[str, str] = {}
    for op in operations:
        op_id = op["name"].rsplit("/", 1)[-1]
        md_state = ((op.get("metadata") or {}).get("state") or "").upper()
        if op.get("done"):
            if md_state == "SUCCEEDED":
                state_by_id[op_id] = "COMPLETED"
            elif md_state in ("CANCELLED", "CANCELLING"):
                state_by_id[op_id] = "CANCELLED"
            else:
                state_by_id[op_id] = "FAILED" if "error" in op or md_state == "FAILED" else "COMPLETED"
        else:
            state_by_id[op_id] = md_state or "RUNNING"

    now = datetime.now(timezone.utc).isoformat()
    updated = 0
    for idx in manifest.index[pending_mask]:
        task_id = manifest.at[idx, "export_task_id"]
        new_state = state_by_id.get(task_id)
        if new_state and new_state != manifest.at[idx, "export_status"]:
            manifest.at[idx, "export_status"] = new_state
            manifest.at[idx, "updated_at"] = now
            updated += 1
    print(f"polled {pending_mask.sum()} pending; updated {updated}")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true", help="poll only; do not schedule")
    parser.add_argument("--dry-run", action="store_true", help="print plan; no EE calls")
    parser.add_argument("--poll-wait", type=int, default=0, help="seconds to wait between polls (0 = single poll)")
    args = parser.parse_args()

    if args.dry_run:
        manifest = pd.DataFrame(columns=MANIFEST_COLUMNS)
    else:
        if not (GCP_PROJECT and GCS_BUCKET):
            parser.error("GCP_PROJECT and GCS_BUCKET must be set (copy .env.example to .env)")
        init_ee()
        manifest = load_manifest()
        print(f"manifest: {len(manifest)} rows ({MANIFEST_URI})")

    df_input = pd.read_csv(INPUT_CSV)
    print(f"input: {len(df_input)} rows ({INPUT_CSV})")

    if not args.resume:
        manifest = schedule_all(df_input, manifest, dry_run=args.dry_run)
        if not args.dry_run:
            save_manifest(manifest)

    if args.dry_run:
        return 0

    while True:
        try:
            manifest = poll_once(manifest)
            save_manifest(manifest)
        except (ConnectionError, OSError) as e:
            print(f"poll/save transient error ({type(e).__name__}: {e}); will retry")
        if args.poll_wait <= 0:
            break
        if not manifest["export_status"].isin(["PENDING", "RUNNING"]).any():
            break
        time.sleep(args.poll_wait)

    summary = manifest["export_status"].value_counts().to_dict()
    print(f"final manifest status: {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
