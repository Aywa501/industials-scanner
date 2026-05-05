"""Phase 2 step 1: Sentinel-2 yearly composites for anchors + random CONUS negatives.

For each site (316 anchors + N random CONUS land samples) and each year in
[S2_YEAR_START, S2_YEAR_END], builds a cloud-filtered (CLOUDY_PIXEL_PERCENTAGE
< S2_CLOUD_PCT) median composite of Sentinel-2 SR HARMONIZED imagery, exports
to GCS as 256x256 GeoTIFFs (B4/B3/B2/B8 = R/G/B/NIR), and maintains a Parquet
manifest at gs://{bucket}/manifest/s2_chip_manifest.parquet.

Random negatives are sampled in CONUS (TIGER/2018/States minus AK/HI/territories)
and rejection-filtered against any point within S2_NEG_EXCLUSION_KM of an anchor.

Usage:
    python phase2_pull_s2.py             # schedule new exports + poll once
    python phase2_pull_s2.py --resume    # poll only; do not schedule
    python phase2_pull_s2.py --dry-run   # print plan; do not call EE
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
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

GCP_PROJECT = os.getenv("GCP_PROJECT", "")
GCS_BUCKET = os.getenv("GCS_BUCKET", "")
INPUT_CSV = os.getenv(
    "INPUT_CSV",
    str(Path(__file__).parent.parent / "data_us" / "manufacturing_announcements_geocoded.csv"),
)

S2_AOI_BUFFER_M = int(os.getenv("S2_AOI_BUFFER_M", "1280"))
S2_TILE_SIZE_PX = int(os.getenv("S2_TILE_SIZE_PX", "256"))
S2_CLOUD_PCT = float(os.getenv("S2_CLOUD_PCT", "20"))
S2_YEAR_START = int(os.getenv("S2_YEAR_START", "2017"))
S2_YEAR_END = int(os.getenv("S2_YEAR_END", "2025"))
S2_YEARS = list(range(S2_YEAR_START, S2_YEAR_END + 1))
S2_BANDS = ["B4", "B3", "B2", "B8"]  # R, G, B, NIR

NEG_SAMPLES = int(os.getenv("S2_NEG_SAMPLES", "500"))
NEG_EXCLUSION_KM = float(os.getenv("S2_NEG_EXCLUSION_KM", "5"))
NEG_SEED = int(os.getenv("S2_NEG_SEED", "42"))

MANIFEST_URI = f"gs://{GCS_BUCKET}/manifest/s2_chip_manifest.parquet"

MANIFEST_COLUMNS = [
    "site_id", "site_type", "canonical_project_name", "state",
    "lat", "lng", "announcement_date", "year",
    "aoi_buffer_m", "tile_size_px", "tile_uri",
    "export_task_id", "export_status",
    "created_at", "updated_at",
]

EXCLUDED_NAMES = [
    "Alaska", "Hawaii", "Puerto Rico", "United States Virgin Islands",
    "American Samoa", "Guam", "Commonwealth of the Northern Mariana Islands",
    "District of Columbia",
]


def init_ee() -> None:
    sa = os.getenv("GEE_SERVICE_ACCOUNT") or ""
    key = os.getenv("GEE_KEY_FILE") or ""
    if sa and key:
        ee.Initialize(ee.ServiceAccountCredentials(sa, key_file=key), project=GCP_PROJECT)
    else:
        ee.Initialize(project=GCP_PROJECT)


def anchor_site_id(row: pd.Series) -> str:
    key = f"{row['canonical_project_name']}|{row['state']}|{row['lat']:.6f}|{row['lng']:.6f}"
    return "a_" + hashlib.sha1(key.encode()).hexdigest()[:10]


def negative_site_id(lat: float, lng: float) -> str:
    key = f"neg|{lat:.6f}|{lng:.6f}"
    return "n_" + hashlib.sha1(key.encode()).hexdigest()[:10]


def load_manifest() -> pd.DataFrame:
    try:
        return pd.read_parquet(MANIFEST_URI)
    except (FileNotFoundError, OSError):
        return pd.DataFrame(columns=MANIFEST_COLUMNS)


def save_manifest(df: pd.DataFrame) -> None:
    df.to_parquet(MANIFEST_URI, index=False)


def conus_geometry() -> ee.Geometry:
    states = ee.FeatureCollection("TIGER/2018/States")
    return states.filter(ee.Filter.inList("NAME", EXCLUDED_NAMES).Not()).geometry()


def sample_negatives(anchor_df: pd.DataFrame) -> list[dict]:
    """Sample random CONUS land points; reject any within NEG_EXCLUSION_KM of an anchor."""
    over = int(NEG_SAMPLES * 1.5) + 50
    print(f"sampling {over} random CONUS points (target {NEG_SAMPLES} after anchor filter)")
    pts = ee.FeatureCollection.randomPoints(region=conus_geometry(), points=over, seed=NEG_SEED)
    info = pts.getInfo()
    coords = [(f["geometry"]["coordinates"][1], f["geometry"]["coordinates"][0])
              for f in info["features"]]

    anchor_pts = anchor_df[["lat", "lng"]].dropna().to_numpy()
    R = 6371.0
    kept: list[dict] = []
    for lat, lng in coords:
        dlat = np.radians(anchor_pts[:, 0] - lat)
        dlng = np.radians(anchor_pts[:, 1] - lng)
        a = (np.sin(dlat / 2) ** 2
             + np.cos(np.radians(lat)) * np.cos(np.radians(anchor_pts[:, 0])) * np.sin(dlng / 2) ** 2)
        d_km = 2 * R * np.arcsin(np.sqrt(a))
        if d_km.min() >= NEG_EXCLUSION_KM:
            kept.append({"site_id": negative_site_id(lat, lng), "site_type": "negative",
                         "lat": lat, "lng": lng})
        if len(kept) >= NEG_SAMPLES:
            break
    print(f"  kept {len(kept)} negatives after anchor-proximity filter ({NEG_EXCLUSION_KM} km)")
    return kept


def build_anchor_sites(df_input: pd.DataFrame) -> list[dict]:
    sites = []
    for _, row in df_input.iterrows():
        if pd.isna(row.get("lat")) or pd.isna(row.get("lng")) or pd.isna(row.get("state")):
            continue
        ann_dt = pd.to_datetime(row.get("announcement_date"), errors="coerce", utc=True)
        sites.append({
            "site_id": anchor_site_id(row),
            "site_type": "anchor",
            "canonical_project_name": row["canonical_project_name"],
            "state": row["state"],
            "lat": float(row["lat"]),
            "lng": float(row["lng"]),
            "announcement_date": ann_dt.date().isoformat() if pd.notna(ann_dt) else None,
        })
    return sites


def yearly_composite(aoi: ee.Geometry, year: int) -> ee.Image:
    coll = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(f"{year}-01-01", f"{year}-12-31")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", S2_CLOUD_PCT))
    )
    return coll.median().select(S2_BANDS).clip(aoi)


def schedule_export(site_id: str, lat: float, lng: float, year: int) -> str:
    aoi = ee.Geometry.Point([lng, lat]).buffer(S2_AOI_BUFFER_M).bounds()
    img = yearly_composite(aoi, year)
    prefix = f"s2/{site_id}/{year}"
    export = ee.batch.Export.image.toCloudStorage(
        image=img,
        description=f"s2_{site_id}_{year}"[:100],
        bucket=GCS_BUCKET,
        fileNamePrefix=prefix,
        region=aoi,
        dimensions=f"{S2_TILE_SIZE_PX}x{S2_TILE_SIZE_PX}",
        fileFormat="GeoTIFF",
        maxPixels=int(1e9),
    )
    export.start()
    return export.id


def schedule_all(sites: list[dict], manifest: pd.DataFrame, dry_run: bool) -> pd.DataFrame:
    existing = set(zip(manifest["site_id"], manifest["year"])) if len(manifest) else set()
    now = datetime.now(timezone.utc).isoformat()
    new_rows: list[dict] = []
    for site in sites:
        for year in S2_YEARS:
            if (site["site_id"], year) in existing:
                continue
            if dry_run:
                print(f"[dry-run] {site['site_id']} ({site['site_type']}) {year}")
                continue
            try:
                task_id = schedule_export(site["site_id"], site["lat"], site["lng"], year)
            except Exception as e:
                if "Too many tasks" in str(e):
                    print(f"  queue full at {len(new_rows)} new exports; deferring rest to next pass")
                    if new_rows:
                        manifest = pd.concat([manifest, pd.DataFrame(new_rows, columns=MANIFEST_COLUMNS)],
                                             ignore_index=True)
                    print(f"scheduled {len(new_rows)} new S2 exports this pass")
                    return manifest
                print(f"  schedule fail {site['site_id']} {year}: {e}")
                continue
            new_rows.append({
                "site_id": site["site_id"],
                "site_type": site["site_type"],
                "canonical_project_name": site.get("canonical_project_name"),
                "state": site.get("state"),
                "lat": float(site["lat"]),
                "lng": float(site["lng"]),
                "announcement_date": site.get("announcement_date"),
                "year": int(year),
                "aoi_buffer_m": S2_AOI_BUFFER_M,
                "tile_size_px": S2_TILE_SIZE_PX,
                "tile_uri": f"gs://{GCS_BUCKET}/s2/{site['site_id']}/{year}.tif",
                "export_task_id": task_id,
                "export_status": "PENDING",
                "created_at": now,
                "updated_at": now,
            })
            if len(new_rows) % 250 == 0 and len(new_rows) > 0:
                print(f"  scheduled {len(new_rows)} so far")
    if new_rows:
        manifest = pd.concat([manifest, pd.DataFrame(new_rows, columns=MANIFEST_COLUMNS)],
                             ignore_index=True)
    print(f"scheduled {len(new_rows)} new S2 exports across {len(sites)} sites × {len(S2_YEARS)} years")
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
        print("listOperations failed 6x in a row; skipping this poll")
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
    parser.add_argument("--poll-wait", type=int, default=0,
                        help="seconds between polls (0 = single poll)")
    args = parser.parse_args()

    if not args.dry_run and not (GCP_PROJECT and GCS_BUCKET):
        parser.error("GCP_PROJECT and GCS_BUCKET must be set (copy .env.example to .env)")

    if args.dry_run:
        manifest = pd.DataFrame(columns=MANIFEST_COLUMNS)
    else:
        init_ee()
        manifest = load_manifest()
        print(f"manifest: {len(manifest)} rows ({MANIFEST_URI})")

    df_input = pd.read_csv(INPUT_CSV)
    print(f"input: {len(df_input)} anchor rows ({INPUT_CSV})")

    anchors = build_anchor_sites(df_input)
    print(f"anchor sites: {len(anchors)}")

    if args.dry_run:
        # Dry-run shortcut: stub negatives by lat/lng grid sample (no EE call)
        negatives = [{"site_id": f"n_dry{i:03d}", "site_type": "negative",
                       "lat": 35.0 + i * 0.01, "lng": -90.0 + i * 0.01}
                      for i in range(min(NEG_SAMPLES, 5))]
        print(f"[dry-run] stubbed {len(negatives)} negatives (no EE sampling)")
    else:
        negatives = sample_negatives(df_input.dropna(subset=["lat", "lng"]))

    sites = anchors + negatives
    print(f"total sites: {len(sites)} ({len(anchors)} anchors + {len(negatives)} negatives)")

    if not args.resume:
        manifest = schedule_all(sites, manifest, dry_run=args.dry_run)
        if not args.dry_run:
            save_manifest(manifest)

    if args.dry_run:
        return 0

    target_keys = {(s["site_id"], y) for s in sites for y in S2_YEARS}
    while True:
        try:
            manifest = poll_once(manifest)
            scheduled_keys = set(zip(manifest["site_id"], manifest["year"])) if len(manifest) else set()
            missing = target_keys - scheduled_keys
            if missing and not args.resume:
                print(f"re-scheduling pass: {len(missing)} (site, year) pairs still unqueued")
                manifest = schedule_all(sites, manifest, dry_run=False)
            save_manifest(manifest)
        except (ConnectionError, OSError) as e:
            print(f"poll/save transient error ({type(e).__name__}: {e}); will retry")
        if args.poll_wait <= 0:
            break
        all_queued = not (target_keys - set(zip(manifest["site_id"], manifest["year"])))
        no_pending = not manifest["export_status"].isin(["PENDING", "RUNNING"]).any()
        if all_queued and no_pending:
            break
        time.sleep(args.poll_wait)

    summary = manifest["export_status"].value_counts().to_dict()
    print(f"final manifest status: {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
