"""Render chips + build queue for the relabel-shortlist webapp pass.

Reads:
- data_us/stage1_relabel_shortlist.json   (50 chips × 12 sites)
- gs://{GCS_BUCKET}/manifest/s2_chip_manifest.parquet

For each site in the shortlist, downloads + renders ALL 9 years (gives the
labeler temporal context) and writes:
- sites_us/.artifacts/labeling/chips/{site_id}/{year}.png
- sites_us/.artifacts/labeling/shortlist_queue.json
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google.cloud import storage
from PIL import Image
from rasterio.io import MemoryFile

load_dotenv(Path(__file__).parent.parent.parent / ".env")

GCP_PROJECT = os.getenv("GCP_PROJECT", "")
GCS_BUCKET = os.getenv("GCS_BUCKET", "")
MANIFEST_URI = f"gs://{GCS_BUCKET}/manifest/s2_chip_manifest.parquet"

ROOT = Path(__file__).parent.parent.parent
ARTIFACTS = ROOT / ".artifacts" / "labeling"
CHIPS_DIR = ARTIFACTS / "chips"
QUEUE_PATH = ARTIFACTS / "shortlist_queue.json"

DATA_US = ROOT.parent / "data_us"
SHORTLIST_PATH = DATA_US / "stage1_relabel_shortlist.json"

RENDER_PX = 512


def parse_uri(uri: str) -> tuple[str, str]:
    bucket_name, _, blob_path = uri[len("gs://"):].partition("/")
    return bucket_name, blob_path


def render_tiff_to_png(tiff_bytes: bytes, out_path: Path, dim: int) -> None:
    with MemoryFile(tiff_bytes) as mf, mf.open() as src:
        rgb = src.read([1, 2, 3]).astype(np.float32)
    lo, hi = np.percentile(rgb, (1, 99))
    if hi <= lo:
        hi = lo + 1
    arr = np.clip((rgb - lo) / (hi - lo), 0, 1) * 255
    arr = arr.astype(np.uint8).transpose(1, 2, 0)
    img = Image.fromarray(arr).resize((dim, dim), Image.LANCZOS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, "PNG", optimize=False)


def process_one(site_id: str, year: int, tile_uri: str,
                gcs: storage.Client) -> tuple[str, int, bool, str | None]:
    out = CHIPS_DIR / site_id / f"{year}.png"
    if out.exists():
        return (site_id, year, True, None)
    try:
        _, blob_path = parse_uri(tile_uri)
        tiff = gcs.bucket(GCS_BUCKET).blob(blob_path).download_as_bytes()
        render_tiff_to_png(tiff, out, RENDER_PX)
        return (site_id, year, True, None)
    except Exception as e:
        return (site_id, year, False, f"{type(e).__name__}: {e}")


def main() -> int:
    if not (GCP_PROJECT and GCS_BUCKET):
        print("error: GCP_PROJECT and GCS_BUCKET must be set", file=sys.stderr)
        return 1

    shortlist = json.loads(SHORTLIST_PATH.read_text())
    flagged_by_site: dict[str, list[dict]] = {}
    for s in shortlist:
        flagged_by_site.setdefault(s["site_id"], []).append(s)

    sl_sites = sorted(flagged_by_site.keys())
    print(f"shortlist: {len(shortlist)} chips × {len(sl_sites)} unique sites")

    print(f"reading manifest from {MANIFEST_URI}")
    manifest = pd.read_parquet(MANIFEST_URI)
    completed = manifest[manifest.export_status == "COMPLETED"].copy()
    sub = completed[completed.site_id.isin(sl_sites)].copy()
    print(f"  {len(sub)} chips to ensure (across {sub.site_id.nunique()} sites)")

    gcs = storage.Client(project=GCP_PROJECT)
    jobs = [(r.site_id, int(r.year), r.tile_uri) for r in sub.itertuples(index=False)]
    print(f"downloading + rendering {len(jobs)} chips with 16 workers...")
    failures: list = []
    done = 0
    with ThreadPoolExecutor(max_workers=16) as ex:
        futs = [ex.submit(process_one, sid, y, uri, gcs) for sid, y, uri in jobs]
        for f in as_completed(futs):
            sid, year, ok, err = f.result()
            done += 1
            if not ok:
                failures.append((sid, year, err))
            if done % 20 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)} (failures: {len(failures)})")
    if failures:
        print(f"first 5 failures of {len(failures)}:")
        for sid, year, err in failures[:5]:
            print(f"  {sid} {year}: {err}")

    site_meta = sub.drop_duplicates("site_id").set_index("site_id")
    queue = []
    for sid in sl_sites:
        meta = site_meta.loc[sid]
        years_present = sorted(int(y) for y in sub[sub.site_id == sid].year.unique()
                                if (CHIPS_DIR / sid / f"{int(y)}.png").exists())
        if not years_present:
            continue
        flagged_years = sorted({int(s["year"]) for s in flagged_by_site[sid]})
        queue.append({
            "site_id": sid,
            "site_type": meta.site_type,
            "project_type": None,
            "canonical_project_name": None,
            "state": meta.state if pd.notna(meta.state) else None,
            "lat": float(meta.lat),
            "lng": float(meta.lng),
            "ann_year": None,
            "years": years_present,
            "flagged_years": flagged_years,
            "shortlist_meta": flagged_by_site[sid],
        })

    QUEUE_PATH.write_text(json.dumps(queue, indent=2))
    print(f"\nwrote {QUEUE_PATH} ({len(queue)} sites)")
    print(f"chips cache → {CHIPS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
