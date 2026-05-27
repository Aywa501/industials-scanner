"""Build chip strips + queue for wild-FP bucketing pass.

Reads:
- data_us/phase1/stage1_wild_topk.csv         per-site top-K wild predictions
- gs://{GCS_BUCKET}/manifest/s2_chip_manifest.parquet

Outputs:
- sites_us/.artifacts/labeling/chips/{site_id}/{year}.png
- sites_us/.artifacts/labeling/wild_topk_queue.json

Usage in webapp:
  LABELING_QUEUE_FILE=wild_topk_queue.json python -m uvicorn ...

Use the note input to type a category (quarry, solar, big_box, parking,
data_center, mine, landfill, airport, agri_processing, dense_suburban,
industrial, other). Categories get extracted by category_summary.py.
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
QUEUE_PATH = ARTIFACTS / "wild_topk_queue.json"

DATA_US = ROOT.parent / "data_us"
TOPK_CSV = DATA_US / "phase1" / "stage1_wild_topk.csv"

N_SITES = 50  # how many top-K sites to surface for bucketing
RENDER_PX = 512


def parse_uri(uri: str) -> tuple[str, str]:
    bucket, _, blob = uri[len("gs://"):].partition("/")
    return bucket, blob


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


def process_one(site_id, year, tile_uri, gcs):
    out = CHIPS_DIR / site_id / f"{year}.png"
    if out.exists():
        return (site_id, year, True, None)
    try:
        _, blob = parse_uri(tile_uri)
        tiff = gcs.bucket(GCS_BUCKET).blob(blob).download_as_bytes()
        render_tiff_to_png(tiff, out, RENDER_PX)
        return (site_id, year, True, None)
    except Exception as e:
        return (site_id, year, False, f"{type(e).__name__}: {e}")


def main() -> int:
    if not (GCP_PROJECT and GCS_BUCKET):
        print("error: GCP_PROJECT and GCS_BUCKET must be set", file=sys.stderr)
        return 1

    df = pd.read_csv(TOPK_CSV).head(N_SITES)
    sites = df.site_id.tolist()
    best_year_by_site = dict(zip(df.site_id, df.year))
    prob_by_site = dict(zip(df.site_id, df.prob))
    print(f"surfacing top {len(sites)} wild predictions for FP bucketing")

    print(f"reading manifest from {MANIFEST_URI}")
    manifest = pd.read_parquet(MANIFEST_URI)
    completed = manifest[manifest.export_status == "COMPLETED"].copy()
    sub = completed[completed.site_id.isin(sites)].copy()
    print(f"  {len(sub)} chips to ensure across {sub.site_id.nunique()} sites")

    gcs = storage.Client(project=GCP_PROJECT)
    jobs = [(r.site_id, int(r.year), r.tile_uri) for r in sub.itertuples(index=False)]
    failures = []
    done = 0
    with ThreadPoolExecutor(max_workers=16) as ex:
        futs = [ex.submit(process_one, sid, y, uri, gcs) for sid, y, uri in jobs]
        for f in as_completed(futs):
            sid, year, ok, err = f.result()
            done += 1
            if not ok:
                failures.append((sid, year, err))
            if done % 50 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)} (failures: {len(failures)})")

    queue = []
    site_meta = sub.drop_duplicates("site_id").set_index("site_id")
    for sid in sites:
        if sid not in site_meta.index:
            continue
        meta = site_meta.loc[sid]
        years_present = sorted(int(y) for y in sub[sub.site_id == sid].year.unique()
                                if (CHIPS_DIR / sid / f"{int(y)}.png").exists())
        if not years_present:
            continue
        queue.append({
            "site_id": sid,
            "site_type": "negative",
            "project_type": None,
            "canonical_project_name": f"wild prediction (p={prob_by_site[sid]:.2f})",
            "state": meta.state if pd.notna(meta.state) else None,
            "lat": float(meta.lat),
            "lng": float(meta.lng),
            "ann_year": None,
            "years": years_present,
            "flagged_years": [int(best_year_by_site[sid])],
            "model_prob": float(prob_by_site[sid]),
        })

    QUEUE_PATH.write_text(json.dumps(queue, indent=2))
    print(f"\nwrote {QUEUE_PATH} ({len(queue)} sites)")
    print(f"chips cache → {CHIPS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
