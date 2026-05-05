"""One-shot prep for the manual-labeling webapp.

1. Read s2_chip_manifest.parquet from GCS.
2. Stratified-sample ~120 anchors (preferring ann_year in [2017, 2023] where
   the not_a_site → partial → complete transition is observable in S2) and 15
   random CONUS sites.
3. Download each selected chip's GeoTIFF from GCS.
4. Render B4/B3/B2 as RGB PNG with 1–99 percentile stretch, upsampled to 512px.
5. Write queue.json describing the strip per site.

Outputs (under sites_us/.artifacts/labeling/):
    chips/{site_id}/{year}.png
    queue.json
"""

from __future__ import annotations

import io
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from dotenv import load_dotenv
from google.cloud import storage
from rasterio.io import MemoryFile

load_dotenv(Path(__file__).parent.parent.parent / ".env")

GCP_PROJECT = os.getenv("GCP_PROJECT", "")
GCS_BUCKET = os.getenv("GCS_BUCKET", "")
MANIFEST_URI = f"gs://{GCS_BUCKET}/manifest/s2_chip_manifest.parquet"

ROOT = Path(__file__).parent.parent.parent
ARTIFACTS = ROOT / ".artifacts" / "labeling"
CHIPS_DIR = ARTIFACTS / "chips"
QUEUE_PATH = ARTIFACTS / "queue.json"

ANCHORS_CSV = ROOT.parent / "data_us" / "manufacturing_announcements_geocoded.csv"

# Per project_type targets — expansions are skipped almost entirely because
# they're trivially industrial pre-announcement (the press release names an
# existing address). Greenfield is where partial→complete transitions matter.
N_GREENFIELD = 35
N_BROWNFIELD = 12
N_EXPANSION = 5
N_NEGATIVES = 50
RENDER_PX = 512
DECIDABLE_YEARS = range(2017, 2024)  # ann_year ∈ [2017, 2023]
MIN_GEOCODE_DP = 4  # exclude city-level geocodes (min decimal places of lat/lng)
SEED = 42


def parse_uri(uri: str) -> tuple[str, str]:
    bucket_name, _, blob_path = uri[len("gs://"):].partition("/")
    return bucket_name, blob_path


def render_tiff_to_png(tiff_bytes: bytes, out_path: Path, dim: int) -> None:
    with MemoryFile(tiff_bytes) as mf, mf.open() as src:
        # Bands are B4, B3, B2, B8 in that order
        rgb = src.read([1, 2, 3]).astype(np.float32)
    lo, hi = np.percentile(rgb, (1, 99))
    if hi <= lo:
        hi = lo + 1
    arr = np.clip((rgb - lo) / (hi - lo), 0, 1) * 255
    arr = arr.astype(np.uint8).transpose(1, 2, 0)  # CHW -> HWC
    img = Image.fromarray(arr).resize((dim, dim), Image.LANCZOS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, "PNG", optimize=False)


def stratified_anchor_sample(manifest: pd.DataFrame, project_type_by_name: dict,
                              n_per_type: dict[str, int], rng: random.Random) -> list[str]:
    """Pick anchors stratified by (project_type, ann_year), with per-type caps.

    Within each project_type, prefer ann_years in DECIDABLE_YEARS (the window
    where pre/post construction is observable in S2) via round-robin sampling.
    """
    anchors = manifest[manifest.site_type == "anchor"].drop_duplicates("site_id").copy()
    anchors["ann_year"] = pd.to_datetime(anchors.announcement_date).dt.year
    anchors["project_type"] = anchors.canonical_project_name.map(project_type_by_name)

    out: list[str] = []
    for ptype, target in n_per_type.items():
        pool = anchors[anchors.project_type == ptype].copy()
        decidable = pool[pool.ann_year.isin(DECIDABLE_YEARS)]
        edge = pool[~pool.ann_year.isin(DECIDABLE_YEARS)]

        n_dec = min(int(round(target * 0.85)), len(decidable))
        n_edge = min(target - n_dec, len(edge))

        groups = []
        for _, g in decidable.groupby("ann_year"):
            ids = g.site_id.tolist()
            rng.shuffle(ids)
            groups.append(ids)
        chosen_dec: list[str] = []
        while len(chosen_dec) < n_dec and any(g for g in groups):
            for ids in groups:
                if ids and len(chosen_dec) < n_dec:
                    chosen_dec.append(ids.pop())

        edge_ids = edge.site_id.tolist()
        rng.shuffle(edge_ids)
        chosen_edge = edge_ids[:n_edge]

        out.extend(chosen_dec + chosen_edge)

    return out


def download_chip(blob_path: str, gcs: storage.Client) -> bytes:
    return gcs.bucket(GCS_BUCKET).blob(blob_path).download_as_bytes()


def process_one(site_id: str, year: int, tile_uri: str, gcs: storage.Client) -> tuple[str, int, bool, str | None]:
    out = CHIPS_DIR / site_id / f"{year}.png"
    if out.exists():
        return (site_id, year, True, None)
    try:
        _, blob_path = parse_uri(tile_uri)
        tiff = download_chip(blob_path, gcs)
        render_tiff_to_png(tiff, out, RENDER_PX)
        return (site_id, year, True, None)
    except Exception as e:
        return (site_id, year, False, f"{type(e).__name__}: {e}")


def main() -> int:
    if not (GCP_PROJECT and GCS_BUCKET):
        print("error: GCP_PROJECT and GCS_BUCKET must be set", file=sys.stderr)
        return 1

    rng = random.Random(SEED)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    print(f"reading manifest from {MANIFEST_URI}")
    manifest = pd.read_parquet(MANIFEST_URI)
    completed = manifest[manifest.export_status == "COMPLETED"].copy()
    print(f"  {len(manifest)} rows, {len(completed)} completed")

    anchors_csv = pd.read_csv(ANCHORS_CSV)
    project_type_by_name = dict(zip(anchors_csv.canonical_project_name, anchors_csv.site_type))

    # Drop city-level geocodes: read raw strings to count decimal places.
    raw = pd.read_csv(ANCHORS_CSV, dtype={"lat": str, "lng": str})
    def _dp(s: str) -> int:
        return len(s.split(".", 1)[1]) if isinstance(s, str) and "." in s else 0
    raw["min_dp"] = raw.apply(lambda r: min(_dp(r.lat), _dp(r.lng)), axis=1)
    low_precision = set(raw.loc[raw.min_dp < MIN_GEOCODE_DP, "canonical_project_name"])
    print(f"  excluding {len(low_precision)} city-level geocodes (min_dp < {MIN_GEOCODE_DP})")

    completed_for_anchors = completed[~completed.canonical_project_name.isin(low_precision)]
    anchor_ids = stratified_anchor_sample(
        completed_for_anchors, project_type_by_name,
        {"greenfield": N_GREENFIELD, "brownfield": N_BROWNFIELD,
         "expansion_existing": N_EXPANSION},
        rng,
    )
    neg_ids = completed[completed.site_type != "anchor"].site_id.drop_duplicates().tolist()
    rng.shuffle(neg_ids)
    neg_ids = neg_ids[:N_NEGATIVES]

    site_ids = anchor_ids + neg_ids
    pt_counts = {pt: sum(1 for sid in anchor_ids
                         if project_type_by_name.get(
                             completed[completed.site_id == sid].canonical_project_name.iloc[0]
                         ) == pt)
                 for pt in ["greenfield", "brownfield", "expansion_existing"]}
    print(f"  selected {len(anchor_ids)} anchors ({pt_counts}) + {len(neg_ids)} negatives = {len(site_ids)}")

    sub = completed[completed.site_id.isin(site_ids)].copy()
    print(f"  {len(sub)} chips to process")

    gcs = storage.Client(project=GCP_PROJECT)

    jobs = [(r.site_id, int(r.year), r.tile_uri) for r in sub.itertuples(index=False)]
    print(f"downloading + rendering {len(jobs)} chips with 20 workers...")
    failures: list[tuple] = []
    done = 0
    with ThreadPoolExecutor(max_workers=20) as ex:
        futs = [ex.submit(process_one, sid, y, uri, gcs) for sid, y, uri in jobs]
        for f in as_completed(futs):
            sid, year, ok, err = f.result()
            done += 1
            if not ok:
                failures.append((sid, year, err))
            if done % 100 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)} (failures: {len(failures)})")

    if failures:
        print(f"first 5 failures of {len(failures)}:")
        for sid, year, err in failures[:5]:
            print(f"  {sid} {year}: {err}")

    # Build queue.json
    queue: list[dict] = []
    site_meta = (sub.drop_duplicates("site_id")
                    .set_index("site_id"))
    for sid in site_ids:
        if sid not in site_meta.index:
            continue
        meta = site_meta.loc[sid]
        years_present = sorted(int(y) for y in sub[sub.site_id == sid].year.unique()
                               if (CHIPS_DIR / sid / f"{int(y)}.png").exists())
        if not years_present:
            continue
        ann_year = (pd.to_datetime(meta.announcement_date).year
                    if meta.site_type == "anchor" and pd.notna(meta.announcement_date)
                    else None)
        proj_name = (meta.canonical_project_name
                     if pd.notna(meta.canonical_project_name) else None)
        project_type = project_type_by_name.get(proj_name) if proj_name else None
        queue.append({
            "site_id": sid,
            "site_type": meta.site_type,
            "project_type": project_type,
            "canonical_project_name": proj_name,
            "state": meta.state if pd.notna(meta.state) else None,
            "lat": float(meta.lat),
            "lng": float(meta.lng),
            "ann_year": int(ann_year) if ann_year is not None else None,
            "years": years_present,
        })

    QUEUE_PATH.write_text(json.dumps(queue, indent=2))
    print(f"wrote queue with {len(queue)} sites → {QUEUE_PATH}")
    print(f"chips cache → {CHIPS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
