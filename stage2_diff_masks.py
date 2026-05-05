"""Stage 2 follow-on: diff before/after mask pairs and emit candidate deltas.

For each project_id with masks bracketing announcement_date:
  1. Pick the latest pre-announcement mask as "before" and latest post-announcement
     as "after". (If only "after" masks exist, use the earliest as the before — i.e.
     the earliest available baseline.)
  2. For each polygon in `after`, compute its max overlap with any `before` polygon.
     Polygons with overlap < DIFF_OVERLAP_THRESHOLD and area >= SAM_MIN_AREA_SQM
     are emitted as candidate deltas.

Outputs:
  - GCS GeoJSON: gs://{bucket}/deltas/{project_id}/{run_id}.geojson
  - BigQuery:    {project}.{dataset}.candidate_deltas (one row per delta polygon)

Usage:
    python stage2_diff_masks.py
    python stage2_diff_masks.py --project-id <pid>     # single-project mode
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from shapely.geometry import mapping, shape
from shapely.strtree import STRtree
from pyproj import Transformer
from google.cloud import bigquery, storage

load_dotenv(Path(__file__).parent / ".env")

GCP_PROJECT = os.getenv("GCP_PROJECT", "")
GCS_BUCKET = os.getenv("GCS_BUCKET", "")
BQ_DATASET = os.getenv("BQ_DATASET", "naip_pipeline")
SAM_MIN_AREA_SQM = float(os.getenv("SAM_MIN_AREA_SQM", "4645"))
OVERLAP_THRESHOLD = float(os.getenv("DIFF_OVERLAP_THRESHOLD", "0.3"))

MANIFEST_URI = f"gs://{GCS_BUCKET}/manifest/tile_manifest.parquet"
DELTAS_TABLE = f"{GCP_PROJECT}.{BQ_DATASET}.candidate_deltas"

DELTAS_SCHEMA = [
    bigquery.SchemaField("delta_id", "STRING"),
    bigquery.SchemaField("project_id", "STRING"),
    bigquery.SchemaField("canonical_project_name", "STRING"),
    bigquery.SchemaField("state", "STRING"),
    bigquery.SchemaField("before_naip_date", "DATE"),
    bigquery.SchemaField("after_naip_date", "DATE"),
    bigquery.SchemaField("area_sqm", "FLOAT"),
    bigquery.SchemaField("max_overlap_with_before", "FLOAT"),
    bigquery.SchemaField("centroid_lat", "FLOAT"),
    bigquery.SchemaField("centroid_lng", "FLOAT"),
    bigquery.SchemaField("polygon_geojson", "STRING"),
    bigquery.SchemaField("predicted_iou", "FLOAT"),
    bigquery.SchemaField("stability_score", "FLOAT"),
    bigquery.SchemaField("delta_uri", "STRING"),
    bigquery.SchemaField("created_at", "TIMESTAMP"),
]


def parse_uri(uri: str) -> tuple[str, str]:
    bucket_name, _, blob_path = uri[len("gs://"):].partition("/")
    return bucket_name, blob_path


def read_geojson(uri: str, sc: storage.Client) -> dict:
    bucket_name, blob_path = parse_uri(uri)
    return json.loads(sc.bucket(bucket_name).blob(blob_path).download_as_bytes())


def write_geojson(uri: str, payload: dict, sc: storage.Client) -> None:
    bucket_name, blob_path = parse_uri(uri)
    sc.bucket(bucket_name).blob(blob_path).upload_from_string(
        json.dumps(payload), content_type="application/geo+json"
    )


def ensure_deltas_table(bq: bigquery.Client) -> None:
    bq.create_dataset(BQ_DATASET, exists_ok=True)
    table = bigquery.Table(DELTAS_TABLE, schema=DELTAS_SCHEMA)
    bq.create_table(table, exists_ok=True)


def crs_name_from_geojson(g: dict) -> str | None:
    try:
        return g["crs"]["properties"]["name"]
    except (KeyError, TypeError):
        return None


def pick_pair(masks_df: pd.DataFrame) -> tuple[pd.Series, pd.Series] | None:
    masks_df = masks_df.dropna(subset=["mask_uri"]).sort_values("naip_acquisition_date")
    if len(masks_df) < 2:
        return None
    befores = masks_df[masks_df["relative_to_announcement"] == "before"]
    afters = masks_df[masks_df["relative_to_announcement"] == "after"]
    if len(afters) == 0:
        return None
    if len(befores) == 0:
        # No pre-announcement coverage; use the earliest available as the baseline.
        return masks_df.iloc[0], masks_df.iloc[-1]
    return befores.iloc[-1], afters.iloc[-1]


def diff_pair(before: dict, after: dict) -> list[dict]:
    before_geoms = [shape(f["geometry"]) for f in before.get("features", [])
                    if shape(f["geometry"]).is_valid and not shape(f["geometry"]).is_empty]
    tree = STRtree(before_geoms) if before_geoms else None

    out: list[dict] = []
    for f in after.get("features", []):
        poly = shape(f["geometry"])
        if poly.is_empty or not poly.is_valid:
            continue
        max_overlap = 0.0
        if tree:
            for i in tree.query(poly):
                cand = before_geoms[i]
                inter = poly.intersection(cand).area
                if poly.area > 0:
                    max_overlap = max(max_overlap, inter / poly.area)
                if max_overlap >= 1.0:
                    break
        if max_overlap >= OVERLAP_THRESHOLD:
            continue
        out.append({"polygon": poly, "props": f.get("properties", {}), "max_overlap": max_overlap})
    return out


def project_centroid(poly_centroid_xy: tuple[float, float], src_crs: str | None) -> tuple[float, float]:
    if not src_crs or src_crs.lower() == "unknown":
        return float("nan"), float("nan")
    try:
        tr = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
        lng, lat = tr.transform(poly_centroid_xy[0], poly_centroid_xy[1])
        return lat, lng
    except Exception:
        return float("nan"), float("nan")


def emit_for_project(pid: str, manifest: pd.DataFrame, bq: bigquery.Client, sc: storage.Client,
                     min_area: float) -> int:
    site_rows = manifest[manifest["project_id"] == pid]
    pair = pick_pair(site_rows)
    if pair is None:
        return 0
    before_row, after_row = pair

    before_g = read_geojson(before_row["mask_uri"], sc)
    after_g = read_geojson(after_row["mask_uri"], sc)

    crs_name = crs_name_from_geojson(after_g)
    raw = diff_pair(before_g, after_g)
    deltas = [d for d in raw if d["polygon"].area >= min_area]
    if not deltas:
        return 0

    run_id = uuid.uuid4().hex[:12]
    delta_uri = f"gs://{GCS_BUCKET}/deltas/{pid}/{run_id}.geojson"
    now = datetime.now(timezone.utc).isoformat()

    features = []
    rows = []
    for d in deltas:
        poly = d["polygon"]
        cx, cy = poly.centroid.x, poly.centroid.y
        lat, lng = project_centroid((cx, cy), crs_name)
        delta_id = uuid.uuid4().hex
        rows.append({
            "delta_id": delta_id,
            "project_id": pid,
            "canonical_project_name": site_rows.iloc[0]["canonical_project_name"],
            "state": site_rows.iloc[0]["state"],
            "before_naip_date": before_row["naip_acquisition_date"],
            "after_naip_date": after_row["naip_acquisition_date"],
            "area_sqm": float(poly.area),
            "max_overlap_with_before": float(d["max_overlap"]),
            "centroid_lat": None if pd.isna(lat) else float(lat),
            "centroid_lng": None if pd.isna(lng) else float(lng),
            "polygon_geojson": json.dumps(mapping(poly)),
            "predicted_iou": float(d["props"].get("predicted_iou", 0.0)),
            "stability_score": float(d["props"].get("stability_score", 0.0)),
            "delta_uri": delta_uri,
            "created_at": now,
        })
        features.append({
            "type": "Feature",
            "geometry": mapping(poly),
            "properties": {
                **d["props"],
                "delta_id": delta_id,
                "max_overlap_with_before": d["max_overlap"],
                "area_sqm": float(poly.area),
            },
        })

    write_geojson(delta_uri, {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": crs_name or "unknown"}},
        "features": features,
    }, sc)

    errors = bq.insert_rows_json(DELTAS_TABLE, rows)
    if errors:
        print(f"[{pid}] BigQuery insert errors: {errors}", file=sys.stderr)
    return len(deltas)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-id", help="single-project mode")
    parser.add_argument("--min-area", type=float, default=SAM_MIN_AREA_SQM,
                        help=f"min polygon area in m² (default {SAM_MIN_AREA_SQM:g})")
    args = parser.parse_args()

    if not (GCP_PROJECT and GCS_BUCKET):
        parser.error("GCP_PROJECT and GCS_BUCKET must be set")

    bq = bigquery.Client(project=GCP_PROJECT)
    sc = storage.Client(project=GCP_PROJECT)
    ensure_deltas_table(bq)

    manifest = pd.read_parquet(MANIFEST_URI)
    eligible = manifest.dropna(subset=["mask_uri"])
    pids = [args.project_id] if args.project_id else sorted(eligible["project_id"].unique())
    print(f"diffing {len(pids)} project(s); min_area={args.min_area:g} sqm; overlap_thresh={OVERLAP_THRESHOLD}")

    total = 0
    for pid in pids:
        n = emit_for_project(pid, manifest, bq, sc, args.min_area)
        print(f"[{pid}] {n} candidate deltas")
        total += n
    print(f"total: {total} candidate deltas")
    return 0


if __name__ == "__main__":
    sys.exit(main())
