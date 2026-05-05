"""Phase 3 step 6: SAM segmentation runner.

Reads tiles from GCS, runs SAM AutomaticMaskGenerator, writes mask polygons as
GeoJSON back to GCS, and records mask_uri / mask_status in the Parquet manifest.

Designed to run as a Cloud Run Job (with GPU) or a Vertex AI Custom Job. Pulls
work from the manifest where export_status='COMPLETED' and mask_uri IS NULL.

Usage:
    python phase3_refinement/sam_inference.py                                # batch all pending
    python phase3_refinement/sam_inference.py --tile-uri gs://.../foo.tif    # single tile
    python phase3_refinement/sam_inference.py --limit 10                     # batch first N
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import shapes
from rasterio.io import MemoryFile
from shapely.geometry import mapping, shape

import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from google.cloud import storage

GCP_PROJECT = os.getenv("GCP_PROJECT", "")
GCS_BUCKET = os.getenv("GCS_BUCKET", "")
SAM_CHECKPOINT = os.getenv("SAM_CHECKPOINT", "/opt/sam/sam_vit_h_4b8939.pth")
SAM_MODEL_TYPE = os.getenv("SAM_MODEL_TYPE", "vit_h")

MANIFEST_URI = f"gs://{GCS_BUCKET}/manifest/tile_manifest.parquet"

_mask_gen: SamAutomaticMaskGenerator | None = None
_storage: storage.Client | None = None


def storage_client() -> storage.Client:
    global _storage
    if _storage is None:
        _storage = storage.Client(project=GCP_PROJECT)
    return _storage


def get_mask_generator() -> SamAutomaticMaskGenerator:
    global _mask_gen
    if _mask_gen is not None:
        return _mask_gen
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading SAM ({SAM_MODEL_TYPE}) on {device}")
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT).to(device)
    _mask_gen = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=0,
        min_mask_region_area=400,
    )
    return _mask_gen


def parse_uri(uri: str) -> tuple[str, str]:
    bucket_name, _, blob_path = uri[len("gs://"):].partition("/")
    return bucket_name, blob_path


def read_tile(uri: str) -> tuple[np.ndarray, rasterio.Affine, rasterio.crs.CRS]:
    bucket_name, blob_path = parse_uri(uri)
    data = storage_client().bucket(bucket_name).blob(blob_path).download_as_bytes()
    with MemoryFile(data) as mf, mf.open() as src:
        n = min(src.count, 3)
        arr = src.read(list(range(1, n + 1)))
        if n < 3:
            pad = np.repeat(arr[:1], 3 - n, axis=0)
            arr = np.concatenate([arr, pad], axis=0)
        return arr, src.transform, src.crs


def to_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr
    a = arr.astype(np.float32)
    lo, hi = np.percentile(a, (1, 99))
    if hi <= lo:
        hi = lo + 1
    a = np.clip((a - lo) / (hi - lo), 0, 1) * 255
    return a.astype(np.uint8)


def run_sam(rgb_chw: np.ndarray) -> list[dict]:
    img_hwc = np.transpose(to_uint8(rgb_chw), (1, 2, 0))
    return get_mask_generator().generate(img_hwc)


def masks_to_geojson(masks: list[dict], transform: rasterio.Affine, crs: rasterio.crs.CRS) -> dict:
    features = []
    for i, m in enumerate(masks):
        seg = m["segmentation"].astype(np.uint8)
        for geom, _ in shapes(seg, mask=seg.astype(bool), transform=transform):
            poly = shape(geom)
            if poly.is_empty or not poly.is_valid:
                continue
            features.append({
                "type": "Feature",
                "geometry": mapping(poly),
                "properties": {
                    "mask_id": i,
                    "area_px": float(m["area"]),
                    "predicted_iou": float(m["predicted_iou"]),
                    "stability_score": float(m["stability_score"]),
                    "bbox": [float(v) for v in m["bbox"]],
                },
            })
    return {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": str(crs) if crs else "unknown"}},
        "features": features,
    }


def mask_uri_for(tile_uri: str) -> str:
    bucket_name, blob_path = parse_uri(tile_uri)
    if not blob_path.startswith("tiles/"):
        raise ValueError(f"unexpected tile path: {blob_path}")
    mask_path = "masks/" + blob_path[len("tiles/"):].rsplit(".", 1)[0] + ".geojson"
    return f"gs://{bucket_name}/{mask_path}"


def write_geojson(uri: str, payload: dict) -> None:
    bucket_name, blob_path = parse_uri(uri)
    storage_client().bucket(bucket_name).blob(blob_path).upload_from_string(
        json.dumps(payload), content_type="application/geo+json"
    )


def load_manifest() -> pd.DataFrame:
    df = pd.read_parquet(MANIFEST_URI)
    if "mask_uri" not in df.columns:
        df["mask_uri"] = pd.NA
        df["mask_status"] = pd.NA
    return df


def save_manifest(df: pd.DataFrame) -> None:
    df.to_parquet(MANIFEST_URI, index=False)


def process_one(tile_uri: str) -> tuple[str, int]:
    rgb, transform, crs = read_tile(tile_uri)
    masks = run_sam(rgb)
    geojson = masks_to_geojson(masks, transform, crs)
    out_uri = mask_uri_for(tile_uri)
    write_geojson(out_uri, geojson)
    return out_uri, len(geojson["features"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tile-uri", help="single-tile mode")
    parser.add_argument("--limit", type=int, default=None, help="cap on tiles per run")
    args = parser.parse_args()

    if not (GCP_PROJECT and GCS_BUCKET):
        parser.error("GCP_PROJECT and GCS_BUCKET must be set")

    if args.tile_uri:
        out_uri, n = process_one(args.tile_uri)
        print(f"{args.tile_uri} → {out_uri} ({n} polygons)")
        return 0

    df = load_manifest()
    pending = df[(df["export_status"] == "COMPLETED") & df["mask_uri"].isna()]
    if args.limit:
        pending = pending.head(args.limit)
    print(f"processing {len(pending)} tiles")

    now = datetime.now(timezone.utc).isoformat()
    for idx, row in pending.iterrows():
        try:
            out_uri, n = process_one(row["tile_uri"])
            df.at[idx, "mask_uri"] = out_uri
            df.at[idx, "mask_status"] = "COMPLETED"
            df.at[idx, "updated_at"] = now
            print(f"[{row['project_id']}/{row['image_id']}] → {out_uri} ({n} polygons)")
        except Exception as e:
            df.at[idx, "mask_status"] = f"FAILED: {type(e).__name__}: {e}"[:500]
            df.at[idx, "updated_at"] = now
            print(f"[{row['project_id']}/{row['image_id']}] FAILED: {e}", file=sys.stderr)
            traceback.print_exc()
        save_manifest(df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
