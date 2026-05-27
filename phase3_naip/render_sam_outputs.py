"""Render a cluster's SAM 3 output as a polygon overlay on the NAIP imagery.

Pulls masks.parquet from S3, refetches the NAIP mosaic, draws each mask polygon
in a per-label colour, saves PNG. For showing-the-work / qualitative QA only;
not part of the production pipeline.

Usage:
  python -m phase3_naip.render_sam_outputs <cluster_id> [<cluster_id> ...]
  python -m phase3_naip.render_sam_outputs --auto 8        # pick 8 varied clusters
"""
from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
import pyproj
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont
from shapely import wkt as shapely_wkt
from shapely.ops import transform as shapely_transform

SITES_US = Path(__file__).resolve().parents[1]
load_dotenv(SITES_US / ".env")

sys.path.insert(0, str(SITES_US))
from phase3_naip.naip_sam import (  # noqa: E402
    OUTPUT_BUCKET, OUTPUT_PREFIX, MANIFEST_PATH,
    _rasterio_env, read_naip_mosaic, to_uint8_rgb,
)

OUT_DIR = SITES_US.parent / "data_us" / "phase3_naip" / "renders"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Colour map per label (RGB tuples). 9 prompts × distinct hues.
LABEL_COLORS = {
    "industrial building": (255, 60, 60),
    "warehouse":           (255, 165, 0),
    "office building":     (255, 230, 50),
    "parking lot":         (140, 110, 200),
    "loading dock":        (255, 50, 220),
    "storage tank":        (50, 220, 220),
    "silo":                (90, 230, 110),
    "vegetation":          (40, 160, 60),
    "road":                (220, 220, 220),
}


def fetch_masks(cluster_id: str) -> pd.DataFrame:
    s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-west-2"))
    key = f"{OUTPUT_PREFIX}/{cluster_id}/masks.parquet"
    obj = s3.get_object(Bucket=OUTPUT_BUCKET, Key=key)
    return pd.read_parquet(io.BytesIO(obj["Body"].read()))


def render_cluster(row: pd.Series, masks: pd.DataFrame) -> Image.Image | None:
    cid = row.cluster_id
    with _rasterio_env():
        mosaic = read_naip_mosaic(
            list(row.naip_uris),
            float(row.fetch_lon_min), float(row.fetch_lat_min),
            float(row.fetch_lon_max), float(row.fetch_lat_max))
    if mosaic is None:
        print(f"[render] {cid}: read failed")
        return None
    arr, transform = mosaic
    rgb = to_uint8_rgb(arr)
    h, w = rgb.shape[:2]

    base = Image.fromarray(rgb).convert("RGBA")
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Transform geom from EPSG:4326 (parquet WKT) back to mosaic pixel coords.
    to_5070 = pyproj.Transformer.from_crs(4326, 5070, always_xy=True).transform
    inv = ~transform  # affine inverse: world(5070) -> pixel
    for _, m in masks.iterrows():
        poly = shapely_wkt.loads(m["geom_wkt"])
        poly_5070 = shapely_transform(lambda x, y, z=None: to_5070(x, y), poly)
        polys = [poly_5070] if poly_5070.geom_type == "Polygon" else list(poly_5070.geoms)
        color = LABEL_COLORS.get(m["label"], (200, 200, 200))
        fill = (*color, 80)
        outline = (*color, 255)
        for p in polys:
            if p.is_empty:
                continue
            xs, ys = p.exterior.coords.xy
            pixels = [inv * (x, y) for x, y in zip(xs, ys)]
            draw.polygon(pixels, fill=fill, outline=outline, width=2)

    out = Image.alpha_composite(base, overlay)
    # Caption strip
    caption_h = 80
    canvas = Image.new("RGBA", (w, h + caption_h), (20, 20, 20, 255))
    canvas.paste(out, (0, caption_h))
    cd = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
        font_sm = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 13)
    except Exception:
        font = ImageFont.load_default()
        font_sm = ImageFont.load_default()
    cd.text((10, 6),
            f"{cid}   {h}×{w}px   {len(masks)} masks   "
            f"buildings={int(getattr(row, 'n_buildings', 0) or 0)}   "
            f"span={int(getattr(row, 'span_m', 0) or 0)}m",
            fill=(255, 255, 255, 255), font=font)
    # Legend
    counts = masks["label"].value_counts().to_dict()
    legend_items = [(lbl, counts.get(lbl, 0), color)
                    for lbl, color in LABEL_COLORS.items() if counts.get(lbl, 0) > 0]
    x = 10
    for lbl, n, color in legend_items:
        cd.rectangle([x, 36, x + 12, 48], fill=(*color, 255))
        cd.text((x + 16, 33), f"{lbl}({n})", fill=(220, 220, 220, 255), font=font_sm)
        x += 18 + cd.textlength(f"{lbl}({n})", font=font_sm) + 12
    return canvas.convert("RGB")


def pick_auto_clusters(manifest: pd.DataFrame, n: int) -> list[str]:
    """Pick n clusters with outputs on S3, spanning small/medium/large mosaics."""
    s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-west-2"))
    paginator = s3.get_paginator("list_objects_v2")
    existing = set()
    for page in paginator.paginate(Bucket=OUTPUT_BUCKET, Prefix=OUTPUT_PREFIX + "/"):
        for obj in page.get("Contents", []) or []:
            parts = obj["Key"].split("/")
            if len(parts) >= 3 and parts[-1] == "masks.parquet":
                existing.add(parts[-2])
            if len(existing) > 2000:
                break
        if len(existing) > 2000:
            break
    avail = manifest[manifest.cluster_id.isin(existing)].copy()
    avail["est_mp"] = (avail["fetch_lat_max"] - avail["fetch_lat_min"]) * 111e3 * \
                     (avail["fetch_lon_max"] - avail["fetch_lon_min"]) * 85e3 / 1e6
    avail = avail.sort_values("est_mp")
    pick = []
    if len(avail) >= n:
        # span low/mid/high
        idxs = np.linspace(0, len(avail) - 1, n).astype(int)
        pick = avail.iloc[idxs].cluster_id.tolist()
    else:
        pick = avail.cluster_id.tolist()
    return pick


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cluster_ids", nargs="*", help="cluster_ids to render")
    ap.add_argument("--auto", type=int, default=0,
                    help="auto-pick N clusters spanning small/medium/large mosaics")
    ap.add_argument("--manifest", default=str(MANIFEST_PATH))
    args = ap.parse_args()

    manifest = pd.read_parquet(args.manifest)
    manifest = manifest.set_index("cluster_id", drop=False)

    cluster_ids = list(args.cluster_ids)
    if args.auto > 0:
        cluster_ids += pick_auto_clusters(manifest, args.auto)
    if not cluster_ids:
        ap.error("provide cluster_ids or --auto N")

    for cid in cluster_ids:
        if cid not in manifest.index:
            print(f"[render] {cid}: not in manifest, skipping")
            continue
        row = manifest.loc[cid]
        try:
            masks = fetch_masks(cid)
        except Exception as e:
            print(f"[render] {cid}: fetch_masks failed: {e!r}")
            continue
        img = render_cluster(row, masks)
        if img is None:
            continue
        out_path = OUT_DIR / f"{cid}.png"
        img.save(out_path)
        print(f"[render] {cid} -> {out_path} ({len(masks)} masks, "
              f"{img.size[0]}x{img.size[1]}px)")


if __name__ == "__main__":
    main()
