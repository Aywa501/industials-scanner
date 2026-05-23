"""Overlay surviving Overture building bboxes onto NAIP inspect PNGs.

Re-fetches the NAIP mosaic for each requested cluster_id (so the EPSG:5070
transform is in hand), looks up the cluster's member buildings in
data_us/phase3_naip/cluster_buildings.parquet, and draws each building bbox
in green on a copy of the inspect image.

Usage:
  python -m phase3_naip.overlay_inspect c_0000000_c1 c_0000000_c23
  python -m phase3_naip.overlay_inspect --from-inspect-dir
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import pyproj
from dotenv import load_dotenv
from PIL import Image, ImageDraw

SITES_US = Path(__file__).resolve().parents[1]
load_dotenv(SITES_US / ".env")
DATA_US = SITES_US.parent / "data_us"

from phase3_naip.naip_sam import (  # noqa: E402
    LOCAL_TMP_RGB, _rasterio_env, read_naip_mosaic, to_uint8_rgb,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cluster_ids", nargs="*")
    ap.add_argument("--from-inspect-dir", action="store_true",
                    help="overlay every PNG in --inspect-dir (excluding existing overlays)")
    ap.add_argument("--inspect-dir", type=str, default=str(LOCAL_TMP_RGB / "inspect"))
    args = ap.parse_args()

    inspect_dir = Path(args.inspect_dir)
    if args.from_inspect_dir:
        cluster_ids = sorted(f.stem for f in inspect_dir.glob("*.png")
                             if not f.stem.endswith("_overlay"))
    else:
        cluster_ids = args.cluster_ids
    if not cluster_ids:
        raise SystemExit("provide cluster_ids or use --from-inspect-dir")

    mfst = pd.read_parquet(DATA_US / "phase3_naip" / "naip_manifest.parquet")
    cb = pd.read_parquet(DATA_US / "phase3_naip" / "cluster_buildings.parquet")
    to_5070 = pyproj.Transformer.from_crs(4326, 5070, always_xy=True).transform

    n_ok = n_fail = 0
    with _rasterio_env():
        for cid in cluster_ids:
            row = mfst[mfst.cluster_id == cid]
            if row.empty:
                print(f"[overlay] {cid}: not in manifest, skipping")
                n_fail += 1
                continue
            r = row.iloc[0]
            print(f"[overlay] {cid}: fetching NAIP...", flush=True)
            mosaic = read_naip_mosaic(list(r.naip_uris),
                                      float(r.fetch_lon_min), float(r.fetch_lat_min),
                                      float(r.fetch_lon_max), float(r.fetch_lat_max))
            if mosaic is None:
                print(f"[overlay] {cid}: NAIP read failed")
                n_fail += 1
                continue
            arr, tr = mosaic
            rgb = to_uint8_rgb(arr)

            sub_b = cb[cb.cluster_id == cid]
            print(f"[overlay] {cid}: {len(sub_b)} buildings", flush=True)

            img = Image.fromarray(rgb).convert("RGB")
            draw = ImageDraw.Draw(img)
            inv = ~tr
            for _, b in sub_b.iterrows():
                xs5, ys5 = to_5070([b.xmin, b.xmax, b.xmax, b.xmin],
                                   [b.ymin, b.ymin, b.ymax, b.ymax])
                pxs, pys = [], []
                for x, y in zip(xs5, ys5):
                    px, py = inv * (x, y)
                    pxs.append(px)
                    pys.append(py)
                bbox = (min(pxs), min(pys), max(pxs), max(pys))
                draw.rectangle(bbox, outline=(0, 255, 0), width=3)

            out = inspect_dir / f"{cid}_overlay.png"
            img.save(out)
            print(f"[overlay] {cid}: wrote {out}", flush=True)
            n_ok += 1

    print(f"\n[overlay] {n_ok} ok / {n_fail} failed")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
