"""Dump actual NAIP crops as PNGs for the sample_bands.csv buildings.

Saves to data_us/phase2/v3/scan_validate/crops/<band>/<idx>_<p_mean>_<bid>.png
so they sort by band → confidence → id.
"""

from __future__ import annotations
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
ENV = ROOT / "sites_us" / ".env"
if ENV.exists():
    for line in ENV.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from rasterio.windows import Window, from_bounds
from rasterio.warp import transform_bounds

from sites_us.phase2_classifier.v3.v3_train import setup_rasterio_env

DATA = ROOT / "data_us"
SAMPLE = DATA / "phase2" / "v3" / "scan_validate" / "sample_bands.csv"
SCENES = DATA / "phase2" / "v3_scan_scenes_index.parquet"
OUT_ROOT = DATA / "phase2" / "v3" / "scan_validate" / "crops"


def fetch_crop(uri: str, fxmin: float, fymin: float, fxmax: float, fymax: float) -> np.ndarray | None:
    """RGB crop = building bbox + buffer, as (H, W, 3) uint8."""
    try:
        with rasterio.open(uri) as src:
            if src.crs is None:
                return None
            xmin, ymin, xmax, ymax = transform_bounds(
                "EPSG:4326", src.crs, fxmin, fymin, fxmax, fymax, densify_pts=21)
            win = from_bounds(xmin, ymin, xmax, ymax, transform=src.transform)
            col = max(0, int(round(win.col_off)))
            row = max(0, int(round(win.row_off)))
            col_end = min(src.width, int(round(win.col_off + win.width)))
            row_end = min(src.height, int(round(win.row_off + win.height)))
            if col >= col_end or row >= row_end:
                return None
            arr = src.read([1, 2, 3], window=Window(col, row, col_end - col, row_end - row))
            # Percentile-stretch to 0-255 for viewable PNG (NAIP raw can be muddy).
            out = np.empty_like(arr)
            for c in range(3):
                lo, hi = np.percentile(arr[c], [1, 99])
                if hi - lo < 1.0:
                    out[c] = arr[c]
                else:
                    out[c] = np.clip(((arr[c].astype(np.float32) - lo) / max(hi - lo, 1e-6)) * 255,
                                      0, 255).astype(np.uint8)
            return out.transpose(1, 2, 0)
    except Exception as e:
        print(f"  fail: {e!r}")
        return None


def main() -> None:
    s = pd.read_csv(SAMPLE)
    scenes = pd.read_parquet(SCENES, columns=["building_id", "naip_uris",
                                              "fetch_xmin", "fetch_ymin",
                                              "fetch_xmax", "fetch_ymax"])
    s = s.merge(scenes, on="building_id", how="left")
    print(f"{len(s)} buildings to fetch")

    def one(i_row):
        i, r = i_row
        uris = r["naip_uris"]
        if uris is None or len(uris) == 0:
            return None
        out_dir = OUT_ROOT / r["band"]
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{i:02d}_p{r['p_mean']:.2f}_{r['building_id']}.png"
        out_path = out_dir / fname
        if out_path.exists():
            return out_path
        arr = fetch_crop(uris[0], r["fetch_xmin"], r["fetch_ymin"],
                         r["fetch_xmax"], r["fetch_ymax"])
        if arr is None:
            return None
        Image.fromarray(arr).save(out_path)
        return out_path

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    with setup_rasterio_env():
        with ThreadPoolExecutor(max_workers=16) as ex:
            for p in ex.map(one, list(enumerate(s.to_dict("records")))):
                if p is not None:
                    n_ok += 1
                    if n_ok % 10 == 0:
                        print(f"  {n_ok}/{len(s)} done")
    print(f"done: {n_ok}/{len(s)} crops -> {OUT_ROOT}")


if __name__ == "__main__":
    main()
