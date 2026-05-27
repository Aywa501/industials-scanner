"""Generate a CONUS-wide tile grid for the Sentinel-2 inference scan.

For each grid cell we emit:
  - tile_id          stable string id
  - mgrs_tile        Sentinel-2 100km MGRS tile (e.g. "10TGS")
  - lon, lat         centroid in EPSG:4326
  - x_5070, y_5070   centroid in CONUS Albers (EPSG:5070)

Tile geometry: 224 px at S2 10m → 2240 m. Stride 1680 m → 25% overlap.
Cells are kept iff the centroid falls inside the CONUS (lower-48 + DC) land
polygon, rasterized at the grid resolution.

Run locally once:
    cd sites_us
    python -m phase3_scan.make_grid

Output: data_us/phase3_scan/phase3_grid.parquet
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import mgrs
import numpy as np
import pandas as pd
import pyproj
import rasterio
from dotenv import load_dotenv
from rasterio.features import rasterize
from shapely.geometry import shape
from shapely.ops import transform as shp_transform
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
DATA_US = ROOT.parent / "data_us"
GRID_PATH = DATA_US / "phase3_scan" / "phase3_grid.parquet"
CACHE_DIR = DATA_US / "phase3_scan" / "cache"

TILE_SIZE_M = 2240.0      # 224 px × 10 m/px
STRIDE_M = 1680.0         # 25% overlap
ALBERS_BOUNDS = (-2_400_000.0, 200_000.0, 2_300_000.0, 3_200_000.0)

STATES_GEOJSON_URL = (
    "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/"
    "data/geojson/us-states.json"
)
NON_CONUS = {"Alaska", "Hawaii", "Puerto Rico"}


def _load_conus_polygon():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = CACHE_DIR / "us-states.geojson"
    if not cached.exists():
        print(f"[grid] downloading {STATES_GEOJSON_URL}")
        with urllib.request.urlopen(STATES_GEOJSON_URL) as r:
            cached.write_bytes(r.read())
    fc = json.loads(cached.read_text())
    geoms = [
        shape(f["geometry"]) for f in fc["features"]
        if f["properties"].get("name") not in NON_CONUS
    ]
    return unary_union(geoms)


def _build_mask(conus_5070, xmin, ymin, xmax, ymax):
    width = int(round((xmax - xmin) / STRIDE_M))
    height = int(round((ymax - ymin) / STRIDE_M))
    transform = rasterio.transform.from_origin(xmin, ymax, STRIDE_M, STRIDE_M)
    print(f"[grid] rasterizing CONUS into {width} × {height} grid")
    mask = rasterize(
        [(conus_5070, 1)],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype="uint8",
    )
    return mask, transform


def _build_grid() -> pd.DataFrame:
    conus_4326 = _load_conus_polygon()
    to_5070 = pyproj.Transformer.from_crs(4326, 5070, always_xy=True).transform
    to_4326 = pyproj.Transformer.from_crs(5070, 4326, always_xy=True).transform
    conus_5070 = shp_transform(to_5070, conus_4326)

    xmin, ymin, xmax, ymax = ALBERS_BOUNDS
    mask, _ = _build_mask(conus_5070, xmin, ymin, xmax, ymax)
    rows, cols = np.nonzero(mask)
    print(f"[grid] CONUS-filtered cells: {len(rows):,}")

    x_5070 = xmin + (cols + 0.5) * STRIDE_M
    y_5070 = ymax - (rows + 0.5) * STRIDE_M

    lon, lat = to_4326(x_5070, y_5070)

    print("[grid] computing MGRS tile assignment")
    m = mgrs.MGRS()
    mgrs_tiles = np.empty(len(lon), dtype=object)
    for i, (la, lo) in enumerate(zip(lat, lon)):
        if i and i % 200_000 == 0:
            print(f"[grid]   mgrs {i:,}/{len(lon):,}")
        # toMGRS returns bytes on some builds; decode defensively.
        s = m.toMGRS(la, lo, MGRSPrecision=0)
        if isinstance(s, bytes):
            s = s.decode()
        mgrs_tiles[i] = s

    return pd.DataFrame({
        "tile_id": [f"t_{i:08d}" for i in range(len(lon))],
        "mgrs_tile": mgrs_tiles,
        "lon": np.round(lon, 6),
        "lat": np.round(lat, 6),
        "x_5070": x_5070.astype(np.int64),
        "y_5070": y_5070.astype(np.int64),
    })


def main() -> None:
    df = _build_grid()
    DATA_US.mkdir(parents=True, exist_ok=True)
    df.to_parquet(GRID_PATH, index=False)
    print(f"[grid] wrote {len(df):,} tiles → {GRID_PATH}")
    print(f"[grid] {df.mgrs_tile.nunique()} MGRS tiles; top 10 by tile count:")
    print(df.mgrs_tile.value_counts().head(10).to_string())


if __name__ == "__main__":
    main()
