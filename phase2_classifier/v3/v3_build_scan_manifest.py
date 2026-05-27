"""Build the CONUS scan manifest: every Overture building >= 3000 m² in CONUS.

Output schema matches v3_dataset_manifest so v3_build_scenes_index runs unmodified
and v3_scan_infer reuses the same fetch+embed path.

class_id stays 0 throughout (unknown — that's what the scan determines). The
inference script ignores class_id and writes p_industrial per row.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DATA_US = ROOT / "data_us"

OVERTURE = DATA_US / "external" / "overture_industrial_conus_2025_aligned.parquet"
CONUS_GEOJSON = DATA_US / "phase3_scan" / "cache" / "us-states.geojson"
OUT = DATA_US / "phase2" / "v3_scan_manifest.parquet"

NON_CONUS_STATES = {"Alaska", "Hawaii", "Puerto Rico"}
AREA_FLOOR_M2 = 5000.0


def short_id(prefix: str, *parts) -> str:
    h = hashlib.md5("|".join(str(p) for p in parts).encode()).hexdigest()[:10]
    return f"{prefix}_{h}"


def _load_conus_polygon():
    import json
    from shapely.geometry import shape
    from shapely.ops import unary_union
    fc = json.loads(CONUS_GEOJSON.read_text())
    geoms = [shape(f["geometry"]) for f in fc["features"]
             if f["properties"].get("name") not in NON_CONUS_STATES]
    return unary_union(geoms)


def main() -> None:
    print(f"[scan-manifest] loading Overture ({OVERTURE.name})...")
    bldg = pd.read_parquet(
        OVERTURE,
        columns=["id", "lon", "lat", "xmin", "xmax", "ymin", "ymax",
                 "approx_area_m2", "class", "subtype", "name"],
    )
    bldg = bldg.rename(columns={"class": "ovt_class"})
    bldg = bldg[bldg["approx_area_m2"] >= AREA_FLOOR_M2].reset_index(drop=True)
    print(f"  >= {AREA_FLOOR_M2:.0f} m^2: {len(bldg):,}")

    print("[scan-manifest] CONUS land filter...")
    from shapely import STRtree
    from shapely.geometry import Point
    conus = _load_conus_polygon()
    pts = [Point(lo, la) for la, lo in zip(bldg["lat"].to_numpy(), bldg["lon"].to_numpy())]
    inside_ix = STRtree([conus]).query(pts, predicate="intersects")[0]
    keep = np.zeros(len(bldg), dtype=bool); keep[inside_ix] = True
    bldg = bldg[keep].reset_index(drop=True)
    print(f"  after CONUS: {len(bldg):,}")

    df = pd.DataFrame({
        "building_id": [short_id("scan", r) for r in bldg["id"].astype(str)],
        "ovt_id": bldg["id"].astype(str),
        "class_id": 0,                       # unknown — that's what we're scanning
        "source": "scan",
        "weight": 1.0,
        "lat": bldg["lat"].astype(float),
        "lon": bldg["lon"].astype(float),
        "xmin": bldg["xmin"].astype(float),
        "xmax": bldg["xmax"].astype(float),
        "ymin": bldg["ymin"].astype(float),
        "ymax": bldg["ymax"].astype(float),
        "approx_area_m2": bldg["approx_area_m2"].astype(float),
        "ovt_class": bldg["ovt_class"],
        "ovt_subtype": bldg["subtype"],
        "ovt_name": bldg["name"],
        "site_id": None,
        "split": "scan",
    })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"[scan-manifest] wrote {len(df):,} rows -> {OUT}")
    print(f"  ovt_class top-10:")
    print(df["ovt_class"].value_counts(dropna=False).head(10).to_string())


if __name__ == "__main__":
    main()
