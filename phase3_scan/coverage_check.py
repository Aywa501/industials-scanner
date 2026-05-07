"""Tie-breaker measurement: do the 316 known announcement positives have a
matching Overture industrial-class building nearby?

If yes (>=80% covered), Overture-first is a viable candidate-source strategy
and the sat detector should be repositioned to greenfield-only.
If no (<50%), the sat detector earns its keep across the board.
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

ROOT = Path(__file__).parents[2]
ANNOUNCEMENTS = ROOT / "data_us" / "manufacturing_announcements_geocoded.csv"
OVERTURE = ROOT / "data_us" / "overture_industrial_conus.parquet"

INDUSTRIAL_CLASSES = {"industrial", "warehouse", "hangar", "farm_auxiliary"}
RADII_M = [200, 500, 1000, 2000]


def main() -> None:
    ann = pd.read_csv(ANNOUNCEMENTS)
    ann_geo = ann.dropna(subset=["lat", "lng"]).copy()

    def decimals(x: float) -> int:
        s = f"{x:.10f}".rstrip("0").rstrip(".")
        return len(s.split(".")[1]) if "." in s else 0

    ann_geo["dec_lat"] = ann_geo["lat"].apply(decimals)
    ann_geo["dec_lng"] = ann_geo["lng"].apply(decimals)
    site_precise = ann_geo[(ann_geo["dec_lat"] >= 4) & (ann_geo["dec_lng"] >= 4)].copy()
    print(f"announcements total={len(ann)}  with-coords={len(ann_geo)}  site-precise(>=4 dec)={len(site_precise)}")

    bldg = pd.read_parquet(OVERTURE, columns=["lon", "lat", "class", "subtype", "name", "approx_area_m2"])
    keep = bldg["class"].isin(INDUSTRIAL_CLASSES) | (bldg["subtype"] == "industrial")
    ind = bldg[keep].reset_index(drop=True)
    print(f"overture industrial buildings={len(ind):,}  named={ind['name'].notna().sum():,}")

    tree = BallTree(np.radians(ind[["lat", "lon"]].values), metric="haversine")
    q = np.radians(site_precise[["lat", "lng"]].values)
    dist_rad, idx = tree.query(q, k=1)
    nearest_m = dist_rad[:, 0] * 6371000.0
    site_precise["nearest_industrial_m"] = nearest_m

    nearest_idx = idx[:, 0]
    site_precise["nearest_industrial_name"] = ind["name"].iloc[nearest_idx].values
    site_precise["nearest_industrial_class"] = ind["class"].iloc[nearest_idx].values
    site_precise["nearest_industrial_area_m2"] = ind["approx_area_m2"].iloc[nearest_idx].values

    print()
    print("Coverage at radius:")
    for r in RADII_M:
        n_within = (nearest_m <= r).sum()
        pct = n_within / len(site_precise) * 100
        print(f"  <={r}m: {n_within}/{len(site_precise)} ({pct:.1f}%)")

    print()
    print("Distance distribution (m):")
    for q in [0.1, 0.25, 0.5, 0.75, 0.9, 0.95]:
        print(f"  {int(q*100)}th pct: {np.quantile(nearest_m, q):.0f}")

    site_precise[["canonical_project_name", "parent_company", "city", "state", "lat", "lng",
                  "nearest_industrial_m", "nearest_industrial_name", "nearest_industrial_class", "nearest_industrial_area_m2"]] \
        .sort_values("nearest_industrial_m") \
        .to_csv(ROOT / "data_us" / "phase3_overture_coverage.csv", index=False)
    print()
    print(f"wrote -> {ROOT / 'data_us' / 'phase3_overture_coverage.csv'}")


if __name__ == "__main__":
    main()
