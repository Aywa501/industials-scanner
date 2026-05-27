"""Match the 316 announcement set to Overture polygons in our 344K candidate cohort.

Filter:
  - site_type = greenfield   (only sites that didn't exist in 2008)
  - status_current = operational
  - lat dp >= 5  (≤ ~1m coord precision)

Match:
  - BallTree haversine over the 344K stage2_candidates centroids
  - For each kept announcement, find polygons within 100m / 500m / 2km
  - Quality:
      strong  : 1-3 polygons within 100m
      multi   : >3 within 100m  (city-block / cluster)
      far     : 0 within 100m, >=1 within 500m  (coord drift)
      none    : 0 within 500m   (not in our cohort)
"""
import os
import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANN = os.path.join(ROOT, "..", "data_us/labels/manufacturing_announcements_geocoded.csv")
CANDS = os.path.join(ROOT, "..", "data_us/phase2/v3/stage2_candidates.parquet")
POLYS = os.path.join(ROOT, "..", "data_us/phase2/v3/stage2_candidate_polygons.parquet")
OUT = os.path.join(ROOT, "..", "data_us/phase2/v3/announcement_polygon_matches.parquet")

EARTH_R_M = 6_371_000.0


def _lat_dp(x):
    if pd.isna(x):
        return -1
    s = f"{x:.10f}".rstrip("0")
    return max(0, len(s.split(".")[1])) if "." in s else 0


def main():
    ann = pd.read_csv(ANN)
    print(f"announcements raw: {len(ann)}")

    ann["lat_dp"] = ann["lat"].apply(_lat_dp)
    keep = (
        (ann["site_type"] == "greenfield")
        & (ann["status_current"] == "operational")
        & (ann["lat_dp"] >= 5)
        & ann["lat"].notna()
        & ann["lng"].notna()
    )
    ann_k = ann[keep].reset_index(drop=True)
    print(f"after filter (greenfield + operational + lat_dp>=5): {len(ann_k)}")

    cands = pd.read_parquet(CANDS, columns=["building_id", "ovt_id", "lat", "lon"])
    print(f"candidate cohort: {len(cands):,}")

    tree = BallTree(np.radians(cands[["lat", "lon"]].values), metric="haversine")

    q = np.radians(ann_k[["lat", "lng"]].values)
    # Query within 2km (largest radius); we'll classify by counts at sub-radii
    idx_2km, dist_2km = tree.query_radius(q, r=2000 / EARTH_R_M, return_distance=True)

    rows = []
    for i, (ann_row, idxs, dists) in enumerate(zip(ann_k.itertuples(index=False), idx_2km, dist_2km)):
        dists_m = dists * EARTH_R_M
        n_100m = int((dists_m <= 100).sum())
        n_500m = int((dists_m <= 500).sum())
        n_2km = len(idxs)

        if n_100m == 0 and n_2km == 0:
            quality = "none"
            bids = []
        elif 1 <= n_100m <= 3:
            quality = "strong"
            keep_idx = idxs[dists_m <= 100]
            bids = cands.iloc[keep_idx]["building_id"].tolist()
        elif n_100m > 3:
            quality = "multi"
            keep_idx = idxs[dists_m <= 100]
            bids = cands.iloc[keep_idx]["building_id"].tolist()
        elif n_500m >= 1:
            quality = "far"
            keep_idx = idxs[dists_m <= 500]
            bids = cands.iloc[keep_idx]["building_id"].tolist()
        else:
            quality = "none"
            bids = []

        rows.append({
            "project": ann_row.canonical_project_name,
            "lat": ann_row.lat,
            "lng": ann_row.lng,
            "lat_dp": ann_row.lat_dp,
            "announcement_date": ann_row.announcement_date,
            "actual_first_production_date": ann_row.actual_first_production_date,
            "state": ann_row.state,
            "n_100m": n_100m,
            "n_500m": n_500m,
            "n_2km": n_2km,
            "quality": quality,
            "matched_building_ids": bids,
            "n_matched": len(bids),
        })

    out = pd.DataFrame(rows)
    print()
    print(f"=== Quality breakdown ===")
    print(out["quality"].value_counts())
    print()
    print(f"=== Usable for calibration (strong + multi): {len(out[out.quality.isin(['strong', 'multi'])]):,} sites, "
          f"{sum(out[out.quality.isin(['strong','multi'])]['n_matched']):,} polygons ===")
    print()
    print(out.head(15)[["project", "state", "lat_dp", "n_100m", "n_500m", "quality", "n_matched"]].to_string(index=False))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
