"""Joint coverage check: how does the current pipeline output (14,141 facilities,
segmented via Overture industrial filter) cover the 276 site-precise positives,
and how does it decompose against Overture-only and sat-detector-only?

For each positive, classify into:
  A. found_by_pipeline: a phase3_facilities.parquet row within 1km
  B. found_by_overture_only: an Overture industrial building within 500m AND no pipeline facility within 1km
  C. found_by_sat_only: a pipeline cluster_centroid/singleton/sub_cluster nearby BUT no Overture
     industrial building within 500m
  D. missed_by_both
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

ROOT = Path(__file__).parents[3]

ann = pd.read_csv(ROOT / "data_us" / "manufacturing_announcements_geocoded.csv")

def decimals(x):
    s = f"{x:.10f}".rstrip("0").rstrip(".")
    return len(s.split(".")[1]) if "." in s else 0

ann = ann.dropna(subset=["lat", "lng"]).copy()
ann["dec_lat"] = ann["lat"].apply(decimals)
ann["dec_lng"] = ann["lng"].apply(decimals)
pos = ann[(ann["dec_lat"] >= 4) & (ann["dec_lng"] >= 4)].reset_index(drop=True)

fac = pd.read_parquet(ROOT / "data_us" / "phase3_facilities.parquet")

bldg = pd.read_parquet(ROOT / "data_us" / "overture_industrial_conus.parquet",
                       columns=["lon", "lat", "class", "subtype"])
INDUSTRIAL = {"industrial", "warehouse", "hangar", "farm_auxiliary"}
bldg_ind = bldg[bldg["class"].isin(INDUSTRIAL) | (bldg["subtype"] == "industrial")].reset_index(drop=True)

q = np.radians(pos[["lat", "lng"]].values)

# nearest pipeline facility
tree_fac = BallTree(np.radians(fac[["lat", "lon"]].values), metric="haversine")
d_fac, i_fac = tree_fac.query(q, k=1)
pos["nearest_fac_m"] = d_fac[:, 0] * 6371000.0
pos["nearest_fac_source"] = fac["source"].iloc[i_fac[:, 0]].values

# nearest Overture industrial building
tree_bld = BallTree(np.radians(bldg_ind[["lat", "lon"]].values), metric="haversine")
d_bld, _ = tree_bld.query(q, k=1)
pos["nearest_overture_m"] = d_bld[:, 0] * 6371000.0

PIPE_R = 1000.0
OVRT_R = 500.0
pos["found_by_pipeline"] = pos["nearest_fac_m"] <= PIPE_R
pos["found_by_overture"] = pos["nearest_overture_m"] <= OVRT_R

def classify(r):
    if r["found_by_pipeline"] and r["found_by_overture"]:
        return "A_both"
    if r["found_by_pipeline"] and not r["found_by_overture"]:
        return "A_pipeline_only_sat_credit"  # sat saw it, Overture didn't tag a building near it
    if not r["found_by_pipeline"] and r["found_by_overture"]:
        return "B_overture_only_sat_missed"
    return "D_missed_by_both"

pos["bucket"] = pos.apply(classify, axis=1)

print(f"positives: {len(pos)}")
print()
print(f"pipeline coverage @ {PIPE_R:.0f}m: {pos['found_by_pipeline'].sum()}/{len(pos)} ({pos['found_by_pipeline'].mean()*100:.1f}%)")
print(f"overture coverage @ {OVRT_R:.0f}m:  {pos['found_by_overture'].sum()}/{len(pos)} ({pos['found_by_overture'].mean()*100:.1f}%)")
print()
print("Joint decomposition:")
print(pos["bucket"].value_counts().to_string())
print()
print("Pipeline coverage by source (when found):")
print(pos[pos['found_by_pipeline']]['nearest_fac_source'].value_counts().to_string())
print()
print("Sat-credit (pipeline found, Overture didn't): sample of 10")
sat_credit = pos[pos["bucket"] == "A_pipeline_only_sat_credit"]
cols = ["canonical_project_name", "parent_company", "city", "state", "site_type", "status_current", "nearest_fac_m", "nearest_overture_m"]
print(sat_credit[cols].head(10).to_string())
print()
print("Overture-only (pipeline missed it): sample of 10")
ovrt_only = pos[pos["bucket"] == "B_overture_only_sat_missed"]
print(ovrt_only[cols].head(10).to_string())
print()
print("Missed by both: sample of 10")
print(pos[pos["bucket"] == "D_missed_by_both"][cols].head(10).to_string())

pos.to_csv(ROOT / "data_us" / "phase3_joint_coverage.csv", index=False)
print()
print(f"wrote -> {ROOT / 'data_us' / 'phase3_joint_coverage.csv'}")
