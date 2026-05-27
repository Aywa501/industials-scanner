"""Stage 2b final assembly — apply change threshold + 2 km proximity rescue.

Reads:
  - data_us/phase2/v3/stage2_candidates.parquet         (~345K input rows)
  - data_us/phase2/v3/stage2b_change_chunks/*.parquet   (change scores)

Output:
  - data_us/phase2/v3/stage3_candidates.parquet         (final Stage 3 input)

Statuses:
  changed         change >= CHANGE_T          KEEP
  stable          change <  CHANGE_T          DROP unless rescued
  stable_rescued  stable but a CHANGED within 2 km  KEEP
  ambiguous       footprint_pixels < 10 or NaN change  KEEP
  error           change scoring failed       KEEP (conservative)
"""
import os
import glob

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CANDS = os.path.join(ROOT, "..", "data_us/phase2/v3/stage2_candidates.parquet")
CHUNK_DIR = os.path.join(ROOT, "..", "data_us/phase2/v3/stage2b_change_chunks")
OUT = os.path.join(ROOT, "..", "data_us/phase2/v3/stage3_candidates.parquet")

CHANGE_T = float(os.environ.get("STAGE2B_CHANGE_T", "0.10"))
MIN_FP_PX = int(os.environ.get("STAGE2B_MIN_FOOTPRINT_PX", "10"))
PROX_KM = float(os.environ.get("STAGE2B_PROX_KM", "2.0"))
EARTH_R_KM = 6371.0088


def main():
    cands = pd.read_parquet(CANDS)
    chunks = sorted(glob.glob(os.path.join(CHUNK_DIR, "chunk_*.parquet")))
    if not chunks:
        raise SystemExit(f"no chunks in {CHUNK_DIR}")
    print(f"reading {len(chunks)} chunks")
    chg = pd.concat([pd.read_parquet(c) for c in chunks], ignore_index=True)
    print(f"change rows: {len(chg):,}   candidate rows: {len(cands):,}")

    chg = chg.drop_duplicates(subset=["building_id"], keep="last")
    df = cands.merge(
        chg[["building_id", "ratio_2008", "ratio_2022", "change", "footprint_pixels",
             "n_scenes_2008", "n_scenes_2022", "error"]] if "error" in chg.columns
        else chg[["building_id", "ratio_2008", "ratio_2022", "change", "footprint_pixels",
                  "n_scenes_2008", "n_scenes_2022"]],
        on="building_id", how="left",
    )
    missing = df["change"].isna().sum()
    print(f"missing change score: {missing:,}  ({missing/len(df):.1%})")

    # Status assignment.
    df["change_status"] = "stable"
    has_err = (df.get("error").notna() if "error" in df.columns else pd.Series(False, index=df.index))
    df.loc[has_err, "change_status"] = "error"

    small_fp = df["footprint_pixels"].fillna(0) < MIN_FP_PX
    df.loc[small_fp & ~has_err, "change_status"] = "ambiguous"

    no_chg = df["change"].isna() & ~has_err & ~small_fp
    df.loc[no_chg, "change_status"] = "ambiguous"

    chg_mask = df["change"].fillna(-1) >= CHANGE_T
    df.loc[chg_mask, "change_status"] = "changed"

    # Proximity rescue.
    changed = df[df["change_status"] == "changed"]
    stable = df[df["change_status"] == "stable"]
    print(f"\npre-rescue:  changed={len(changed):,}  stable={len(stable):,}  "
          f"ambiguous={(df['change_status'] == 'ambiguous').sum():,}  "
          f"error={(df['change_status'] == 'error').sum():,}")

    if len(changed) and len(stable):
        chg_xy = np.deg2rad(changed[["lat", "lon"]].values)
        stb_xy = np.deg2rad(stable[["lat", "lon"]].values)
        tree = BallTree(chg_xy, metric="haversine")
        radius_rad = PROX_KM / EARTH_R_KM
        rescued_mask = tree.query_radius(stb_xy, r=radius_rad, count_only=True) > 0
        rescued_ids = stable.loc[rescued_mask, "building_id"].values
        df.loc[df["building_id"].isin(rescued_ids), "change_status"] = "stable_rescued"
        print(f"rescued via {PROX_KM} km proximity: {len(rescued_ids):,}")

    # Final keep set.
    keep_statuses = {"changed", "stable_rescued", "ambiguous", "error"}
    final = df[df["change_status"].isin(keep_statuses)].copy()
    print(f"\nstatus breakdown (final):")
    print(df["change_status"].value_counts().to_string())
    print(f"\nfinal kept: {len(final):,} / {len(df):,}  ({len(final)/len(df):.1%})")

    final.to_parquet(OUT, index=False)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
