"""Segment Phase 3 candidate clusters into individual facility points.

Inputs:
  - data_us/phase3_candidates.parquet  (3,028 clusters)
  - data_us/phase3_singletons.parquet  (2,876 isolated p>=0.95 tiles)
  - data_us/overture_industrial_conus.parquet  (CONUS industrial buildings)

Approach:
  1. Compute each cluster's bbox area in km^2.
  2. For clusters with bbox area > BIG_KM2, spatial-join against Overture buildings
     and DBSCAN their centroids (eps_m = FACILITY_EPS_M) to produce one (lon,lat)
     per facility. The cluster is replaced by N facility points.
  3. For clusters with bbox area <= BIG_KM2, keep the cluster centroid as-is
     (already facility-scale).
  4. Singletons are kept as-is (one tile = one candidate point).

Output:
  - data_us/phase3_facilities.parquet
    Columns: facility_id, parent_candidate_id, lon, lat, n_buildings, source
              source in {'cluster_centroid', 'sub_cluster', 'singleton'}
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

ROOT = Path(__file__).parents[3]
CANDIDATES = ROOT / "data_us" / "phase3_candidates.parquet"
SINGLETONS = ROOT / "data_us" / "phase3_singletons.parquet"
OVERTURE = ROOT / "data_us" / "overture_industrial_conus.parquet"
OUT = ROOT / "data_us" / "phase3_facilities.parquet"

BIG_KM2 = 10.0
FACILITY_EPS_M = 300.0


def bbox_area_km2(row):
    lat_mid = (row["lat_min"] + row["lat_max"]) / 2.0
    w_km = (row["lon_max"] - row["lon_min"]) * 111.0 * math.cos(math.radians(lat_mid))
    h_km = (row["lat_max"] - row["lat_min"]) * 111.0
    return max(w_km, 0.0) * max(h_km, 0.0)


def cluster_buildings(buildings: pd.DataFrame) -> pd.DataFrame:
    cols = ["lon", "lat", "n", "names", "n_distinct_names", "max_height_m", "max_num_floors", "top_class"]
    if buildings.empty:
        return pd.DataFrame(columns=cols)
    lat_mid = buildings["lat"].mean()
    mx = 111000.0 * math.cos(math.radians(lat_mid))
    my = 111000.0
    X = np.column_stack([buildings["lon"].values * mx, buildings["lat"].values * my])
    db = DBSCAN(eps=FACILITY_EPS_M).fit(X)
    buildings = buildings.assign(_lbl=db.labels_)
    buildings = buildings[buildings["_lbl"] >= 0]
    if buildings.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    for lbl, grp in buildings.groupby("_lbl"):
        names = sorted({n for n in grp["name"].dropna().tolist() if n})
        cls = grp["class"].dropna()
        rows.append({
            "lon": grp["lon"].mean(),
            "lat": grp["lat"].mean(),
            "n": int(len(grp)),
            "names": names,
            "n_distinct_names": len(names),
            "max_height_m": float(grp["height"].max()) if grp["height"].notna().any() else None,
            "max_num_floors": int(grp["num_floors"].max()) if grp["num_floors"].notna().any() else None,
            "top_class": cls.mode().iloc[0] if not cls.empty else None,
        })
    return pd.DataFrame(rows)


def main() -> None:
    cand = pd.read_parquet(CANDIDATES)
    sing = pd.read_parquet(SINGLETONS)
    cand["bbox_km2"] = cand.apply(bbox_area_km2, axis=1)

    big = cand[cand["bbox_km2"] > BIG_KM2].copy()
    small = cand[cand["bbox_km2"] <= BIG_KM2].copy()
    print(f"[segment] big={len(big)}  small={len(small)}  singletons={len(sing)}")

    print(f"[segment] loading buildings -> {OVERTURE}")
    bldg = pd.read_parquet(OVERTURE, columns=["id", "lon", "lat", "approx_area_m2", "class", "subtype", "name", "height", "num_floors"])
    print(f"[segment] buildings loaded: {len(bldg):,}")

    INDUSTRIAL_CLASSES = {"industrial", "warehouse", "hangar", "farm_auxiliary"}
    keep = (
        bldg["class"].isin(INDUSTRIAL_CLASSES)
        | (bldg["subtype"] == "industrial")
        | (bldg["class"].isna() & (bldg["approx_area_m2"] >= 3000))
    )
    bldg = bldg[keep].reset_index(drop=True)
    print(f"[segment] after class+area filter: {len(bldg):,}")

    facilities = []
    for i, row in enumerate(big.itertuples(index=False), 1):
        sub = bldg[
            (bldg["lon"] >= row.lon_min) & (bldg["lon"] <= row.lon_max)
            & (bldg["lat"] >= row.lat_min) & (bldg["lat"] <= row.lat_max)
        ]
        clusters = cluster_buildings(sub)
        if clusters.empty:
            facilities.append(dict(parent_candidate_id=row.candidate_id, lon=row.lon, lat=row.lat, n_buildings=0, source="cluster_centroid",
                                   names=[], n_distinct_names=0, max_height_m=None, max_num_floors=None, top_class=None))
        else:
            for c in clusters.itertuples(index=False):
                facilities.append(dict(parent_candidate_id=row.candidate_id, lon=c.lon, lat=c.lat, n_buildings=int(c.n), source="sub_cluster",
                                       names=list(c.names), n_distinct_names=int(c.n_distinct_names),
                                       max_height_m=c.max_height_m, max_num_floors=c.max_num_floors, top_class=c.top_class))
        if i % 100 == 0:
            print(f"[segment] big {i}/{len(big)}")

    for row in small.itertuples(index=False):
        facilities.append(dict(parent_candidate_id=row.candidate_id, lon=row.lon, lat=row.lat, n_buildings=0, source="cluster_centroid",
                               names=[], n_distinct_names=0, max_height_m=None, max_num_floors=None, top_class=None))

    for row in sing.itertuples(index=False):
        facilities.append(dict(parent_candidate_id=row.candidate_id, lon=row.lon, lat=row.lat, n_buildings=0, source="singleton",
                               names=[], n_distinct_names=0, max_height_m=None, max_num_floors=None, top_class=None))

    df = pd.DataFrame(facilities)
    df.insert(0, "facility_id", [f"f_{i:07d}" for i in range(len(df))])
    df.to_parquet(OUT, index=False)
    print(f"[segment] wrote {len(df):,} facilities -> {OUT}")
    print(df.groupby("source").size().to_string())


if __name__ == "__main__":
    main()
