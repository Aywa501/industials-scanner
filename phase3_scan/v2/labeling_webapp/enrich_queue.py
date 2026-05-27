"""Enrich queue.json with ground-truth context for each candidate.

For each (lat, lon) candidate, attach:
  - nearest_anchor: distance/name/sector/parent from the 316-site CSV (within 5km)
  - nearest_named_building: nearest Overture building with a name (within 500m)
  - industrial_buildings_500m: count of industrial-class Overture buildings within 500m
  - state: from nearest anchor or inferred

Run after prep_candidates.py. Modifies queue.json in place.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = ROOT / ".artifacts" / "labeling_v2"
QUEUE_PATH = ARTIFACTS / "queue.json"
DATA_US = ROOT.parent / "data_us"
OVERTURE_PATH = DATA_US / "external" / "overture_industrial_conus_2025_aligned.parquet"
ANCHORS_CSV = DATA_US / "labels" / "manufacturing_announcements_geocoded.csv"

EARTH_M = 6_371_000.0
INDUSTRIAL_CLASSES = {
    "industrial", "warehouse", "manufacture", "factory", "hangar",
    "silo", "storage_tank", "works", "greenhouse",
}
ANCHOR_RADIUS_M = 5000.0
NAMED_RADIUS_M = 500.0
INDUSTRIAL_RADIUS_M = 500.0
# For map pins, we want every nearby POI inside the wide chip (~2.6 km half-extent
# corner-to-corner). 3000m gives a small buffer.
PIN_RADIUS_M = 3000.0
PIN_MAX = 15


def to_rad(latlon: np.ndarray) -> np.ndarray:
    return np.deg2rad(latlon)


def main() -> int:
    queue = json.loads(QUEUE_PATH.read_text())
    n = len(queue)
    print(f"loaded queue: {n} candidates")

    cand_rad = to_rad(np.array([[c["lat"], c["lon"]] for c in queue]))

    print("loading anchors")
    anchors = pd.read_csv(ANCHORS_CSV)
    anchors = anchors.dropna(subset=["lat", "lng"]).reset_index(drop=True)
    a_rad = to_rad(anchors[["lat", "lng"]].to_numpy())
    a_tree = BallTree(a_rad, metric="haversine")
    a_dist, a_idx = a_tree.query(cand_rad, k=1)
    a_dist_m = a_dist[:, 0] * EARTH_M

    print(f"loading overture ({OVERTURE_PATH.name})")
    ov = pd.read_parquet(OVERTURE_PATH, columns=["lon", "lat", "class", "subtype", "name", "approx_area_m2"])

    print("  building industrial subset")
    ind_mask = ov["class"].isin(INDUSTRIAL_CLASSES) | (ov.subtype == "industrial")
    ind = ov.loc[ind_mask, ["lat", "lon", "class", "name", "approx_area_m2"]].reset_index(drop=True)
    i_rad = to_rad(ind[["lat", "lon"]].to_numpy())
    print(f"  industrial buildings: {len(ind):,}")
    i_tree = BallTree(i_rad, metric="haversine")
    # Count within 500m (radius / earth radius in radians)
    counts = i_tree.query_radius(cand_rad, r=INDUSTRIAL_RADIUS_M / EARTH_M, count_only=True)
    i_dist, i_idx = i_tree.query(cand_rad, k=1)
    i_dist_m = i_dist[:, 0] * EARTH_M

    print("  building named subset")
    named = ov.loc[ov.name.notna(), ["lat", "lon", "class", "subtype", "name", "approx_area_m2"]].reset_index(drop=True)
    n_rad = to_rad(named[["lat", "lon"]].to_numpy())
    print(f"  named buildings: {len(named):,}")
    n_tree = BallTree(n_rad, metric="haversine")
    nm_dist, nm_idx = n_tree.query(cand_rad, k=1)
    nm_dist_m = nm_dist[:, 0] * EARTH_M

    # All-nearby pins within PIN_RADIUS_M for both industrial and named subsets.
    print(f"  finding ≤{PIN_MAX} pins within {PIN_RADIUS_M:.0f}m per candidate")
    i_pin_idx, i_pin_dist = i_tree.query_radius(
        cand_rad, r=PIN_RADIUS_M / EARTH_M, return_distance=True, sort_results=True,
    )
    n_pin_idx, n_pin_dist = n_tree.query_radius(
        cand_rad, r=PIN_RADIUS_M / EARTH_M, return_distance=True, sort_results=True,
    )
    # Anchors within PIN_RADIUS too
    a_pin_idx, a_pin_dist = a_tree.query_radius(
        cand_rad, r=PIN_RADIUS_M / EARTH_M, return_distance=True, sort_results=True,
    )

    print("enriching queue entries")
    for i, c in enumerate(queue):
        # Nearest anchor
        if a_dist_m[i] <= ANCHOR_RADIUS_M:
            row = anchors.iloc[a_idx[i, 0]]
            c["nearest_anchor"] = {
                "name": str(row.get("canonical_project_name") or "")[:120] or None,
                "company": str(row.get("parent_company") or "")[:80] or None,
                "sector": str(row.get("sector") or "")[:60] or None,
                "site_type": str(row.get("site_type") or "")[:30] or None,
                "state": str(row.get("state") or "")[:20] or None,
                "distance_m": int(round(a_dist_m[i])),
                "lat": float(row["lat"]),
                "lon": float(row["lng"]),
            }
            c["state"] = c["nearest_anchor"]["state"]
        else:
            c["nearest_anchor"] = None
        # Nearest industrial Overture building
        if i_dist_m[i] <= NAMED_RADIUS_M:
            row = ind.iloc[i_idx[i, 0]]
            c["nearest_industrial"] = {
                "class": str(row["class"]) if pd.notna(row["class"]) else None,
                "name": str(row["name"]) if pd.notna(row["name"]) else None,
                "area_m2": int(row["approx_area_m2"]) if pd.notna(row["approx_area_m2"]) else None,
                "distance_m": int(round(i_dist_m[i])),
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
            }
        else:
            c["nearest_industrial"] = None
        # Industrial count
        c["industrial_buildings_500m"] = int(counts[i])
        # Nearest named building
        if nm_dist_m[i] <= NAMED_RADIUS_M:
            row = named.iloc[nm_idx[i, 0]]
            c["nearest_named_building"] = {
                "name": str(row["name"]),
                "class": str(row["class"]) if pd.notna(row["class"]) else None,
                "subtype": str(row["subtype"]) if pd.notna(row["subtype"]) else None,
                "area_m2": int(row["approx_area_m2"]) if pd.notna(row["approx_area_m2"]) else None,
                "distance_m": int(round(nm_dist_m[i])),
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
            }
        else:
            c["nearest_named_building"] = None

        # Pins for in-chip markers (capped, sorted nearest-first)
        i_p = i_pin_idx[i][:PIN_MAX]
        i_d = i_pin_dist[i][:PIN_MAX] * EARTH_M
        c["industrial_pins"] = [
            {"lat": float(ind.iloc[j]["lat"]), "lon": float(ind.iloc[j]["lon"]),
             "class": (str(ind.iloc[j]["class"]) if pd.notna(ind.iloc[j]["class"]) else None),
             "name": (str(ind.iloc[j]["name"]) if pd.notna(ind.iloc[j]["name"]) else None),
             "area_m2": (int(ind.iloc[j]["approx_area_m2"]) if pd.notna(ind.iloc[j]["approx_area_m2"]) else None),
             "distance_m": int(round(d))}
            for j, d in zip(i_p, i_d)
        ]
        n_p = n_pin_idx[i][:PIN_MAX]
        n_d = n_pin_dist[i][:PIN_MAX] * EARTH_M
        c["named_pins"] = [
            {"lat": float(named.iloc[j]["lat"]), "lon": float(named.iloc[j]["lon"]),
             "name": str(named.iloc[j]["name"]),
             "class": (str(named.iloc[j]["class"]) if pd.notna(named.iloc[j]["class"]) else None),
             "subtype": (str(named.iloc[j]["subtype"]) if pd.notna(named.iloc[j]["subtype"]) else None),
             "distance_m": int(round(d))}
            for j, d in zip(n_p, n_d)
        ]
        a_p = a_pin_idx[i][:PIN_MAX]
        a_d = a_pin_dist[i][:PIN_MAX] * EARTH_M
        c["anchor_pins"] = [
            {"lat": float(anchors.iloc[j]["lat"]), "lon": float(anchors.iloc[j]["lng"]),
             "name": str(anchors.iloc[j].get("canonical_project_name") or "")[:120] or None,
             "company": str(anchors.iloc[j].get("parent_company") or "")[:80] or None,
             "sector": str(anchors.iloc[j].get("sector") or "")[:60] or None,
             "distance_m": int(round(d))}
            for j, d in zip(a_p, a_d)
        ]

    QUEUE_PATH.write_text(json.dumps(queue, indent=2))
    n_anchor = sum(1 for c in queue if c.get("nearest_anchor"))
    n_named = sum(1 for c in queue if c.get("nearest_named_building"))
    n_ind = sum(1 for c in queue if c.get("nearest_industrial"))
    print(f"wrote enriched queue → {QUEUE_PATH}")
    print(f"  with nearest_anchor (≤{ANCHOR_RADIUS_M/1000:.0f}km): {n_anchor}/{n}")
    print(f"  with nearest_industrial (≤{INDUSTRIAL_RADIUS_M:.0f}m): {n_ind}/{n}")
    print(f"  with nearest_named_building (≤{NAMED_RADIUS_M:.0f}m): {n_named}/{n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
