"""Stage 3 Part 2, Steps 1-2: per-S2-candidate Overture retrieval + Prune A.

Reads:
  data_us/phase3_candidates_v2.parquet
  data_us/overture_industrial_conus_2025_aligned.parquet
Writes:
  data_us/phase3_naip/candidates_with_buildings.parquet   (per-candidate summary)
  data_us/phase3_naip/candidate_buildings.parquet         (per-(candidate, building))

The aligned Overture parquet stores axis-aligned bounding boxes, not true polygons.
Per S2 candidate we keep every Overture building within the candidate's bbox plus
a retrieval buffer (default ~half a tile-width + context). Buildings whose
class/subtype is confidently non-industrial — or whose footprint is below the
area floor — are **dropped outright** (not flagged); the Overture gate then
drops S2 candidates with zero remaining buildings. Step 3 (OSM-cut clustering)
takes over from here.

Drop set is widened from the v1 residential-only floor: educational, religious,
medical, hospitality, sports/assembly, agricultural, and pure-office classes
are confidently non-industrial when labelled. `commercial / retail / civic /
government / military / transportation / NaN` are kept (ambiguous — many
distribution centers tag as commercial, and military/government complexes can
be industrial). Overture class coverage is patchy (NaN dominates), so this
prune is a noise reducer, not a comprehensive filter.

Usage:
  python -m phase3_naip.overture_groups
  python -m phase3_naip.overture_groups --retrieval-buffer-m 1500 --limit 200
  python -m phase3_naip.overture_groups --candidate-ids c_0000000,c_0000005
  python -m phase3_naip.overture_groups --area-floor-m2 100
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

DATA_US = Path(__file__).resolve().parents[2] / "data_us"
CANDIDATES_PATH = DATA_US / "phase3_candidates_v2.parquet"
OVERTURE_PATH = DATA_US / "overture_industrial_conus_2025_aligned.parquet"
OUT_DIR = DATA_US / "phase3_naip"

EARTH_R_M = 6_371_000.0
M_PER_DEG_LAT = 110_540.0
M_PER_DEG_LON_EQ = 111_320.0

OVERTURE_COLS = ["id", "lon", "lat", "xmin", "xmax", "ymin", "ymax",
                 "approx_area_m2", "class", "subtype"]

# Prune A — class/subtype values that are confidently non-industrial. Bar is
# high confidence (false drops lose sites permanently). Ambiguous classes
# (commercial / retail / civic / government / military / transportation / NaN)
# are NOT in this set — distribution centers / warehouses often tag as
# commercial, and military / government complexes can be industrial.
NON_INDUSTRIAL_CLASS = {
    # residential (v1 locked)
    "house", "detached", "terrace", "semidetached_house", "apartments",
    "bungalow", "dormitory", "residential", "garage", "garages", "shed",
    "hut", "cabin", "houseboat", "static_caravan",
    # education
    "school", "university", "college", "kindergarten", "library",
    # religious
    "church", "mosque", "synagogue", "cathedral", "chapel", "temple",
    "religious",
    # medical
    "hospital", "clinic",
    # hospitality
    "hotel", "motel",
    # sports / public assembly
    "stadium", "grandstand", "sports_centre", "sports_hall", "pavilion",
    # farm / agricultural
    "farm", "farm_auxiliary", "barn", "greenhouse", "stable", "sty", "cowshed",
    # pure office (distribution complexes tag separately as warehouse/industrial)
    "office",
}
NON_INDUSTRIAL_SUBTYPE = {
    "residential", "education", "religious", "medical",
    "agricultural", "entertainment",
}
DEFAULT_AREA_FLOOR_M2 = 100.0


def load_candidates(path: Path, limit: int | None,
                    candidate_ids: list[str] | None) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if candidate_ids:
        df = df[df.candidate_id.isin(candidate_ids)].reset_index(drop=True)
    elif limit:
        df = df.head(limit).reset_index(drop=True)
    print(f"[overture] {len(df):,} S2 candidates", flush=True)
    return df


def load_overture() -> pd.DataFrame:
    t0 = time.time()
    ov = pd.read_parquet(OVERTURE_PATH, columns=OVERTURE_COLS)
    print(f"[overture] loaded {len(ov):,} Overture buildings ({time.time()-t0:.1f}s)",
          flush=True)
    return ov


def retrieve_buildings(cands: pd.DataFrame, ov: pd.DataFrame,
                       retrieval_buffer_m: float) -> pd.DataFrame:
    """Step 1: Overture buildings within each candidate bbox + retrieval buffer.

    Buildings retrieved by multiple S2 candidates are assigned to the candidate
    with the highest max_prob, so downstream sites stay non-duplicating.
    """
    t0 = time.time()
    tree = BallTree(np.radians(ov[["lat", "lon"]].to_numpy()), metric="haversine")

    cen_rad = np.radians(cands[["lat", "lon"]].to_numpy())
    radius_rad = (cands.span_m.to_numpy() / 2.0 + retrieval_buffer_m) / EARTH_R_M
    idx_per_cand = tree.query_radius(cen_rad, r=radius_rad)

    ov_lon = ov.lon.to_numpy()
    ov_lat = ov.lat.to_numpy()
    best_prob: dict[int, float] = {}
    best_cand: dict[int, str] = {}
    for ci in range(len(cands)):
        cand_idx = idx_per_cand[ci]
        if len(cand_idx) == 0:
            continue
        c = cands.iloc[ci]
        dlat = retrieval_buffer_m / M_PER_DEG_LAT
        dlon = retrieval_buffer_m / (M_PER_DEG_LON_EQ * max(np.cos(np.radians(c.lat)), 0.1))
        blon = ov_lon[cand_idx]
        blat = ov_lat[cand_idx]
        keep = ((blon >= c.lon_min - dlon) & (blon <= c.lon_max + dlon) &
                (blat >= c.lat_min - dlat) & (blat <= c.lat_max + dlat))
        prob = float(c.max_prob)
        for bi in cand_idx[keep]:
            bi = int(bi)
            if prob > best_prob.get(bi, -1.0):
                best_prob[bi] = prob
                best_cand[bi] = c.candidate_id
        if (ci + 1) % 5000 == 0:
            print(f"[overture]   {ci+1:,}/{len(cands):,} candidates "
                  f"({len(best_prob):,} buildings)", flush=True)

    if not best_prob:
        return ov.iloc[:0].assign(s2_max_prob=pd.Series(dtype=float),
                                  candidate_id=pd.Series(dtype=str))
    sel = np.fromiter(best_prob.keys(), dtype=np.int64)
    out = ov.iloc[sel].reset_index(drop=True)
    out["s2_max_prob"] = [best_prob[int(i)] for i in sel]
    out["candidate_id"] = [best_cand[int(i)] for i in sel]
    print(f"[overture] Step 1: {len(out):,} distinct buildings retrieved "
          f"({time.time()-t0:.1f}s)", flush=True)
    return out


def prune_a(buildings: pd.DataFrame, area_floor_m2: float) -> pd.DataFrame:
    """Step 2: drop confidently non-industrial buildings + tiny footprints."""
    n0 = len(buildings)
    cls = buildings["class"].fillna("").str.lower()
    sub = buildings["subtype"].fillna("").str.lower()
    non_ind = cls.isin(NON_INDUSTRIAL_CLASS) | sub.isin(NON_INDUSTRIAL_SUBTYPE)
    too_small = buildings["approx_area_m2"].fillna(0).to_numpy() < area_floor_m2
    drop = non_ind.to_numpy() | too_small
    out = buildings[~drop].reset_index(drop=True)
    print(f"[overture] Step 2: Prune A — kept {len(out):,}/{n0:,} buildings "
          f"(dropped {int(non_ind.sum()):,} non-industrial-class, "
          f"{int(too_small.sum()):,} below {area_floor_m2:.0f} m² floor)",
          flush=True)
    return out


def assemble_per_candidate(buildings: pd.DataFrame) -> pd.DataFrame:
    """Per-candidate summary; the Overture gate (zero kept buildings) is implicit
    here since buildings have already been dropped upstream."""
    g = buildings.groupby("candidate_id", sort=False)
    agg = g.agg(
        n_buildings=("building_id", "size"),
        s2_max_prob=("s2_max_prob", "first"),
        lat=("lat", "mean"),
        lon=("lon", "mean"),
        lat_min=("ymin", "min"),
        lat_max=("ymax", "max"),
        lon_min=("xmin", "min"),
        lon_max=("xmax", "max"),
    )
    agg["span_m"] = np.hypot(
        (agg.lon_max - agg.lon_min) * M_PER_DEG_LON_EQ * np.cos(np.radians(agg.lat)),
        (agg.lat_max - agg.lat_min) * M_PER_DEG_LAT,
    ).round(0).astype(int)
    agg = (agg.sort_values("s2_max_prob", ascending=False)
              .reset_index()
              [["candidate_id", "s2_max_prob", "n_buildings",
                "lat", "lon", "lat_min", "lat_max", "lon_min", "lon_max", "span_m"]])
    return agg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval-buffer-m", type=float, default=1500.0,
                    help="how far past each candidate's tile-centre bbox to retrieve "
                         "buildings; default 1500m ~ half tile-width (1.28km) + context")
    ap.add_argument("--candidates", type=str, default=str(CANDIDATES_PATH))
    ap.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    ap.add_argument("--limit", type=int, default=None,
                    help="process only the top-N candidates by score")
    ap.add_argument("--candidate-ids", type=str, default=None,
                    help="comma-separated candidate_ids to process")
    ap.add_argument("--area-floor-m2", type=float, default=DEFAULT_AREA_FLOOR_M2,
                    help=f"drop buildings smaller than this floor "
                         f"(default {DEFAULT_AREA_FLOOR_M2:.0f} m²)")
    args = ap.parse_args()

    cand_ids = args.candidate_ids.split(",") if args.candidate_ids else None
    cands = load_candidates(Path(args.candidates), args.limit, cand_ids)
    ov = load_overture()

    buildings = retrieve_buildings(cands, ov, args.retrieval_buffer_m)
    if buildings.empty:
        raise SystemExit("[overture] no Overture buildings retrieved for any candidate")
    buildings = buildings.rename(columns={"id": "building_id"})
    buildings = prune_a(buildings, args.area_floor_m2)
    if buildings.empty:
        raise SystemExit("[overture] no buildings survived Prune A")

    summary = assemble_per_candidate(buildings)
    kept = set(summary.candidate_id)
    members = buildings[buildings.candidate_id.isin(kept)].copy()
    members = members[["candidate_id", "building_id", "lon", "lat", "xmin", "xmax",
                       "ymin", "ymax", "approx_area_m2", "class", "subtype",
                       "s2_max_prob"]]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "candidates_with_buildings.parquet"
    members_path = out_dir / "candidate_buildings.parquet"
    summary.to_parquet(summary_path, index=False)
    members.to_parquet(members_path, index=False)

    n_dropped = len(cands) - len(summary)
    print(f"\n[overture] {len(summary):,} candidates kept "
          f"({n_dropped:,} dropped — no buildings survived Prune A)", flush=True)
    nb = summary.n_buildings
    sp = summary.span_m
    print(f"[overture] n_buildings/candidate min/median/p90/max = "
          f"{nb.min()}/{int(nb.median())}/{int(nb.quantile(0.9))}/{nb.max()}")
    print(f"[overture] span_m/candidate     median/p90/max = "
          f"{int(sp.median())}/{int(sp.quantile(0.9))}/{sp.max()}")
    print(f"[overture] wrote -> {summary_path}")
    print(f"[overture] wrote -> {members_path}")


if __name__ == "__main__":
    main()
