"""Step 3 (production): OSM-cut clustering across all S2 candidates.

Per S2 candidate, buffer-merge Overture buildings into a graph and cut edges
where a Geofabrik cut-class road (motorway/trunk/primary/secondary/tertiary +
their _links, plus residential) crosses the LineString between two buildings.
Connected components → final clusters. Bounds the metro-scale chains BEFORE
the expensive NAIP step.

Architecture validated on c_0000000 (see NAIP_STAGE_NOTES.md go/no-go gate).

Per-state Geofabrik shapefiles are downloaded on demand to data_us/external/osm/<st>/.
Roads are loaded once per state, then re-used across every candidate in that
state. Multi-state candidates (rare) load both states.

Reads:
  data_us/phase3_naip/candidate_buildings.parquet     (post Prune A)
  data_us/phase3_naip/candidates_with_buildings.parquet
  data_us/phase3_naip/naip_tile_index.parquet         (for state assignment)
Writes:
  data_us/phase3_naip/cluster_buildings.parquet       (per-building, cluster_id)
  data_us/phase3_naip/clusters.parquet                (per-cluster summary)

Usage:
  python -m phase3_naip.cluster_osm_cut                            # all states
  python -m phase3_naip.cluster_osm_cut --states nc                # one state
  python -m phase3_naip.cluster_osm_cut --candidate-ids c_0000000  # one cand
  python -m phase3_naip.cluster_osm_cut --merge-buffer-m 500
"""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import shapely
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from shapely.strtree import STRtree

DATA_US = Path(__file__).resolve().parents[2] / "data_us"
OUT_DIR = DATA_US / "phase3_naip"
OSM_DIR = DATA_US / "external" / "osm"

CANDIDATE_BUILDINGS = OUT_DIR / "candidate_buildings.parquet"
CANDIDATES_SUMMARY = OUT_DIR / "candidates_with_buildings.parquet"
NAIP_TILE_INDEX = OUT_DIR / "naip_tile_index.parquet"

CUT_FCLASS = {
    "motorway", "motorway_link",
    "trunk", "trunk_link",
    "primary", "primary_link",
    "secondary", "secondary_link",
    "tertiary", "tertiary_link",
    "residential",
}

# Geofabrik per-state download paths. Most states are a single shp.zip; a few
# (CA) are split into sub-regions. Each value is a list of (slug, url_path)
# pairs whose road files we concatenate at load time.
GEOFABRIK_BASE = "https://download.geofabrik.de/north-america/us"


def _one(slug: str) -> list[tuple[str, str]]:
    return [(slug, f"{slug}-latest-free.shp.zip")]


STATE_PARTS: dict[str, list[tuple[str, str]]] = {
    "al": _one("alabama"), "az": _one("arizona"), "ar": _one("arkansas"),
    "ca": [("norcal", "california/norcal-latest-free.shp.zip"),
           ("socal", "california/socal-latest-free.shp.zip")],
    "co": _one("colorado"), "ct": _one("connecticut"), "de": _one("delaware"),
    "dc": _one("district-of-columbia"), "fl": _one("florida"),
    "ga": _one("georgia"), "id": _one("idaho"), "il": _one("illinois"),
    "in": _one("indiana"), "ia": _one("iowa"), "ks": _one("kansas"),
    "ky": _one("kentucky"), "la": _one("louisiana"), "me": _one("maine"),
    "md": _one("maryland"), "ma": _one("massachusetts"),
    "mi": _one("michigan"), "mn": _one("minnesota"),
    "ms": _one("mississippi"), "mo": _one("missouri"),
    "mt": _one("montana"), "ne": _one("nebraska"), "nv": _one("nevada"),
    "nh": _one("new-hampshire"), "nj": _one("new-jersey"),
    "nm": _one("new-mexico"), "ny": _one("new-york"),
    "nc": _one("north-carolina"), "nd": _one("north-dakota"),
    "oh": _one("ohio"), "ok": _one("oklahoma"), "or": _one("oregon"),
    "pa": _one("pennsylvania"), "ri": _one("rhode-island"),
    "sc": _one("south-carolina"), "sd": _one("south-dakota"),
    "tn": _one("tennessee"), "tx": _one("texas"), "ut": _one("utah"),
    "vt": _one("vermont"), "va": _one("virginia"),
    "wa": _one("washington"), "wv": _one("west-virginia"),
    "wi": _one("wisconsin"), "wy": _one("wyoming"),
}

EPSG_M = 5070
DEFAULT_MERGE_BUFFER_M = 300.0

# Post-cluster anchor filter. A cluster survives if its largest building is
# ≥ ANCHOR_AREA_M2, OR if any member has class ∈ ANCHOR_CLASSES (Overture also
# tags small auxiliary structures — security booths, conveyor housing — as
# industrial; the escape valve preserves small standalone industrial-class
# clusters that the area floor would otherwise drop).
DEFAULT_ANCHOR_AREA_M2 = 1000.0
ANCHOR_CLASSES = {"industrial", "warehouse", "hangar", "factory", "manufacture"}

# Pre-cluster building bbox-area floor. Industrial site announcements
# correspond to large footprints; sub-5000 m² bbox buildings are dominated by
# residential/small-commercial noise that we don't expect to match an
# announcement. Drop them before clustering.
DEFAULT_BBOX_AREA_FLOOR_M2 = 5000.0
# Post-cluster total bbox-area floor. After per-building filter, a cluster's
# summed building bbox area must reach this floor; industrial complexes
# converge on ≥ 1 ha in remote-sensing literature.
DEFAULT_CLUSTER_TOTAL_BBOX_M2 = 10000.0


# ---------- state assignment ------------------------------------------------

def assign_state_per_candidate(cands: pd.DataFrame,
                               tile_index: pd.DataFrame) -> dict[str, str]:
    """Determine the primary NAIP state for each candidate via tile-bbox overlap.

    Returns {candidate_id: state_lower}; candidates with no tile match are omitted.
    """
    cand_boxes = shapely.box(cands.lon_min.to_numpy(), cands.lat_min.to_numpy(),
                             cands.lon_max.to_numpy(), cands.lat_max.to_numpy())
    tile_boxes = shapely.box(tile_index.lon_min.to_numpy(),
                             tile_index.lat_min.to_numpy(),
                             tile_index.lon_max.to_numpy(),
                             tile_index.lat_max.to_numpy())
    tree = STRtree(tile_boxes)
    pairs = tree.query(cand_boxes, predicate="intersects")
    cand_ix, tile_ix = pairs[0], pairs[1]
    states = tile_index.state.to_numpy()
    # For each candidate, take the most-frequent intersecting tile state.
    out: dict[str, str] = {}
    counts: dict[str, dict[str, int]] = {}
    cand_id = cands.candidate_id.to_numpy()
    for ci, ti in zip(cand_ix.tolist(), tile_ix.tolist()):
        cid = str(cand_id[ci])
        st = str(states[ti])
        counts.setdefault(cid, {})[st] = counts.setdefault(cid, {}).get(st, 0) + 1
    for cid, sts in counts.items():
        out[cid] = max(sts.items(), key=lambda kv: kv[1])[0]
    return out


# ---------- per-state Geofabrik road loading -------------------------------

def _part_zip_path(state: str, slug: str) -> Path:
    return OSM_DIR / state / f"{slug}-latest-free.shp.zip"


def _download_part(state: str, slug: str, url_path: str) -> Path:
    zp = _part_zip_path(state, slug)
    if zp.exists() and zp.stat().st_size > 1_000_000:
        return zp
    if zp.exists():
        print(f"[cluster]   discarding truncated/empty {zp.name}", flush=True)
        zp.unlink()
    zp.parent.mkdir(parents=True, exist_ok=True)
    url = f"{GEOFABRIK_BASE}/{url_path}"
    print(f"[cluster] downloading {url}", flush=True)
    t0 = time.time()
    subprocess.run(["curl", "-L", "-f", "-o", str(zp), url], check=True)
    print(f"[cluster]   downloaded {zp.stat().st_size/1e6:.0f} MB "
          f"({time.time()-t0:.0f}s)", flush=True)
    return zp


def load_state_cut_roads(state: str) -> gpd.GeoSeries:
    """Cut-class road geoms in EPSG:5070 for one state (concatenated across
    parts for split states like CA). Returns empty series if state unknown."""
    if state not in STATE_PARTS:
        print(f"[cluster] WARN unknown state {state!r}, skipping", flush=True)
        return gpd.GeoSeries([], crs=EPSG_M)
    t0 = time.time()
    series_list: list[gpd.GeoSeries] = []
    for slug, url_path in STATE_PARTS[state]:
        zp = _download_part(state, slug, url_path)
        uri = f"zip://{zp}!gis_osm_roads_free_1.shp"
        g = gpd.read_file(uri, columns=["fclass"])
        g = g[g.fclass.isin(CUT_FCLASS)].copy()
        g = g.to_crs(EPSG_M)
        series_list.append(g.geometry)
    geom = (gpd.GeoSeries(pd.concat([s.reset_index(drop=True) for s in series_list],
                                    ignore_index=True), crs=EPSG_M)
            if series_list else gpd.GeoSeries([], crs=EPSG_M))
    print(f"[cluster]   {state}: {len(geom):,} cut-class roads "
          f"({time.time()-t0:.1f}s)", flush=True)
    return geom


# ---------- one-candidate cluster ------------------------------------------

def project_buildings(b: pd.DataFrame):
    tr = pyproj.Transformer.from_crs(4326, EPSG_M, always_xy=True).transform
    cx_lon = ((b.xmin + b.xmax) / 2.0).to_numpy()
    cy_lat = ((b.ymin + b.ymax) / 2.0).to_numpy()
    cx, cy = tr(cx_lon, cy_lat)
    xmin, ymin = tr(b.xmin.to_numpy(), b.ymin.to_numpy())
    xmax, ymax = tr(b.xmax.to_numpy(), b.ymax.to_numpy())
    centroids = np.column_stack([cx, cy]).astype(np.float64)
    bboxes = np.column_stack([xmin, ymin, xmax, ymax]).astype(np.float64)
    return centroids, bboxes


def cluster_one_candidate(b: pd.DataFrame, road_tree: STRtree | None,
                          buffer_m: float) -> np.ndarray:
    """Run buffer-merge + OSM-cut on one candidate's buildings. Returns
    cluster labels [N] (0-indexed per candidate)."""
    n = len(b)
    if n == 0:
        return np.array([], dtype=np.int32)
    if n == 1:
        return np.zeros(1, dtype=np.int32)
    centroids, bboxes = project_buildings(b)

    tree = cKDTree(centroids)
    diag = np.hypot(bboxes[:, 2] - bboxes[:, 0], bboxes[:, 3] - bboxes[:, 1])
    search_r = float(diag.max()) + buffer_m + 100.0
    pairs = tree.query_pairs(search_r, output_type="ndarray")
    if len(pairs):
        i, j = pairs[:, 0], pairs[:, 1]
        dx = np.maximum(0, np.maximum(bboxes[i, 0], bboxes[j, 0]) -
                           np.minimum(bboxes[i, 2], bboxes[j, 2]))
        dy = np.maximum(0, np.maximum(bboxes[i, 1], bboxes[j, 1]) -
                           np.minimum(bboxes[i, 3], bboxes[j, 3]))
        pairs = pairs[np.hypot(dx, dy) <= buffer_m]

    if len(pairs) and road_tree is not None:
        line_coords = np.stack([centroids[pairs[:, 0]], centroids[pairs[:, 1]]],
                               axis=1)
        lines = shapely.linestrings(line_coords)
        hit_input, _ = road_tree.query(lines, predicate="intersects")
        cut = np.zeros(len(pairs), dtype=bool)
        cut[np.unique(hit_input)] = True
        pairs = pairs[~cut]

    if len(pairs) == 0:
        return np.arange(n, dtype=np.int32)
    data = np.ones(len(pairs), dtype=np.int8)
    A = coo_matrix((data, (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, labels = connected_components(A + A.T, directed=False)
    return labels.astype(np.int32)


# ---------- aggregation -----------------------------------------------------

M_PER_DEG_LAT = 110_540.0
M_PER_DEG_LON_EQ = 111_320.0


def cluster_summary(b: pd.DataFrame) -> pd.DataFrame:
    """Per-cluster aggregation in 4326 (lat/lon)."""
    g = b.groupby("cluster_id", sort=False)
    agg = g.agg(
        candidate_id=("candidate_id", "first"),
        s2_max_prob=("s2_max_prob", "first"),
        n_buildings=("building_id", "size"),
        lat=("lat", "mean"),
        lon=("lon", "mean"),
        lat_min=("ymin", "min"),
        lat_max=("ymax", "max"),
        lon_min=("xmin", "min"),
        lon_max=("xmax", "max"),
    ).reset_index()
    coslat = np.cos(np.radians(agg.lat.to_numpy()))
    span_x = (agg.lon_max - agg.lon_min) * M_PER_DEG_LON_EQ * coslat
    span_y = (agg.lat_max - agg.lat_min) * M_PER_DEG_LAT
    agg["span_m"] = np.hypot(span_x, span_y).round(0).astype(np.int32)
    return agg[["cluster_id", "candidate_id", "s2_max_prob", "n_buildings",
                "lat", "lon", "lat_min", "lat_max", "lon_min", "lon_max",
                "span_m"]]


# ---------- main ------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=str, default=None,
                    help="comma-separated state codes; default = all encountered")
    ap.add_argument("--candidate-ids", type=str, default=None,
                    help="comma-separated candidate_ids to process (overrides --states)")
    ap.add_argument("--merge-buffer-m", type=float, default=DEFAULT_MERGE_BUFFER_M)
    ap.add_argument("--anchor-area-m2", type=float, default=DEFAULT_ANCHOR_AREA_M2,
                    help="cluster must have max(area) ≥ this OR an anchor-class member")
    ap.add_argument("--bbox-area-floor-m2", type=float,
                    default=DEFAULT_BBOX_AREA_FLOOR_M2,
                    help=f"drop buildings whose lon/lat bbox area is below this "
                         f"floor before clustering (default "
                         f"{DEFAULT_BBOX_AREA_FLOOR_M2:.0f} m²; 0 disables)")
    ap.add_argument("--cluster-total-bbox-m2", type=float,
                    default=DEFAULT_CLUSTER_TOTAL_BBOX_M2,
                    help=f"drop clusters whose summed building bbox area is "
                         f"below this floor (default "
                         f"{DEFAULT_CLUSTER_TOTAL_BBOX_M2:.0f} m²; 0 disables)")
    ap.add_argument("--out-suffix", type=str, default="",
                    help="suffix on output parquet names (for parallel experiments)")
    args = ap.parse_args()

    print(f"[cluster] loading candidates + buildings ...", flush=True)
    cands = pd.read_parquet(CANDIDATES_SUMMARY)
    buildings = pd.read_parquet(CANDIDATE_BUILDINGS)
    tile_index = pd.read_parquet(NAIP_TILE_INDEX)
    print(f"[cluster] {len(cands):,} candidates, {len(buildings):,} buildings, "
          f"{len(tile_index):,} NAIP tiles", flush=True)

    if args.bbox_area_floor_m2 > 0:
        lat_mid = 0.5 * (buildings["ymin"] + buildings["ymax"])
        dx_m = ((buildings["xmax"] - buildings["xmin"])
                * 111320.0 * np.cos(np.radians(lat_mid)))
        dy_m = (buildings["ymax"] - buildings["ymin"]) * 111320.0
        bbox_area = (dx_m * dy_m).abs()
        pre = len(buildings)
        buildings = buildings[bbox_area >= args.bbox_area_floor_m2].reset_index(drop=True)
        print(f"[cluster] bbox-area floor ≥{args.bbox_area_floor_m2:.0f} m²: "
              f"{pre:,} → {len(buildings):,} buildings "
              f"({pre - len(buildings):,} dropped)", flush=True)

    cand_state = assign_state_per_candidate(cands, tile_index)
    no_state = set(cands.candidate_id) - set(cand_state)
    if no_state:
        print(f"[cluster] WARN {len(no_state):,} candidates have no NAIP tile, "
              "skipping (will not be clustered)", flush=True)

    if args.candidate_ids:
        keep = set(args.candidate_ids.split(","))
        cand_state = {k: v for k, v in cand_state.items() if k in keep}
    elif args.states:
        sel = set(s.strip().lower() for s in args.states.split(","))
        cand_state = {k: v for k, v in cand_state.items() if v in sel}
    print(f"[cluster] selected {len(cand_state):,} candidates "
          f"across {len(set(cand_state.values()))} states", flush=True)

    # Group candidates by state for batched road loading.
    by_state: dict[str, list[str]] = {}
    for cid, st in cand_state.items():
        by_state.setdefault(st, []).append(cid)

    all_labels: list[pd.DataFrame] = []
    t0 = time.time()
    n_done = 0
    for state, cand_ids in sorted(by_state.items()):
        print(f"\n[cluster] === state {state} : {len(cand_ids):,} candidates ===",
              flush=True)
        roads = load_state_cut_roads(state)
        road_tree = STRtree(list(roads.values)) if len(roads) else None
        for cid in cand_ids:
            cb = buildings[buildings.candidate_id == cid].reset_index(drop=True)
            labels = cluster_one_candidate(cb, road_tree, args.merge_buffer_m)
            cb = cb[["candidate_id", "building_id", "lon", "lat",
                     "xmin", "xmax", "ymin", "ymax", "approx_area_m2",
                     "class", "subtype", "s2_max_prob"]].copy()
            cb["cluster_id"] = [f"{cid}_c{k}" for k in labels]
            all_labels.append(cb)
            n_done += 1
            if n_done % 500 == 0:
                rate = n_done / max(1.0, time.time() - t0)
                eta = (len(cand_state) - n_done) / max(rate, 1e-6) / 60.0
                print(f"[cluster]   {n_done:,}/{len(cand_state):,} cands "
                      f"({rate:.1f}/s, ETA {eta:.0f} min)", flush=True)

    if not all_labels:
        raise SystemExit("[cluster] no candidates processed")
    out_b = pd.concat(all_labels, ignore_index=True)
    pre_n_clusters = out_b.cluster_id.nunique()
    print(f"\n[cluster] {len(out_b):,} buildings clustered "
          f"({pre_n_clusters:,} distinct clusters)", flush=True)

    # Anchor filter: keep clusters with max(area) ≥ anchor_area_m2 OR an
    # anchor-class member. Drops small-NaN noise singletons while preserving
    # tagged small industrial outliers.
    g = out_b.groupby("cluster_id", sort=False)
    cluster_keep = (
        (g["approx_area_m2"].max() >= args.anchor_area_m2)
        | (g["class"].apply(lambda s: s.isin(ANCHOR_CLASSES).any()))
    )
    keep_ids = set(cluster_keep[cluster_keep].index)
    pre_b = len(out_b)
    out_b = out_b[out_b.cluster_id.isin(keep_ids)].reset_index(drop=True)
    post_n_clusters = out_b.cluster_id.nunique()
    print(f"[cluster] anchor filter (≥{args.anchor_area_m2:.0f} m² OR anchor-class): "
          f"{pre_n_clusters:,} → {post_n_clusters:,} clusters "
          f"({pre_n_clusters - post_n_clusters:,} dropped); "
          f"{pre_b:,} → {len(out_b):,} buildings", flush=True)

    if args.cluster_total_bbox_m2 > 0:
        lat_mid = 0.5 * (out_b["ymin"] + out_b["ymax"])
        dx_m = ((out_b["xmax"] - out_b["xmin"])
                * 111320.0 * np.cos(np.radians(lat_mid)))
        dy_m = (out_b["ymax"] - out_b["ymin"]) * 111320.0
        bbox_area = (dx_m * dy_m).abs()
        totals = bbox_area.groupby(out_b["cluster_id"]).sum()
        keep_total = set(totals[totals >= args.cluster_total_bbox_m2].index)
        pre_b2 = len(out_b); pre_c2 = out_b.cluster_id.nunique()
        out_b = out_b[out_b.cluster_id.isin(keep_total)].reset_index(drop=True)
        print(f"[cluster] cluster-total bbox ≥{args.cluster_total_bbox_m2:.0f} m²: "
              f"{pre_c2:,} → {out_b.cluster_id.nunique():,} clusters "
              f"({pre_c2 - out_b.cluster_id.nunique():,} dropped); "
              f"{pre_b2:,} → {len(out_b):,} buildings", flush=True)

    summary = cluster_summary(out_b)
    suf = args.out_suffix
    bp = OUT_DIR / f"cluster_buildings{suf}.parquet"
    sp = OUT_DIR / f"clusters{suf}.parquet"
    out_b.to_parquet(bp, index=False)
    summary.to_parquet(sp, index=False)
    print(f"[cluster] wrote -> {bp}")
    print(f"[cluster] wrote -> {sp}")

    nb = summary.n_buildings
    sm = summary.span_m
    print(f"[cluster] cluster n_buildings  median={int(nb.median())} "
          f"p90={int(nb.quantile(0.9))} max={nb.max():,}")
    print(f"[cluster] cluster span_m       median={int(sm.median())} "
          f"p90={int(sm.quantile(0.9)):,} max={sm.max():,}")
    print(f"[cluster] clusters > 3 km extent: {int((sm > 3000).sum()):,}; "
          f"> 5 km: {int((sm > 5000).sum()):,}")


if __name__ == "__main__":
    main()
