"""Step 3 go/no-go: OSM-cut on c_0000000 (or any single candidate).

Buffer-merge survived_a Overture buildings into a graph, then cut edges where
a cut-class OSM road crosses the LineString between two buildings. Connected
components → final clusters. Report cluster size + extent distribution.

Decides whether OSM-cut bounds the 155 km c_0000000 chain before NAIP. If max
cluster extent > ~5 km, widen cut classes or revisit the architecture.

Cut classes (Geofabrik fclass): motorway/trunk/primary/secondary/tertiary +
their _links, plus residential. Do NOT cut: service, unclassified, track,
footway, internal driveways, rail (rail is its own layer, not in roads).

Usage:
  python -m phase3_naip.osm_cut_test --candidate-id c_0000000
  python -m phase3_naip.osm_cut_test --merge-buffer-m 500
"""

from __future__ import annotations

import argparse
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
CANDIDATE_BUILDINGS = DATA_US / "phase3_naip" / "candidate_buildings.parquet"
NC_GEOFABRIK_ZIP = DATA_US / "osm" / "nc" / "north-carolina-latest-free.shp.zip"
OUT_DIR = DATA_US / "phase3_naip" / "osm_cut_test"

CUT_FCLASS = {
    "motorway", "motorway_link",
    "trunk", "trunk_link",
    "primary", "primary_link",
    "secondary", "secondary_link",
    "tertiary", "tertiary_link",
    "residential",
}

EPSG_M = 5070


def load_buildings(candidate_id: str) -> pd.DataFrame:
    cb = pd.read_parquet(CANDIDATE_BUILDINGS)
    out = cb[cb.candidate_id == candidate_id].reset_index(drop=True)
    print(f"[cut] {candidate_id}: {len(out):,} Prune-A-kept buildings", flush=True)
    return out


def project_buildings(b: pd.DataFrame):
    """Return centroids_5070 [N,2] and bboxes_5070 [N,4] (xmin,ymin,xmax,ymax)."""
    tr = pyproj.Transformer.from_crs(4326, EPSG_M, always_xy=True).transform
    cx_lon = ((b.xmin + b.xmax) / 2.0).to_numpy()
    cy_lat = ((b.ymin + b.ymax) / 2.0).to_numpy()
    cx, cy = tr(cx_lon, cy_lat)
    xmin, ymin = tr(b.xmin.to_numpy(), b.ymin.to_numpy())
    xmax, ymax = tr(b.xmax.to_numpy(), b.ymax.to_numpy())
    centroids = np.column_stack([cx, cy]).astype(np.float64)
    bboxes = np.column_stack([xmin, ymin, xmax, ymax]).astype(np.float64)
    return centroids, bboxes


def load_cut_roads(zip_path: Path, bbox5070) -> gpd.GeoSeries:
    """Load Geofabrik roads, filter to CUT_FCLASS, project to 5070, clip to bbox."""
    t0 = time.time()
    uri = f"zip://{zip_path}!gis_osm_roads_free_1.shp"
    print(f"[cut] reading roads: {uri}", flush=True)
    g = gpd.read_file(uri, columns=["fclass"])
    print(f"[cut]   {len(g):,} roads total ({time.time()-t0:.1f}s)", flush=True)
    g = g[g.fclass.isin(CUT_FCLASS)].copy()
    print(f"[cut]   {len(g):,} cut-class roads", flush=True)
    g = g.to_crs(EPSG_M)
    xmin, ymin, xmax, ymax = bbox5070
    pad = 1000.0
    g = g.cx[xmin - pad: xmax + pad, ymin - pad: ymax + pad]
    print(f"[cut]   {len(g):,} cut roads within bbox+1km ({time.time()-t0:.1f}s)", flush=True)
    return g.geometry.reset_index(drop=True)


def candidate_pairs(centroids: np.ndarray, bboxes: np.ndarray, buffer_m: float) -> np.ndarray:
    """All (i,j) building pairs with bbox-edge distance <= buffer_m."""
    tree = cKDTree(centroids)
    diag = np.hypot(bboxes[:, 2] - bboxes[:, 0], bboxes[:, 3] - bboxes[:, 1])
    search_r = float(diag.max()) + buffer_m + 100.0
    print(f"[cut] kdtree centroid search r = {search_r:.0f} m "
          f"(max bbox diag {diag.max():.0f}m)", flush=True)
    pairs = tree.query_pairs(search_r, output_type="ndarray")
    print(f"[cut] candidate pairs (kdtree): {len(pairs):,}", flush=True)
    i, j = pairs[:, 0], pairs[:, 1]
    dx = np.maximum(0, np.maximum(bboxes[i, 0], bboxes[j, 0]) -
                       np.minimum(bboxes[i, 2], bboxes[j, 2]))
    dy = np.maximum(0, np.maximum(bboxes[i, 1], bboxes[j, 1]) -
                       np.minimum(bboxes[i, 3], bboxes[j, 3]))
    bbox_dist = np.hypot(dx, dy)
    keep = bbox_dist <= buffer_m
    pairs = pairs[keep]
    print(f"[cut] pairs within bbox-buffer {buffer_m:.0f}m: {len(pairs):,}", flush=True)
    return pairs


def cut_edges(centroids: np.ndarray, pairs: np.ndarray,
              roads: gpd.GeoSeries) -> np.ndarray:
    """Return edges (pairs) whose centroid-to-centroid LineString does NOT cross
    any cut-class road."""
    if len(pairs) == 0 or len(roads) == 0:
        return pairs
    t0 = time.time()
    strtree = STRtree(list(roads.values))
    line_coords = np.stack([centroids[pairs[:, 0]], centroids[pairs[:, 1]]], axis=1)
    lines = shapely.linestrings(line_coords)
    print(f"[cut] STRtree query {len(lines):,} lines vs {len(roads):,} roads ...",
          flush=True)
    hit_input, _ = strtree.query(lines, predicate="intersects")
    cut_mask = np.zeros(len(pairs), dtype=bool)
    cut_mask[np.unique(hit_input)] = True
    print(f"[cut]   cut {cut_mask.sum():,} / kept {(~cut_mask).sum():,} edges "
          f"({time.time()-t0:.1f}s)", flush=True)
    return pairs[~cut_mask]


def components(n: int, edges: np.ndarray) -> np.ndarray:
    if len(edges) == 0:
        return np.arange(n)
    data = np.ones(len(edges), dtype=np.int8)
    A = coo_matrix((data, (edges[:, 0], edges[:, 1])), shape=(n, n))
    A = A + A.T
    _, labels = connected_components(A, directed=False)
    return labels


def report(b: pd.DataFrame, labels: np.ndarray, centroids: np.ndarray) -> pd.DataFrame:
    df = b.copy()
    df["cluster"] = labels
    df["cx_5070"] = centroids[:, 0]
    df["cy_5070"] = centroids[:, 1]
    g = df.groupby("cluster")
    sizes = g.size()
    extents = g.apply(
        lambda x: float(np.hypot(x.cx_5070.max() - x.cx_5070.min(),
                                 x.cy_5070.max() - x.cy_5070.min()))
    )
    summary = (pd.DataFrame({
        "cluster": sizes.index,
        "n_buildings": sizes.values,
        "extent_m": extents.reindex(sizes.index).values,
    }).sort_values("extent_m", ascending=False).reset_index(drop=True))

    print(f"\n[cut] {len(summary):,} clusters from {len(df):,} buildings")
    print(f"[cut] n_buildings  median={int(sizes.median())}  "
          f"p90={int(sizes.quantile(0.9))}  max={sizes.max()}")
    print(f"[cut] extent_m     median={int(extents.median()):,}  "
          f"p90={int(extents.quantile(0.9)):,}  max={int(extents.max()):,}")

    print(f"\n[cut] top 20 clusters by extent_m:")
    print(summary.head(20).to_string(index=False))
    print(f"\n[cut] top 20 clusters by n_buildings:")
    print(summary.sort_values("n_buildings", ascending=False).head(20)
                 .to_string(index=False))

    over_5km = int((summary.extent_m > 5000).sum())
    over_3km = int((summary.extent_m > 3000).sum())
    print(f"\n[cut] gate: clusters > 3km extent = {over_3km};  > 5km = {over_5km}")
    return df, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate-id", type=str, default="c_0000000")
    ap.add_argument("--merge-buffer-m", type=float, default=300.0)
    ap.add_argument("--zip", type=Path, default=NC_GEOFABRIK_ZIP)
    args = ap.parse_args()

    if not args.zip.exists():
        raise SystemExit(f"[cut] roads zip not found: {args.zip}")

    b = load_buildings(args.candidate_id)
    if len(b) == 0:
        raise SystemExit(f"[cut] no survived_a buildings for {args.candidate_id}")
    centroids, bboxes = project_buildings(b)

    bbox5070 = (bboxes[:, 0].min(), bboxes[:, 1].min(),
                bboxes[:, 2].max(), bboxes[:, 3].max())
    print(f"[cut] bbox 5070  x={(bbox5070[2]-bbox5070[0])/1000:.1f}km  "
          f"y={(bbox5070[3]-bbox5070[1])/1000:.1f}km", flush=True)

    roads = load_cut_roads(args.zip, bbox5070)
    pairs = candidate_pairs(centroids, bboxes, args.merge_buffer_m)
    edges = cut_edges(centroids, pairs, roads)
    labels = components(len(b), edges)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, summary = report(b, labels, centroids)
    cand = args.candidate_id
    df.to_parquet(OUT_DIR / f"{cand}_buildings_clustered.parquet", index=False)
    summary.to_parquet(OUT_DIR / f"{cand}_cluster_summary.parquet", index=False)
    print(f"\n[cut] wrote -> {OUT_DIR}/{cand}_*.parquet")


if __name__ == "__main__":
    main()
