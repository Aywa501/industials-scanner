"""Step 4 prep: build the per-cluster NAIP manifest.

For each US state, discover the most-recent NAIP year + resolution available as
COG under `s3://naip-analytic/{state}/{year}/{res}/rgbir_cog/`. Download the
tile-index shapefile per chosen state-year and cache a unified national tile
index. Then for each OSM-cut cluster (from cluster_osm_cut.py), look up which
NAIP COG tiles intersect its (buildings + fetch buffer) bbox.

Clusters come from cluster_osm_cut.py — buffer-merged + road-bounded groupings
that are NAIP-feasible by construction. The 2 km grid sub-clustering of an
earlier iteration is gone; the OSM cut provides better semantic bounds.

NAIP bucket auth = requester-pays; free in-region from us-west-2.

Reads:
  data_us/phase3_naip/clusters.parquet           (from cluster_osm_cut.py)
Writes:
  data_us/phase3_naip/naip_tile_index.parquet    (cached unified state-year index)
  data_us/phase3_naip/naip_manifest.parquet      (one row per cluster)

Usage:
  python -m phase3_naip.build_naip_manifest                       # all CONUS
  python -m phase3_naip.build_naip_manifest --states ca,az,ga     # subset
  python -m phase3_naip.build_naip_manifest --refresh-index       # re-probe + DL
  python -m phase3_naip.build_naip_manifest --clusters clusters.parquet
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import boto3
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from shapely import STRtree

SITES_US = Path(__file__).resolve().parents[1]
load_dotenv(SITES_US / ".env")
DATA_US = SITES_US.parent / "data_us"
OUT_DIR = DATA_US / "phase3_naip"
INDEX_DIR = DATA_US / "external" / "naip_indices"
INDEX_CACHE = OUT_DIR / "naip_tile_index.parquet"

NAIP_BUCKET = "naip-analytic"
AWS_REGION = "us-west-2"
RES_PREFERENCE = ["60cm", "30cm", "100cm"]  # smaller pixel preferred (less compute)

# 48 CONUS states + DC (lower-case as in S3 prefixes)
CONUS_STATES = [
    "al", "az", "ar", "ca", "co", "ct", "de", "dc", "fl", "ga",
    "id", "il", "in", "ia", "ks", "ky", "la", "me", "md", "ma",
    "mi", "mn", "ms", "mo", "mt", "ne", "nv", "nh", "nj", "nm",
    "ny", "nc", "nd", "oh", "ok", "or", "pa", "ri", "sc", "sd",
    "tn", "tx", "ut", "vt", "va", "wa", "wv", "wi", "wy",
]

# Filename: m_{quad_id}_{quadrant}_{utm_zone}_{res}_{acq_yyyymmdd}[_{proc_yyyymmdd}].tif
# Both single-date and double-date variants exist depending on state-year.
NAIP_FILENAME_RE = re.compile(
    r"^m_(\d{7,8})_(ne|nw|se|sw)_(\d+)_(\d+)_(\d{8})(?:_(\d{8}))?\.tif$"
)

FETCH_BUFFER_M = 100.0    # extra context around each cluster's building bbox

M_PER_DEG_LAT = 110_540.0
M_PER_DEG_LON_EQ = 111_320.0


# ---------- S3 discovery -----------------------------------------------------

def _s3_client():
    return boto3.client("s3", region_name=AWS_REGION)


def _list_prefix(s3, prefix: str) -> list[str]:
    """Return the immediate sub-prefix names under `s3://NAIP_BUCKET/{prefix}`."""
    out = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=NAIP_BUCKET, Prefix=prefix,
                                   Delimiter="/", RequestPayer="requester"):
        for p in page.get("CommonPrefixes", []):
            name = p["Prefix"][len(prefix):].rstrip("/")
            out.append(name)
    return out


def _has_objects(s3, prefix: str) -> bool:
    resp = s3.list_objects_v2(Bucket=NAIP_BUCKET, Prefix=prefix, MaxKeys=1,
                              RequestPayer="requester")
    return resp.get("KeyCount", 0) > 0


def detect_naming_convention(s3, state: str, year: int, res: str) -> str:
    """Probe rgbir_cog/ to detect 'single_date' (m_..._YYYYMMDD.tif) vs
    'double_date' (m_..._YYYYMMDD_YYYYMMDD.tif). The two-date form has acq + proc,
    the one-date form is acq only (re-encoded COGs sometimes drop proc date)."""
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=NAIP_BUCKET,
                                   Prefix=f"{state}/{year}/{res}/rgbir_cog/",
                                   RequestPayer="requester", MaxKeys=20):
        for obj in page.get("Contents", []):
            name = obj["Key"].rsplit("/", 1)[-1]
            if name.endswith(".tif"):
                parts = name[:-4].split("_")
                if len(parts) == 7:
                    return "double_date"
                if len(parts) == 6:
                    return "single_date"
        break
    return "unknown"


def _find_index_shp(s3, state: str, year: int, res: str) -> str | None:
    """Return the S3 key of an index shapefile under {state}/{year}/{res}/index/, or None.

    NAIP's index filenames are inconsistent across state-years (e.g. NH 2023's
    shapefile is misnamed ND_NAIP23_QQ.shp). Listing the directory and picking
    whatever .shp is there is more robust than guessing STATE_NAIPyy_QQ.shp."""
    prefix = f"{state}/{year}/{res}/index/"
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=NAIP_BUCKET, Prefix=prefix,
                                   RequestPayer="requester", MaxKeys=50):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".shp"):
                return obj["Key"]
    return None


def discover_state_year(s3, state: str) -> tuple[int, str] | None:
    """Latest (year, res) with both an rgbir_cog/ directory AND an index/*.shp.

    Both must be present — NAIP sometimes uploads metadata XMLs (fgdc/) before
    the actual COGs land, or publishes COGs without a tile index (e.g. RI 2023).
    Year iteration is descending so we always prefer the freshest viable
    combo."""
    years = _list_prefix(s3, f"{state}/")
    years = sorted([int(y) for y in years if y.isdigit() and len(y) == 4], reverse=True)
    for year in years:
        resolutions = _list_prefix(s3, f"{state}/{year}/")
        ordered = [r for r in RES_PREFERENCE if r in resolutions]
        ordered.extend(r for r in resolutions if r not in ordered)
        for res in ordered:
            if not _has_objects(s3, f"{state}/{year}/{res}/rgbir_cog/"):
                continue
            if _find_index_shp(s3, state, year, res) is None:
                continue
            return year, res
    return None


def download_index(s3, state: str, year: int, res: str) -> Path | None:
    """Pull the per-state-year index shapefile components to INDEX_DIR.

    The .shp basename is discovered dynamically (NAIP filename conventions
    vary). All sidecar files (.shx/.dbf/.prj/.cpg) are assumed to share the
    same basename."""
    shp_key = _find_index_shp(s3, state, year, res)
    if shp_key is None:
        print(f"[naip-idx] {state} {year} {res}: no .shp under index/")
        return None
    base = shp_key.rsplit("/", 1)[-1][:-4]  # strip directory + .shp
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{state}_{year}_{res}_{base}"
    out_shp = INDEX_DIR / f"{stem}.shp"
    needed = [".shp", ".shx", ".dbf", ".prj"]
    if all((INDEX_DIR / f"{stem}{ext}").exists() for ext in needed):
        return out_shp
    for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg"]:
        key = f"{state}/{year}/{res}/index/{base}{ext}"
        local = INDEX_DIR / f"{stem}{ext}"
        try:
            s3.download_file(NAIP_BUCKET, key, str(local),
                             ExtraArgs={"RequestPayer": "requester"})
        except ClientError as e:
            if ext == ".cpg":  # optional
                continue
            print(f"[naip-idx] {state} {year} {res}: missing {ext} "
                  f"({e.response['Error']['Code']})")
            return None
    return out_shp


# ---------- Tile-index construction -----------------------------------------

def _parse_naip_filename(fn: str) -> dict | None:
    m = NAIP_FILENAME_RE.match(fn)
    if not m:
        return None
    quad_id, quadrant, utm_zone, res_code, acq, proc = m.groups()
    return {"quad_id": quad_id, "quadrant": quadrant,
            "utm_zone": int(utm_zone), "res_code": res_code,
            "naip_acq_date": acq, "proc_date": proc or ""}


def _filename_col(gdf: gpd.GeoDataFrame) -> str:
    """Best-guess the column holding the .tif filename in a NAIP index gdf."""
    for c in ["FileName", "Filename", "FILENAME", "FILE_NAME", "filename"]:
        if c in gdf.columns:
            return c
    for c in gdf.columns:
        if gdf[c].dtype == object:
            v = gdf[c].dropna().astype(str)
            if len(v) and v.iloc[0].lower().endswith(".tif"):
                return c
    raise KeyError(f"no filename column in index gdf; columns={list(gdf.columns)}")


def load_one_index(shp: Path, state: str, year: int, res: str,
                   convention: str) -> gpd.GeoDataFrame:
    g = gpd.read_file(shp).to_crs(4326)
    fcol = _filename_col(g)
    parsed = g[fcol].astype(str).apply(_parse_naip_filename)
    keep = parsed.notna()
    g = g.loc[keep].copy()
    parsed = pd.DataFrame(list(parsed[keep]), index=g.index)
    g["state"] = state
    g["naip_year"] = year
    g["naip_res"] = res
    # Reconstruct the actual S3 filename from parsed components per convention.
    quad_id = parsed["quad_id"].values
    quadrant = parsed["quadrant"].values
    utm = parsed["utm_zone"].astype(str).values
    res_code = parsed["res_code"].values
    acq = parsed["naip_acq_date"].values
    proc = parsed["proc_date"].values if "proc_date" in parsed.columns else None
    if convention == "single_date" or proc is None:
        fnames = ["m_" + q + "_" + qd + "_" + u + "_" + rc + "_" + a + ".tif"
                 for q, qd, u, rc, a in zip(quad_id, quadrant, utm, res_code, acq)]
    else:
        fnames = ["m_" + q + "_" + qd + "_" + u + "_" + rc + "_" + a + "_" + p + ".tif"
                 for q, qd, u, rc, a, p in zip(quad_id, quadrant, utm, res_code,
                                               acq, proc)]
    g["tile_filename"] = fnames
    quad_dir = g["tile_filename"].str[2:7]
    g["tile_uri"] = ("s3://" + NAIP_BUCKET + "/" + state + "/" + str(year) +
                     "/" + res + "/rgbir_cog/" + quad_dir + "/" + g["tile_filename"])
    g["naip_acq_date"] = acq
    bnds = g.geometry.bounds
    g["lon_min"] = bnds.minx.values
    g["lon_max"] = bnds.maxx.values
    g["lat_min"] = bnds.miny.values
    g["lat_max"] = bnds.maxy.values
    return g[["state", "naip_year", "naip_res", "tile_filename", "tile_uri",
              "naip_acq_date", "lon_min", "lon_max", "lat_min", "lat_max",
              "geometry"]]


def build_tile_index(states: list[str]) -> pd.DataFrame:
    """Discover, download, and stitch the NAIP tile indices for `states`."""
    s3 = _s3_client()
    pieces = []
    for state in states:
        t0 = time.time()
        sy = discover_state_year(s3, state)
        if sy is None:
            print(f"[naip-idx] {state.upper()}: no rgbir_cog year found, skipping")
            continue
        year, res = sy
        shp = download_index(s3, state, year, res)
        if shp is None:
            continue
        convention = detect_naming_convention(s3, state, year, res)
        if convention == "unknown":
            print(f"[naip-idx] {state.upper()}: unknown naming convention, skipping")
            continue
        gdf = load_one_index(shp, state, year, res, convention)
        print(f"[naip-idx] {state.upper()}: {year} {res} ({convention}) -> "
              f"{len(gdf):,} tiles ({time.time()-t0:.1f}s)", flush=True)
        pieces.append(gdf)
    if not pieces:
        raise SystemExit("[naip-idx] no tile indices loaded")
    return pd.concat(pieces, ignore_index=True)


# ---------- Cluster fetch-bbox ----------------------------------------------

def add_fetch_bbox(clusters: pd.DataFrame,
                   fetch_buffer_m: float = FETCH_BUFFER_M) -> pd.DataFrame:
    """Add fetch_{lat,lon}_{min,max} columns (cluster bbox + fetch buffer)."""
    agg = clusters.copy()
    coslat = np.cos(np.radians(agg.lat.to_numpy()))
    dlat = fetch_buffer_m / M_PER_DEG_LAT
    dlon = fetch_buffer_m / (M_PER_DEG_LON_EQ * np.maximum(coslat, 0.1))
    agg["fetch_lat_min"] = agg.lat_min - dlat
    agg["fetch_lat_max"] = agg.lat_max + dlat
    agg["fetch_lon_min"] = agg.lon_min - dlon
    agg["fetch_lon_max"] = agg.lon_max + dlon
    return agg


# ---------- Tile lookup -----------------------------------------------------

def lookup_tiles(clusters: pd.DataFrame,
                 tile_index: pd.DataFrame) -> pd.DataFrame:
    """For each cluster fetch bbox, list intersecting NAIP COG URIs."""
    cl_boxes = shapely.box(clusters.fetch_lon_min.to_numpy(),
                           clusters.fetch_lat_min.to_numpy(),
                           clusters.fetch_lon_max.to_numpy(),
                           clusters.fetch_lat_max.to_numpy())
    tile_boxes = shapely.box(tile_index.lon_min.to_numpy(),
                             tile_index.lat_min.to_numpy(),
                             tile_index.lon_max.to_numpy(),
                             tile_index.lat_max.to_numpy())
    tree = STRtree(tile_boxes)
    pairs = tree.query(cl_boxes, predicate="intersects")  # (2, K)
    cl_ix, tile_ix = pairs[0], pairs[1]
    uris = tile_index.tile_uri.to_numpy()
    dates = tile_index.naip_acq_date.to_numpy()
    years = tile_index.naip_year.to_numpy()
    states = tile_index.state.to_numpy()
    reses = tile_index.naip_res.to_numpy()

    n = len(clusters)
    out_uris: list[list[str]] = [[] for _ in range(n)]
    out_dates: list[list[str]] = [[] for _ in range(n)]
    out_year: list[int | None] = [None] * n
    out_state: list[str | None] = [None] * n
    out_res: list[str | None] = [None] * n
    for ci, ti in zip(cl_ix.tolist(), tile_ix.tolist()):
        out_uris[ci].append(str(uris[ti]))
        out_dates[ci].append(str(dates[ti]))
        if out_year[ci] is None:
            out_year[ci] = int(years[ti])
            out_state[ci] = str(states[ti])
            out_res[ci] = str(reses[ti])
    clusters = clusters.copy()
    clusters["naip_uris"] = out_uris
    clusters["naip_acq_dates"] = out_dates
    clusters["naip_year"] = out_year
    clusters["naip_state"] = out_state
    clusters["naip_res"] = out_res
    return clusters


# ---------- main ------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=str, default=None,
                    help="comma-separated CONUS state codes; default = all 49")
    ap.add_argument("--refresh-index", action="store_true",
                    help="re-probe S3 and re-download tile indices")
    ap.add_argument("--clusters", type=str, default="clusters.parquet",
                    help="cluster summary parquet (in data_us/phase3_naip/)")
    ap.add_argument("--out-name", type=str, default="naip_manifest.parquet")
    ap.add_argument("--fetch-buffer-m", type=float, default=FETCH_BUFFER_M,
                    help="context buffer around each cluster's bbox")
    args = ap.parse_args()

    states = ([s.strip().lower() for s in args.states.split(",")]
              if args.states else CONUS_STATES)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.refresh_index or not INDEX_CACHE.exists():
        print(f"[naip-idx] building tile index for {len(states)} states", flush=True)
        tile_index = build_tile_index(states)
        pd.DataFrame(tile_index.drop(columns=["geometry"])).to_parquet(
            INDEX_CACHE, index=False)
        print(f"[naip-idx] {len(tile_index):,} tiles cached -> {INDEX_CACHE}",
              flush=True)
    else:
        tile_index = pd.read_parquet(INDEX_CACHE)
        if args.states:
            tile_index = tile_index[tile_index.state.isin(states)]
        print(f"[naip-idx] loaded {len(tile_index):,} cached tiles "
              f"({tile_index.state.nunique()} states)", flush=True)

    cl_path = OUT_DIR / args.clusters
    clusters = pd.read_parquet(cl_path)
    print(f"[naip-mfst] {len(clusters):,} clusters across "
          f"{clusters.candidate_id.nunique():,} candidates", flush=True)

    clusters = add_fetch_bbox(clusters, args.fetch_buffer_m)
    clusters = lookup_tiles(clusters, tile_index)
    no_tile = clusters.naip_year.isna().sum()
    if no_tile:
        print(f"[naip-mfst] WARN {no_tile:,} clusters had no intersecting NAIP tile",
              flush=True)
    clusters["n_tiles"] = clusters.naip_uris.apply(len)

    out_path = OUT_DIR / args.out_name
    clusters.to_parquet(out_path, index=False)
    print(f"[naip-mfst] wrote -> {out_path}", flush=True)
    nb = clusters.n_buildings
    nt = clusters.n_tiles
    print(f"[naip-mfst] cluster n_buildings  min/median/p90/max = "
          f"{nb.min()}/{int(nb.median())}/{int(nb.quantile(0.9))}/{nb.max()}")
    print(f"[naip-mfst] cluster n_tiles      min/median/p90/max = "
          f"{nt.min()}/{int(nt.median())}/{int(nt.quantile(0.9))}/{nt.max()}")
    print(f"[naip-mfst] top NAIP years: "
          f"{clusters.naip_year.value_counts().head(5).to_dict()}")


if __name__ == "__main__":
    main()
