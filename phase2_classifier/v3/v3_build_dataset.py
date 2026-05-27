"""Build the v3 detector training manifest (per-building binary: industrial vs not).

v3 changes from v2:
  - Unit of analysis: per-building. Each row is one Overture building (no tile concept).
  - Imagery: NAIP (1m GSD, latest-per-state snapshot) instead of multi-year S2
    composites. No target_year column — epoch comes from naip_tile_index at embed time.
  - Output schema carries the building's WGS84 bbox so the embed step can
    compute crop = bbox + buffer directly, no re-join to Overture.
  - No ±offset jitter — at production scoring time each crop is centered on its
    building, so training must match.

Composition (same three sources as v2, same area floor / dedup / sampling targets):
  1. hand-labeled — per-site, Overture buildings within HAND_RADIUS_M of site
     centroid (≥ AREA_FLOOR_M2), labeled by site label
  2. OSM-industrial positives — class∈{industrial, warehouse, hangar} or
     subtype=industrial, OSM lineage, ≥ AREA_FLOOR_M2, 500 m dedup
  3. OSM-categorical negatives — confidently non-industrial OSM classes

Schema:
  building_id        str   unique row id
  ovt_id             str   Overture building id (provenance)
  class_id           int   0=non_industrial, 2=industrial_complete (1 reserved)
  source             str   hand_complete|hand_not_a_site|osm_industrial|osm_neg
  weight             float 5.0 hand-labels (held-out test), 1.0 bulk
  lat, lon           float WGS84 centroid
  xmin/xmax/ymin/ymax float WGS84 building bbox
  approx_area_m2     float
  ovt_class, ovt_subtype, ovt_name  str  Overture metadata (diagnostics)
  site_id            str   hand-label provenance (null for bulk rows)
  split              str   'train' for bulk, 'test' for hand
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

ROOT = Path(__file__).resolve().parents[3]
DATA_US = ROOT / "data_us"

MANUAL_LABELS = DATA_US / "labels" / "manual_labels.parquet"
OVERTURE = DATA_US / "external" / "overture_industrial_conus_2025_aligned.parquet"
CONUS_GEOJSON = DATA_US / "phase3_scan" / "cache" / "us-states.geojson"
OUT = DATA_US / "phase2" / "v3_dataset_manifest.parquet"
TEST_SET = DATA_US / "phase2" / "v3_test_set_manifest.parquet"

# The Overture "conus-aligned" parquet actually bleeds across borders (~13.6% of
# >=3000 m^2 buildings are in Canada / Mexico — names like "Waste Connections
# Canada", "Centrale thermique"). Filter to the CONUS land polygon before
# sampling so we don't emit rows with no NAIP coverage.
NON_CONUS_STATES = {"Alaska", "Hawaii", "Puerto Rico"}

OSM_INDUSTRIAL_CLASSES = {"industrial", "warehouse", "hangar"}

# Expanded for the hard-negative rebuild: commercial/retail/office/parking added
# because the v3-detector 10K validation showed them as the dominant FP source
# at p>=0.95 (7.5% commercial, 7.5% retail). The model never trained on them.
OSM_NEG_CLASSES = {
    # hard negatives (new): the boundary classes the model was missing
    "commercial", "retail", "office", "parking",
    # easy negatives (existing): keep the model calibrated on these
    "residential", "house", "detached", "terrace", "apartments",
    "semidetached_house", "dormitory",
    "school", "university", "college", "kindergarten", "library",
    "hospital",
    "church", "cathedral", "mosque", "synagogue", "religious",
    "hotel", "stadium", "grandstand", "fire_station", "civic", "government",
    "public", "post_office", "greenhouse", "farm_auxiliary", "barn",
}

AREA_FLOOR_M2 = 3000.0              # matches v2; max threshold where small-building aux property holds
OSM_INDUSTRIAL_TARGET = 38_000
OSM_NEG_TARGET = 62_000
# 500m dedup capped industrial at ~30k (below new 38k target). 250m still drops
# adjacent buildings in the same complex but opens the pool enough to hit 38k.
OSM_DEDUP_M = 250
HAND_RADIUS_M = 400                 # buildings within this of hand-site centroid take site's label.
                                    # 200m only captures dead-center buildings; 400m gives ~4× test set
                                    # at modest noise risk (some hand_not_a_site sites may include unrelated neighbors).
EARTH_R = 6_371_000.0
RNG = np.random.default_rng(42)


def short_id(prefix: str, *parts) -> str:
    h = hashlib.md5("|".join(str(p) for p in parts).encode()).hexdigest()[:10]
    return f"{prefix}_{h}"


def _osm_only(bldg: pd.DataFrame) -> pd.DataFrame:
    has_osm = bldg["source_datasets"].apply(
        lambda a: a is not None and "OpenStreetMap" in list(a)
    )
    return bldg[has_osm].reset_index(drop=True)


def _row_dict(r, *, class_id: int, source: str, weight: float, site_id) -> dict:
    return dict(
        building_id=short_id(source[:4], r.id),
        ovt_id=r.id,
        class_id=class_id,
        source=source,
        weight=weight,
        lat=float(r.lat), lon=float(r.lon),
        xmin=float(r.xmin), xmax=float(r.xmax),
        ymin=float(r.ymin), ymax=float(r.ymax),
        approx_area_m2=float(r.approx_area_m2),
        ovt_class=r.ovt_class,
        ovt_subtype=r.subtype,
        ovt_name=r.name,
        site_id=site_id,
    )


# ---------------------------------------------------------------------------
# Hand-site centroid lookup (lifted from v2_build_dataset.py)
# ---------------------------------------------------------------------------

S2_CHIP_CACHE = ROOT / "sites_us" / ".cache" / "s2_chips"


def load_site_centroids(site_ids: set[str]) -> dict[str, tuple[float, float]]:
    """Read each hand site's cached S2 GeoTIFF, return {site_id: (lat, lon)}."""
    import rasterio
    from pyproj import Transformer

    out: dict[str, tuple[float, float]] = {}
    missing: list[str] = []
    for sid in sorted(site_ids):
        site_dir = S2_CHIP_CACHE / sid
        if not site_dir.exists():
            missing.append(sid); continue
        chips = sorted(site_dir.glob("*.tif"))
        if not chips:
            missing.append(sid); continue
        with rasterio.open(chips[0]) as ds:
            b = ds.bounds
            cx = (b.left + b.right) / 2
            cy = (b.top + b.bottom) / 2
            tr = Transformer.from_crs(ds.crs, 4326, always_xy=True)
            lon, lat = tr.transform(cx, cy)
        out[sid] = (float(lat), float(lon))
    if missing:
        print(f"  WARN: no cached chip for {len(missing)} hand sites "
              f"(first 5: {missing[:5]})")
    return out


# ---------------------------------------------------------------------------
# Hand-derived per-building rows
# ---------------------------------------------------------------------------

def _site_label(years_labels: pd.DataFrame) -> str | None:
    """Collapse a site's multi-year hand labels to a single per-site label.

    NAIP is one snapshot per state, so we use the most recent labeled year.
    Skips 'unsure' and 'partial'; only 'complete' and 'not_a_site' emit rows."""
    df = years_labels[years_labels["label"].isin(["complete", "not_a_site"])]
    if df.empty:
        return None
    return df.sort_values("year", ascending=False).iloc[0]["label"]


def hand_rows(bldg: pd.DataFrame) -> list[dict]:
    """Per hand-labeled site, find Overture buildings within HAND_RADIUS_M of
    site centroid and assign the site's label. complete → class_id=2,
    not_a_site → class_id=0."""
    labels = pd.read_parquet(MANUAL_LABELS)
    site_labels = {}
    for sid, grp in labels.groupby("site_id"):
        lab = _site_label(grp)
        if lab is not None:
            site_labels[sid] = lab
    centroids = load_site_centroids(set(site_labels.keys()))
    print(f"  hand sites with centroids: {len(centroids)}/{len(site_labels)}")

    sites = [(sid, lab, *centroids[sid]) for sid, lab in site_labels.items()
             if sid in centroids]
    if not sites:
        return []

    bldg_rad = np.radians(bldg[["lat", "lon"]].to_numpy())
    site_rad = np.radians(np.array([[la, lo] for _, _, la, lo in sites]))
    tree = BallTree(bldg_rad, metric="haversine")
    radius_rad = HAND_RADIUS_M / EARTH_R
    nbrs = tree.query_radius(site_rad, r=radius_rad)

    rows = []
    by_label = {"complete": 0, "not_a_site": 0}
    for (sid, lab, _, _), idx in zip(sites, nbrs):
        if len(idx) == 0:
            continue
        class_id = 2 if lab == "complete" else 0
        source = f"hand_{lab}"
        for r in bldg.iloc[idx].itertuples(index=False):
            rows.append(_row_dict(r, class_id=class_id, source=source,
                                  weight=5.0, site_id=sid))
            by_label[lab] += 1
    print(f"  hand-derived buildings: complete={by_label['complete']} "
          f"not_a_site={by_label['not_a_site']}")
    return rows


# ---------------------------------------------------------------------------
# OSM-industrial positives (component 2)
# ---------------------------------------------------------------------------

def osm_industrial_rows(bldg: pd.DataFrame, hand_ovt_ids: set[str],
                        test_ovt_ids: set[str]) -> list[dict]:
    keep = (bldg["ovt_class"].isin(OSM_INDUSTRIAL_CLASSES)
            | (bldg["subtype"] == "industrial"))
    ind = bldg[keep & (bldg["approx_area_m2"] >= AREA_FLOOR_M2)].reset_index(drop=True)
    print(f"  industrial after class+area filter: {len(ind):,}")
    ind = _osm_only(ind)
    print(f"  after OSM-source filter: {len(ind):,}")
    # Don't include buildings already used as hand rows (they're test-only;
    # bulk training rows would create test/train contamination).
    ind = ind[~ind["id"].isin(hand_ovt_ids)].reset_index(drop=True)
    print(f"  after hand-set exclusion: {len(ind):,}")
    ind = ind[~ind["id"].isin(test_ovt_ids)].reset_index(drop=True)
    print(f"  after osm-test-set exclusion: {len(ind):,}")

    # 500m haversine dedup so dense industrial corridors don't dominate.
    tree = BallTree(np.radians(ind[["lat", "lon"]].values), metric="haversine")
    keep_mask = np.ones(len(ind), dtype=bool)
    rad = OSM_DEDUP_M / EARTH_R
    for i in range(len(ind)):
        if not keep_mask[i]:
            continue
        q = np.radians(ind.iloc[[i]][["lat", "lon"]].values)
        for j in tree.query_radius(q, r=rad)[0]:
            if j != i:
                keep_mask[j] = False
    ind = ind[keep_mask].reset_index(drop=True)
    print(f"  after {OSM_DEDUP_M}m dedup: {len(ind):,}")

    if len(ind) > OSM_INDUSTRIAL_TARGET:
        ind = ind.sample(n=OSM_INDUSTRIAL_TARGET, random_state=42).reset_index(drop=True)
    return [_row_dict(r, class_id=2, source="osm_industrial",
                      weight=1.0, site_id=None)
            for r in ind.itertuples(index=False)]


# ---------------------------------------------------------------------------
# OSM-categorical negatives (component 3)
# ---------------------------------------------------------------------------

def osm_neg_rows(bldg: pd.DataFrame, hand_ovt_ids: set[str],
                 test_ovt_ids: set[str], n: int) -> list[dict]:
    neg = bldg[bldg["ovt_class"].isin(OSM_NEG_CLASSES)
               & (bldg["approx_area_m2"] >= AREA_FLOOR_M2)].reset_index(drop=True)
    print(f"  OSM-neg after class+area filter: {len(neg):,}")
    neg = _osm_only(neg)
    print(f"  after OSM-source filter: {len(neg):,}")
    neg = neg[~neg["id"].isin(hand_ovt_ids)].reset_index(drop=True)
    print(f"  after hand-set exclusion: {len(neg):,}")
    neg = neg[~neg["id"].isin(test_ovt_ids)].reset_index(drop=True)
    print(f"  after osm-test-set exclusion: {len(neg):,}")
    if len(neg) > n:
        neg = neg.sample(n=n, random_state=43).reset_index(drop=True)
    print(f"  class distribution of sampled negatives:")
    print(neg["ovt_class"].value_counts().head(20).to_string().replace("\n", "\n    "))
    return [_row_dict(r, class_id=0, source="osm_neg",
                      weight=1.0, site_id=None)
            for r in neg.itertuples(index=False)]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _load_conus_polygon():
    """CONUS land polygon (lower-48 + DC) for filtering cross-border Overture rows."""
    import json
    from shapely.geometry import shape
    from shapely.ops import unary_union
    fc = json.loads(CONUS_GEOJSON.read_text())
    geoms = [shape(f["geometry"]) for f in fc["features"]
             if f["properties"].get("name") not in NON_CONUS_STATES]
    return unary_union(geoms)


def main() -> None:
    print(f"[v3-manifest] loading Overture ({OVERTURE.name})...")
    bldg = pd.read_parquet(
        OVERTURE,
        columns=["id", "lon", "lat", "xmin", "xmax", "ymin", "ymax",
                 "approx_area_m2", "class", "subtype", "name", "source_datasets"],
    )
    # itertuples mangles 'class' (Python keyword) — rename upstream.
    bldg = bldg.rename(columns={"class": "ovt_class"})
    # Area floor first — by far the cheapest filter, drops ~95% of rows.
    bldg = bldg[bldg["approx_area_m2"] >= AREA_FLOOR_M2].reset_index(drop=True)
    print(f"[v3-manifest] overture rows >= {AREA_FLOOR_M2:.0f} m^2: {len(bldg):,}")

    # CONUS land-polygon filter (drops Canada/Mexico bleed).
    print("[v3-manifest] applying CONUS land filter...")
    from shapely import STRtree
    from shapely.geometry import Point
    conus = _load_conus_polygon()
    pts = [Point(lo, la) for la, lo in zip(bldg["lat"].to_numpy(), bldg["lon"].to_numpy())]
    inside_ix = STRtree([conus]).query(pts, predicate="intersects")[0]
    keep = np.zeros(len(bldg), dtype=bool); keep[inside_ix] = True
    bldg = bldg[keep].reset_index(drop=True)
    print(f"[v3-manifest] after CONUS filter: {len(bldg):,}")

    print("[v3-manifest] deriving hand-label rows...")
    hand = hand_rows(bldg)
    hand_ovt_ids = {h["ovt_id"] for h in hand}
    print(f"[v3-manifest] hand rows: {len(hand)} ({len(hand_ovt_ids)} unique buildings)")

    test_ovt_ids: set[str] = set()
    if TEST_SET.exists():
        t = pd.read_parquet(TEST_SET, columns=["ovt_id"])
        test_ovt_ids = set(t["ovt_id"].astype(str).tolist())
        print(f"[v3-manifest] excluding {len(test_ovt_ids):,} test-set ovt_ids "
              f"({TEST_SET.name})")
    else:
        print(f"[v3-manifest] WARN: no test set at {TEST_SET}, skipping exclusion")

    print("[v3-manifest] building OSM-industrial positives...")
    ind = osm_industrial_rows(bldg, hand_ovt_ids, test_ovt_ids)
    print(f"[v3-manifest] +osm_industrial: {len(ind)}")

    print("[v3-manifest] sampling OSM-categorical negatives...")
    neg = osm_neg_rows(bldg, hand_ovt_ids, test_ovt_ids, OSM_NEG_TARGET)
    print(f"[v3-manifest] +osm_neg: {len(neg)}")

    df = pd.DataFrame(hand + ind + neg)
    df = df.drop_duplicates(subset=["building_id"]).reset_index(drop=True)
    df["split"] = np.where(df["source"].str.startswith("hand_"), "test", "train")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)

    print()
    print(f"[v3-manifest] wrote {len(df):,} rows -> {OUT}")
    print()
    print("=== summary ===")
    print(df.groupby(["class_id", "source"]).agg(
        n=("building_id", "count"),
        weighted=("weight", "sum"),
        median_area=("approx_area_m2", "median"),
    ).to_string())
    print()
    print("=== split breakdown ===")
    print(df.groupby(["class_id", "split"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
