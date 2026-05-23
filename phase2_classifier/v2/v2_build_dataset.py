"""Build the v2 detector training manifest (binary: industrial vs not).

Produces data_us/v2_dataset_manifest.parquet with one row per (location, year)
tile to be embedded.

Composition (3 sources):
  1. hand-labeled — site-tagged manual_labels.parquet (multi-year, gap-filled)
  2. OSM-industrial positives — Overture rows with class∈{industrial, warehouse,
     hangar} or subtype=industrial, sourced from OpenStreetMap, ≥3000 m², 500m
     dedup, ±400m centroid jitter
  3. OSM-categorical negatives — Overture rows tagged in CONFIDENT non-industrial
     OSM classes (residential / education / medical / religious / hotel / civic
     / stadium). Excludes ambiguous classes (commercial / retail / office /
     farm_auxiliary / parking / etc.) where industrial sites can be mis-tagged

Empty-area (no-building) negatives are NOT in this manifest. Instead, stage 3
v2 (the national scan) skips tiles with no nearby Overture footprint at scan
time — Overture is reliable enough on industrial-scale footprints that
empty-tile rejection is safe upstream of the model rather than a training
signal it has to learn.

Year alignment: imagery year (DATA_YEAR) and the Overture source release MUST
match. All bulk rows are stamped target_year=DATA_YEAR; hand-labels keep their
original years (multi-year by design). Supply an Overture release contemporary
with the imagery before regenerating.

Schema:
  tile_id           str    unique row id
  class_id          int    0=non_industrial, 2=industrial_complete (1 reserved)
  source            str    hand_complete|hand_not_a_site|osm_industrial|osm_neg
  weight            float  5.0 for hand-labels (inert: hand is held-out test only),
                           1.0 for osm_industrial / osm_neg
  lat, lon          float  WGS84 centroid for tile
  target_year       int    year for scene selection
  offset_dx_m       float  random offset in metres (osm_industrial only; 0 elsewhere)
  offset_dy_m       float
  tile_uri          str    gs:// path if pre-extracted on GCS (hand-labels), else null
  site_id           str    optional, for traceability of hand-labels
  is_inferred       bool   True for gap-filled hand-label rows
  split             str    'train' for all bulk rows (osm_industrial, osm_neg);
                           'test' for ALL hand rows (held-out gold-standard
                           validation set, never trained on)

class_id keeps 0/2 (skipping 1) so v2 manifests/embeddings stay
schema-compatible with the prior round. Probe training maps class_id==2 → 1.
"""

from __future__ import annotations

import math
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

ROOT = Path(__file__).resolve().parents[3]
DATA_US = ROOT / "data_us"

MANUAL_LABELS = DATA_US / "manual_labels.parquet"
STAGE1_DATASET = DATA_US / "stage1_dataset.parquet"
OVERTURE = DATA_US / "overture_industrial_conus_2025_aligned.parquet"
OUT = DATA_US / "v2_dataset_manifest.parquet"

# OSM-tagged industrial classes (positives).
OSM_INDUSTRIAL_CLASSES = {"industrial", "warehouse", "hangar"}

# Confidently non-industrial OSM classes. Deliberately excludes ambiguous
# categories: commercial, retail, office, farm_auxiliary, parking, roof,
# greenhouse, storage_tank, manufacture, service, post_office, train_station,
# garages, supermarket, public, bunker, barn, farm.
OSM_NEG_CLASSES = {
    # residential
    "residential", "house", "detached", "terrace", "apartments",
    "semidetached_house", "dormitory",
    # education
    "school", "university", "college", "kindergarten", "library",
    # medical
    "hospital",
    # religious
    "church", "cathedral", "mosque", "synagogue", "religious",
    # other confident non-industrial
    "hotel", "stadium", "grandstand", "fire_station", "civic", "government",
}

CONUS = dict(xmin=-125.0, xmax=-66.5, ymin=24.5, ymax=49.5)

OSM_INDUSTRIAL_TARGET = 30_000      # positive bulk count
OSM_NEG_TARGET = 35_000             # categorical negatives count
OSM_DEDUP_M = 500                   # haversine dedup for industrial positives
OSM_OFFSET_M = 400                  # ±400 m random offset for satellite-scan realism
DATA_YEAR = 2025                    # imagery + Overture snapshot era — must align
RNG = np.random.default_rng(42)


def short_id(prefix: str, *parts) -> str:
    h = hashlib.md5("|".join(str(p) for p in parts).encode()).hexdigest()[:10]
    return f"{prefix}_{h}"


# ---------------------------------------------------------------------------
# Hand-site centroid lookup
# ---------------------------------------------------------------------------

S2_CHIP_CACHE = ROOT / "sites_us" / ".cache" / "s2_chips"


def load_site_centroids(site_ids: set[str]) -> dict[str, tuple[float, float]]:
    """Read each hand site's locally-cached S2 GeoTIFF, transform its bbox
    centroid back to WGS84, and return {site_id: (lat, lon)}.

    Chips were pulled by phase1_prep/pull_s2.py keyed on the site's geocoded
    lat/lng — bbox centroid equals the stored lat/lng (no jitter for anchors).
    Reading the chip is the only locally-available source of truth for hand
    site coordinates; the canonical s2_chip_manifest.parquet lives on GCS.
    """
    import rasterio
    from pyproj import Transformer

    out: dict[str, tuple[float, float]] = {}
    missing: list[str] = []
    for sid in sorted(site_ids):
        site_dir = S2_CHIP_CACHE / sid
        if not site_dir.exists():
            missing.append(sid)
            continue
        # Pick any year's chip — they're co-registered for the site.
        chips = sorted(site_dir.glob("*.tif"))
        if not chips:
            missing.append(sid)
            continue
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
# Hand labels
# ---------------------------------------------------------------------------

def load_hand_labels() -> pd.DataFrame:
    """Load hand labels and gap-fill same-label years.

    Years between two same-label hand labels are interpolated as same-label
    rows (transition years stay un-filled because labels disagree). Partial
    labels are kept here so they correctly block complete→complete fills, but
    are filtered out at hand_class_rows call sites — only complete and
    not_a_site emit manifest rows.
    """
    m = pd.read_parquet(MANUAL_LABELS)
    m = m[m["label"] != "unsure"].copy()
    d = pd.read_parquet(STAGE1_DATASET)[["site_id", "year", "tile_uri"]]

    centroids = load_site_centroids(set(m["site_id"].unique()))

    rows = []
    for site_id, grp in m.groupby("site_id"):
        if site_id not in centroids:
            continue  # skip sites without a cached chip — can't STAC-fetch without lat/lon
        lat, lon = centroids[site_id]
        labels = grp.sort_values("year")[["year", "label"]].to_records(index=False)
        site_tiles = d[d["site_id"] == site_id].set_index("year")["tile_uri"].to_dict()
        seen = set()

        for year, label in labels:
            if year in site_tiles:
                rows.append(dict(site_id=site_id, year=int(year), label=label,
                                 tile_uri=site_tiles[int(year)], is_inferred=False,
                                 lat=lat, lon=lon))
                seen.add(int(year))

        for (y_lo, lab_lo), (y_hi, lab_hi) in zip(labels, labels[1:]):
            if lab_lo != lab_hi:
                continue
            for y in range(int(y_lo) + 1, int(y_hi)):
                if y in site_tiles and y not in seen:
                    rows.append(dict(site_id=site_id, year=y, label=lab_lo,
                                     tile_uri=site_tiles[y], is_inferred=True,
                                     lat=lat, lon=lon))
                    seen.add(y)
    return pd.DataFrame(rows)


def hand_class_rows(j: pd.DataFrame, label: str, class_id: int, source_name: str) -> list[dict]:
    sub = j[j["label"] == label]
    rows = []
    for r in sub.itertuples(index=False):
        rows.append(dict(
            tile_id=short_id("hl", r.site_id, r.year),
            class_id=class_id,
            source=source_name,
            weight=5.0,
            lat=float(r.lat), lon=float(r.lon),
            target_year=int(r.year),
            offset_dx_m=0.0, offset_dy_m=0.0,
            tile_uri=r.tile_uri,
            site_id=r.site_id,
            is_inferred=bool(r.is_inferred),
        ))
    return rows


# ---------------------------------------------------------------------------
# OSM-industrial positives (component 2)
# ---------------------------------------------------------------------------

def _osm_filter(bldg: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows whose Overture lineage includes OpenStreetMap.

    All rows with non-null `class` come from OSM tags in practice (Microsoft
    ML Buildings provides geometry but not classification), but filter
    explicitly for transparency.
    """
    has_osm = bldg["source_datasets"].apply(
        lambda a: a is not None and "OpenStreetMap" in list(a)
    )
    return bldg[has_osm].reset_index(drop=True)


def osm_industrial_centroids() -> pd.DataFrame:
    bldg = pd.read_parquet(OVERTURE,
                           columns=["lon", "lat", "approx_area_m2", "class", "subtype",
                                    "source_datasets"])
    keep = (
        bldg["class"].isin(OSM_INDUSTRIAL_CLASSES)
        | (bldg["subtype"] == "industrial")
    )
    bldg = bldg[keep].reset_index(drop=True)
    print(f"  industrial-class candidate buildings: {len(bldg):,}")

    bldg = _osm_filter(bldg)
    print(f"  after OSM-source filter: {len(bldg):,}")

    bldg = bldg[bldg["approx_area_m2"] >= 3000.0].reset_index(drop=True)
    print(f"  after area>=3000 m^2: {len(bldg):,}")

    # global haversine dedup so dense industrial corridors don't dominate
    tree = BallTree(np.radians(bldg[["lat", "lon"]].values), metric="haversine")
    keep_mask = np.ones(len(bldg), dtype=bool)
    rad = OSM_DEDUP_M / 6371000.0
    for i in range(len(bldg)):
        if not keep_mask[i]:
            continue
        q = np.radians(bldg.iloc[[i]][["lat", "lon"]].values)
        idx = tree.query_radius(q, r=rad)[0]
        for j in idx:
            if j != i:
                keep_mask[j] = False
    bldg = bldg[keep_mask].reset_index(drop=True)
    print(f"  after {OSM_DEDUP_M}m dedup: {len(bldg):,}")
    return bldg


def osm_industrial_rows() -> list[dict]:
    grp = osm_industrial_centroids()
    if len(grp) > OSM_INDUSTRIAL_TARGET:
        grp = grp.sample(n=OSM_INDUSTRIAL_TARGET, random_state=42).reset_index(drop=True)

    rows = []
    for r in grp.itertuples(index=False):
        dx = float(RNG.uniform(-OSM_OFFSET_M, OSM_OFFSET_M))
        dy = float(RNG.uniform(-OSM_OFFSET_M, OSM_OFFSET_M))
        lat_off = r.lat + dy / 111000.0
        lon_off = r.lon + dx / (111000.0 * math.cos(math.radians(r.lat)))
        rows.append(dict(
            tile_id=short_id("osm", r.lat, r.lon),
            class_id=2,
            source="osm_industrial",
            weight=1.0,
            lat=float(lat_off), lon=float(lon_off),
            target_year=DATA_YEAR,
            offset_dx_m=dx, offset_dy_m=dy,
            tile_uri=None,
            site_id=None,
            is_inferred=False,
        ))
    return rows


# ---------------------------------------------------------------------------
# OSM-categorical negatives (component 3)
# ---------------------------------------------------------------------------

def osm_neg_rows(n: int) -> list[dict]:
    bldg = pd.read_parquet(OVERTURE,
                           columns=["lon", "lat", "approx_area_m2", "class",
                                    "source_datasets"])
    bldg = bldg[bldg["class"].isin(OSM_NEG_CLASSES) & (bldg["approx_area_m2"] >= 1000)]
    bldg = _osm_filter(bldg)
    print(f"  OSM-neg candidate buildings: {len(bldg):,}")

    if len(bldg) > n:
        bldg = bldg.sample(n=n, random_state=43).reset_index(drop=True)

    rows = []
    for r in bldg.itertuples(index=False):
        rows.append(dict(
            tile_id=short_id("osmn", r.lat, r.lon, DATA_YEAR),
            class_id=0,
            source="osm_neg",
            weight=1.0,
            lat=float(r.lat), lon=float(r.lon),
            target_year=DATA_YEAR,
            offset_dx_m=0.0, offset_dy_m=0.0,
            tile_uri=None,
            site_id=None,
            is_inferred=False,
        ))
    return rows


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

def assign_splits(df: pd.DataFrame) -> pd.Series:
    """Hand labels are the held-out validation set; everything bulk is train.

    Rationale: hand labels are high-trust and low-volume. They serve as the
    gold-standard validation signal. The OSM bulk rows are abundant but
    lower-purity (residual temporal contamination, OSM tag noise), so they're
    the training pool only.
    """
    return pd.Series(
        np.where(df["source"].str.startswith("hand_"), "test", "train"),
        index=df.index,
        dtype="object",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("[v2-manifest] loading hand labels...")
    hand = load_hand_labels()
    print(f"  manual_labels with tile_uri: {len(hand)}")

    rows = []
    rows.extend(hand_class_rows(hand, "complete", 2, "hand_complete"))
    rows.extend(hand_class_rows(hand, "not_a_site", 0, "hand_not_a_site"))
    n_hand = len(rows)
    print(f"[v2-manifest] hand rows: {n_hand}")

    print("[v2-manifest] building OSM-industrial positives...")
    rows.extend(osm_industrial_rows())
    n_after_pos = len(rows)
    print(f"[v2-manifest] +osm_industrial: {n_after_pos - n_hand}")

    print("[v2-manifest] sampling OSM-categorical negatives...")
    rows.extend(osm_neg_rows(OSM_NEG_TARGET))
    n_after_neg = len(rows)
    print(f"[v2-manifest] +osm_neg: {n_after_neg - n_after_pos}")

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["tile_id"]).reset_index(drop=True)
    df["split"] = assign_splits(df)
    df.to_parquet(OUT, index=False)

    print()
    print(f"[v2-manifest] wrote {len(df):,} rows -> {OUT}")
    print()
    print("=== summary ===")
    print(df.groupby(["class_id", "source"]).agg(
        n=("tile_id", "count"),
        weighted=("weight", "sum"),
    ).to_string())
    print()
    print("class totals (weighted):")
    print(df.groupby("class_id")["weight"].sum().to_string())
    print()
    print("class totals (unweighted):")
    print(df.groupby("class_id").size().to_string())
    print()
    print("=== split breakdown ===")
    print(df.groupby(["class_id", "split"]).size().unstack(fill_value=0).to_string())
    print()
    print(df.groupby(["source", "split"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
