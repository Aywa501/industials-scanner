"""Build the v2 detector training manifest (binary: industrial vs not).

Produces data_us/v2_dataset_manifest.parquet with one row per (location, year)
tile to be embedded. UC/partial class is excluded — v3 will revisit.

Schema:
  tile_id           str    unique row id
  class_id          int    0=non_industrial, 2=industrial_complete (1 reserved, unused)
  source            str    hand_complete|hand_not_a_site|overture_industrial|overture_neg|random_bg
  weight            float  5.0 for hand-labels, 1.0 for bulk
  lat, lon          float  WGS84 centroid for tile
  target_year       int    year for scene selection
  offset_dx_m       float  random offset in metres (Overture positives only; 0 elsewhere)
  offset_dy_m       float
  tile_uri          str    gs:// path if pre-extracted on GCS (hand-labels), else null (must STAC-fetch)
  site_id           str    optional, for traceability of hand-labels
  is_inferred       bool   True for gap-filled hand-label rows
  split             str    'train' or 'test' — split-by-site for hand-labels,
                           random for bulk Overture/random rows; ~15% test, stratified by class_id

class_id keeps the 0/2 numbering (skipping 1) so existing v2 manifests and
embeddings stay compatible. Probe training maps class_id==2 → label 1.

Targets (negative-heavy 2:1 vs total positives):
  complete:  ~30K (rounded from Overture sample)
  non:       ~2 × positives = ~70K
"""

from __future__ import annotations

import math
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import BallTree

ROOT = Path(__file__).resolve().parents[3]
DATA_US = ROOT / "data_us"

MANUAL_LABELS = DATA_US / "manual_labels.parquet"
STAGE1_DATASET = DATA_US / "stage1_dataset.parquet"
OVERTURE = DATA_US / "overture_industrial_conus.parquet"
OUT = DATA_US / "v2_dataset_manifest.parquet"

INDUSTRIAL_CLASSES = {"industrial", "warehouse", "hangar"}  # exclude farm_auxiliary (silos/grain bins)
NEG_OVERTURE_CLASSES = {
    "residential", "retail", "office", "hotel", "school", "apartments",
    "hospital", "university", "church", "house", "detached", "terrace",
}

CONUS = dict(xmin=-125.0, xmax=-66.5, ymin=24.5, ymax=49.5)

TEST_FRAC = 0.15
OVERTURE_TARGET = 30_000        # complete bulk count
RANDOM_BG_TARGET = 35_000       # half of negative pool
OVERTURE_NEG_TARGET = 35_000    # other half
OVERTURE_DEDUP_M = 500
OVERTURE_OFFSET_M = 400         # ±400m random offset for satellite-scan realism
OVERTURE_YEAR = 2024            # Overture snapshot era, summer scenes
RANDOM_BG_YEAR_RANGE = (2023, 2025)
RNG = np.random.default_rng(42)


def short_id(prefix: str, *parts) -> str:
    h = hashlib.md5("|".join(str(p) for p in parts).encode()).hexdigest()[:10]
    return f"{prefix}_{h}"


def load_hand_labels() -> pd.DataFrame:
    """Load hand labels and gap-fill same-label years.

    For each site, label years (year_lo, year_hi) of the same label expand to
    cover all stage1_dataset years between them with the same label. Years
    spanning a label transition (e.g. partial→complete) are NOT filled.
    """
    m = pd.read_parquet(MANUAL_LABELS)
    m = m[m["label"] != "unsure"].copy()
    d = pd.read_parquet(STAGE1_DATASET)[["site_id", "year", "tile_uri"]]

    # build per-site label series (sorted by year)
    rows = []
    for site_id, grp in m.groupby("site_id"):
        labels = grp.sort_values("year")[["year", "label"]].to_records(index=False)
        site_tiles = d[d["site_id"] == site_id].set_index("year")["tile_uri"].to_dict()
        seen = set()

        # primary labels (always kept if a tile exists)
        for year, label in labels:
            if year in site_tiles:
                rows.append(dict(site_id=site_id, year=int(year), label=label,
                                 tile_uri=site_tiles[int(year)], is_inferred=False))
                seen.add(int(year))

        # gap-fill for same-label consecutive pairs
        for (y_lo, lab_lo), (y_hi, lab_hi) in zip(labels, labels[1:]):
            if lab_lo != lab_hi:
                continue
            for y in range(int(y_lo) + 1, int(y_hi)):
                if y in site_tiles and y not in seen:
                    rows.append(dict(site_id=site_id, year=y, label=lab_lo,
                                     tile_uri=site_tiles[y], is_inferred=True))
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
            lat=np.nan, lon=np.nan,
            target_year=int(r.year),
            offset_dx_m=0.0, offset_dy_m=0.0,
            tile_uri=r.tile_uri,
            site_id=r.site_id,
            is_inferred=bool(r.is_inferred),
        ))
    return rows


def overture_industrial_centroids() -> pd.DataFrame:
    bldg = pd.read_parquet(OVERTURE, columns=["lon", "lat", "approx_area_m2", "class", "subtype"])
    keep = (
        bldg["class"].isin(INDUSTRIAL_CLASSES)
        | (bldg["subtype"] == "industrial")
    )
    bldg = bldg[keep].reset_index(drop=True)
    print(f"  industrial-class candidate buildings: {len(bldg):,}")

    # facility-sized area threshold; keep singletons (no cluster requirement)
    bldg = bldg[bldg["approx_area_m2"] >= 3000.0].reset_index(drop=True)
    print(f"  after area>=3000 m^2: {len(bldg):,}")

    # global dedup within OVERTURE_DEDUP_M so we don't over-sample dense corridors
    tree = BallTree(np.radians(bldg[["lat", "lon"]].values), metric="haversine")
    keep_mask = np.ones(len(bldg), dtype=bool)
    rad = OVERTURE_DEDUP_M / 6371000.0
    for i in range(len(bldg)):
        if not keep_mask[i]:
            continue
        q = np.radians(bldg.iloc[[i]][["lat", "lon"]].values)
        idx = tree.query_radius(q, r=rad)[0]
        for j in idx:
            if j != i:
                keep_mask[j] = False
    bldg = bldg[keep_mask].reset_index(drop=True)
    print(f"  after {OVERTURE_DEDUP_M}m dedup: {len(bldg):,}")
    return bldg


def overture_industrial_rows() -> list[dict]:
    grp = overture_industrial_centroids()
    if len(grp) > OVERTURE_TARGET:
        grp = grp.sample(n=OVERTURE_TARGET, random_state=42).reset_index(drop=True)

    rows = []
    for r in grp.itertuples(index=False):
        dx = float(RNG.uniform(-OVERTURE_OFFSET_M, OVERTURE_OFFSET_M))
        dy = float(RNG.uniform(-OVERTURE_OFFSET_M, OVERTURE_OFFSET_M))
        # apply offset (approximate metres → degrees)
        lat_off = r.lat + dy / 111000.0
        lon_off = r.lon + dx / (111000.0 * math.cos(math.radians(r.lat)))
        rows.append(dict(
            tile_id=short_id("ov", r.lat, r.lon),
            class_id=2,
            source="overture_industrial",
            weight=1.0,
            lat=float(lat_off), lon=float(lon_off),
            target_year=OVERTURE_YEAR,
            offset_dx_m=dx, offset_dy_m=dy,
            tile_uri=None,
            site_id=None,
            is_inferred=False,
        ))
    return rows


def random_bg_rows(n: int, exclude_lat: np.ndarray, exclude_lon: np.ndarray, exclude_radius_m: float = 1500) -> list[dict]:
    """Random CONUS points, excluded within exclude_radius_m of any positive."""
    tree = BallTree(np.radians(np.column_stack([exclude_lat, exclude_lon])), metric="haversine")
    rad = exclude_radius_m / 6371000.0

    rows = []
    attempts = 0
    while len(rows) < n and attempts < n * 20:
        batch = max(1000, n // 4)
        lons = RNG.uniform(CONUS["xmin"], CONUS["xmax"], batch)
        lats = RNG.uniform(CONUS["ymin"], CONUS["ymax"], batch)
        q = np.radians(np.column_stack([lats, lons]))
        cnt = tree.query_radius(q, r=rad, count_only=True)
        ok = cnt == 0
        for la, lo in zip(lats[ok], lons[ok]):
            if len(rows) >= n:
                break
            year = int(RNG.integers(RANDOM_BG_YEAR_RANGE[0], RANDOM_BG_YEAR_RANGE[1] + 1))
            rows.append(dict(
                tile_id=short_id("rb", la, lo, year),
                class_id=0,
                source="random_bg",
                weight=1.0,
                lat=float(la), lon=float(lo),
                target_year=year,
                offset_dx_m=0.0, offset_dy_m=0.0,
                tile_uri=None,
                site_id=None,
            ))
        attempts += batch
    return rows


def overture_neg_rows(n: int) -> list[dict]:
    bldg = pd.read_parquet(OVERTURE, columns=["lon", "lat", "approx_area_m2", "class"])
    bldg = bldg[bldg["class"].isin(NEG_OVERTURE_CLASSES) & (bldg["approx_area_m2"] >= 1000)]
    if len(bldg) > n:
        bldg = bldg.sample(n=n, random_state=43).reset_index(drop=True)
    rows = []
    for r in bldg.itertuples(index=False):
        year = int(RNG.integers(RANDOM_BG_YEAR_RANGE[0], RANDOM_BG_YEAR_RANGE[1] + 1))
        rows.append(dict(
            tile_id=short_id("ovn", r.lat, r.lon, year),
            class_id=0,
            source="overture_neg",
            weight=1.0,
            lat=float(r.lat), lon=float(r.lon),
            target_year=year,
            offset_dx_m=0.0, offset_dy_m=0.0,
            tile_uri=None,
            site_id=None,
            is_inferred=False,
        ))
    return rows


SPLIT_RNG = np.random.default_rng(123)


def _split_by_group(group_keys: pd.Series, test_frac: float) -> pd.Series:
    """Assign each unique key to train/test; return same-length Series of labels."""
    uniq = group_keys.dropna().unique()
    test_set = set(SPLIT_RNG.choice(uniq, size=max(1, int(len(uniq) * test_frac)), replace=False))
    return group_keys.apply(lambda k: "test" if (pd.notna(k) and k in test_set) else "train")


def assign_splits(df: pd.DataFrame) -> pd.Series:
    """Assign 'train' or 'test' per row.

    - hand_*: split by site_id (multi-year tiles of a site go together)
    - overture_industrial / overture_neg / random_bg: random split per row, stratified by class
    Test fraction = TEST_FRAC, applied per-stratum so each class has held-out tiles.
    """
    out = pd.Series(index=df.index, dtype="object")

    # group-based split for hand-labels
    hand_mask = df["source"].str.startswith("hand_")
    out.loc[hand_mask] = _split_by_group(df.loc[hand_mask, "site_id"], TEST_FRAC).values

    # random split per class for bulk rows
    bulk_mask = ~hand_mask
    for cid in df.loc[bulk_mask, "class_id"].unique():
        sel = bulk_mask & (df["class_id"] == cid)
        idx = df.index[sel].to_numpy()
        n_test = int(round(len(idx) * TEST_FRAC))
        test_idx = SPLIT_RNG.choice(idx, size=n_test, replace=False)
        out.loc[idx] = "train"
        out.loc[test_idx] = "test"
    return out


def main() -> None:
    print("[v2-manifest] loading hand labels...")
    hand = load_hand_labels()
    print(f"  manual_labels with tile_uri: {len(hand)}")

    rows = []
    # partial labels stay out of the manifest but still inform gap-fill in
    # load_hand_labels (a partial year correctly blocks a complete→complete fill).
    rows.extend(hand_class_rows(hand, "complete", 2, "hand_complete"))
    rows.extend(hand_class_rows(hand, "not_a_site", 0, "hand_not_a_site"))
    n_hand = len(rows)
    print(f"[v2-manifest] hand rows: {n_hand}")

    print("[v2-manifest] building Overture industrial centroids...")
    rows.extend(overture_industrial_rows())
    n_after_ov = len(rows)
    print(f"[v2-manifest] +overture_industrial: {n_after_ov - n_hand}")

    print("[v2-manifest] sampling random CONUS background...")
    pos_lat = np.array([r["lat"] for r in rows if r["class_id"] != 0 and not np.isnan(r["lat"])])
    pos_lon = np.array([r["lon"] for r in rows if r["class_id"] != 0 and not np.isnan(r["lon"])])
    rows.extend(random_bg_rows(RANDOM_BG_TARGET, pos_lat, pos_lon))
    n_after_rb = len(rows)
    print(f"[v2-manifest] +random_bg: {n_after_rb - n_after_ov}")

    print("[v2-manifest] sampling Overture excluded-class negatives...")
    rows.extend(overture_neg_rows(OVERTURE_NEG_TARGET))
    n_after_on = len(rows)
    print(f"[v2-manifest] +overture_neg: {n_after_on - n_after_rb}")

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
