"""Filter overture_industrial_conus.parquet to a 2025-04-23 snapshot.

Why: Sentinel-2 imagery in our manifest is summer 2025 (May 1 – Sep 30). To
guarantee that every building labeled positive ACTUALLY existed throughout the
imagery window, we want a snapshot taken BEFORE the imagery window begins.
The 2025-04-23.0 Overture release is the last release dated before May 1 2025.
Trade-off: we'll miss buildings completed during summer 2025 (no positive
label for them), but those still in the model would be ambiguous mid-window
constructions anyway.

Overture's release/ S3 prefix only retains the two latest releases (currently
2026-03-18.0 and 2026-04-15.0). The fetched data file
(overture_industrial_conus.parquet) is the 2026-04-15 vintage. To reconstruct
the 2025-04-23 building set we walk forward through every monthly changelog
AFTER 2025-04-23, accumulate IDs that appear with change_type=added in CONUS
bbox, then drop those IDs from the current Overture file.

Caveats:
  - Misses buildings that existed in 2025-12-17 but were removed by 2026-04.
    Empirically <1% per Overture release notes; acceptable.
  - Attribute mutations between 2025-12 and 2026-04 are inherited from the
    current release. Fine for our purposes (industrial→warehouse retags don't
    change positive class membership).

Run locally:
    cd sites_us
    python -m phase2_classifier.v2.align_overture_to_2025

Inputs:
    data_us/overture_industrial_conus.parquet   (2026-04-15 vintage)

Output:
    data_us/overture_industrial_conus_2025_aligned.parquet
"""

from __future__ import annotations

import time
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DATA_US = ROOT / "data_us"

CONUS = dict(xmin=-125.0, xmax=-66.5, ymin=24.5, ymax=49.5)

# Changelog releases AFTER our target snapshot date 2025-04-23. Any building
# that appears as 'added' in any of these has its first existence post-2025-04.
POST_TARGET_RELEASES = [
    "2025-05-21.0",
    "2025-07-23.0",   # no June 2025 release
    "2025-08-20.0",
    "2025-08-20.1",   # August hotfix
    "2025-09-24.0",
    "2025-10-22.0",
    "2025-11-19.0",
    "2025-12-17.0",
    "2026-01-21.0",
    "2026-02-18.0",
    "2026-03-18.0",
    "2026-04-15.0",
]

CHANGELOG_TPL = (
    "s3://overturemaps-us-west-2/changelog/{rel}/"
    "theme=buildings/type=building/change_type=added/part-*.zstd.parquet"
)

INPUT = DATA_US / "overture_industrial_conus.parquet"
OUTPUT = DATA_US / "overture_industrial_conus_2025_aligned.parquet"


def fetch_added_ids(con: duckdb.DuckDBPyConnection, release: str) -> set[str]:
    """Read CONUS-bbox added IDs from one changelog release. If the release
    has no `change_type=added` partition for buildings (i.e. no buildings were
    added that month), return an empty set rather than failing.
    """
    path = CHANGELOG_TPL.format(rel=release)
    t0 = time.time()
    try:
        df = con.execute(f"""
            SELECT id
            FROM read_parquet('{path}')
            WHERE bbox.xmin >= {CONUS['xmin']} AND bbox.xmax <= {CONUS['xmax']}
              AND bbox.ymin >= {CONUS['ymin']} AND bbox.ymax <= {CONUS['ymax']}
        """).df()
    except duckdb.IOException as e:
        if "No files found" in str(e):
            print(f"  {release}: no added partition (zero added in this release)")
            return set()
        raise
    print(f"  {release}: {len(df):,} added (CONUS) in {time.time()-t0:.1f}s")
    return set(df["id"].tolist())


def main() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(f"missing {INPUT}; run fetch_overture_industrial.py first")

    print(f"[align-overture] loading current Overture file: {INPUT}")
    cur = pd.read_parquet(INPUT)
    print(f"  rows: {len(cur):,}")

    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("SET s3_region='us-west-2';")

    print(f"[align-overture] collecting added IDs from {len(POST_TARGET_RELEASES)} post-2025-04-23 changelogs...")
    added: set[str] = set()
    for rel in POST_TARGET_RELEASES:
        added |= fetch_added_ids(con, rel)
    print(f"  total unique post-2025-04-23 added IDs (CONUS): {len(added):,}")

    n_before = len(cur)
    keep = ~cur["id"].isin(added)
    aligned = cur[keep].reset_index(drop=True)
    n_dropped = n_before - len(aligned)
    print(f"[align-overture] dropped {n_dropped:,} rows ({n_dropped/n_before*100:.2f}%) "
          f"that didn't exist in 2025-04-23 snapshot")
    print(f"  rows after alignment: {len(aligned):,}")

    aligned.to_parquet(OUTPUT, index=False, compression="zstd")
    print(f"[align-overture] wrote -> {OUTPUT}")
    print(f"  size: {OUTPUT.stat().st_size / 1e6:.1f} MB")

    # Quick sanity: industrial-class breakdown before/after
    before_ind = cur[cur["class"].isin(["industrial", "warehouse", "hangar"])
                     | (cur["subtype"] == "industrial")]
    after_ind = aligned[aligned["class"].isin(["industrial", "warehouse", "hangar"])
                        | (aligned["subtype"] == "industrial")]
    print()
    print(f"[align-overture] industrial buildings: {len(before_ind):,} → {len(after_ind):,} "
          f"({(len(before_ind)-len(after_ind))/max(len(before_ind),1)*100:.2f}% dropped)")


if __name__ == "__main__":
    main()
