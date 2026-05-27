"""Bulk-pull CONUS Overture buildings that are plausibly industrial.

Filter: CONUS bbox AND (subtype IN ('industrial','commercial') OR approx_area_m2 >= 1000)

approx_area_m2 is computed from the bbox struct using equirectangular approx
(good enough at building scale — we only need it as a coarse size filter).

Output: data_us/external/overture_industrial_conus.parquet
Columns: id, lon, lat, bbox_xmin/xmax/ymin/ymax, approx_area_m2, class, subtype
"""

import duckdb
import time
from pathlib import Path

OUT = Path(__file__).parents[2] / "data_us" / "overture_industrial_conus.parquet"
OUT.parent.mkdir(parents=True, exist_ok=True)

CONUS = dict(xmin=-125.0, xmax=-66.5, ymin=24.5, ymax=49.5)

con = duckdb.connect()
con.execute("INSTALL httpfs; LOAD httpfs;")
con.execute("INSTALL spatial; LOAD spatial;")
con.execute("SET s3_region='us-west-2';")

q = f"""
COPY (
    WITH src AS (
        SELECT
            id,
            class,
            subtype,
            names.primary AS name,
            height,
            num_floors,
            list_distinct(list_transform(sources, x -> x.dataset)) AS source_datasets,
            bbox.xmin AS xmin, bbox.xmax AS xmax,
            bbox.ymin AS ymin, bbox.ymax AS ymax,
            (bbox.xmin + bbox.xmax) / 2.0 AS lon,
            (bbox.ymin + bbox.ymax) / 2.0 AS lat
        FROM read_parquet(
            's3://overturemaps-us-west-2/release/2026-04-15.0/theme=buildings/type=building/*.zstd.parquet',
            hive_partitioning=0
        )
        WHERE bbox.xmin >= {CONUS['xmin']} AND bbox.xmax <= {CONUS['xmax']}
          AND bbox.ymin >= {CONUS['ymin']} AND bbox.ymax <= {CONUS['ymax']}
    ),
    sized AS (
        SELECT
            *,
            (xmax - xmin) * 111000.0 * cos(radians((ymin + ymax) / 2.0)) AS w_m,
            (ymax - ymin) * 111000.0 AS h_m
        FROM src
    )
    SELECT
        id, lon, lat,
        xmin, xmax, ymin, ymax,
        (w_m * h_m) AS approx_area_m2,
        class, subtype,
        name, height, num_floors, source_datasets
    FROM sized
    WHERE subtype IN ('industrial', 'commercial')
       OR (w_m * h_m) >= 1000.0
) TO '{OUT}' (FORMAT PARQUET, COMPRESSION ZSTD);
"""

t0 = time.time()
print(f"[fetch_overture] writing -> {OUT}")
con.execute(q)
elapsed = time.time() - t0

n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{OUT}')").fetchone()[0]
size_mb = OUT.stat().st_size / 1e6
print(f"[fetch_overture] done in {elapsed/60:.1f} min  rows={n:,}  size={size_mb:.1f} MB")
