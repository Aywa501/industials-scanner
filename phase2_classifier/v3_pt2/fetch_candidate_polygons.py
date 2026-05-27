"""Pull Overture building polygons for the 344K stage2 candidates.

We discarded geometry when building the manifest (kept only bbox). For Stage 2b
bbox-vs-building mismatch, we need the polygon to compute a tight inside mask.

Run on a tiny us-west-2 spot — zero egress to/from S3 within region.

RESUMABLE: each batch writes its own parquet to OUT_DIR (and uploads to
BATCHES_S3 if set). On restart, existing batch files are loaded from S3 and
their id-ranges are skipped. Final concat is written to OUT_PATH + OUT_S3.

Env overrides:
  CANDS_PATH    default: data_us/phase2/v3/stage2_candidates.parquet
  OUT_DIR       default: data_us/phase2/v3/poly_batches
  OUT_PATH      default: data_us/phase2/v3/stage2_candidate_polygons.parquet
  OUT_S3        default: unset; final concat uploaded here
  BATCHES_S3    default: unset; per-batch parquet uploaded here (s3://bucket/prefix/)
  BATCH_SIZE    default: 50000
  DUCKDB_MEM    default: 6GB
"""
import os
import sys
import time
import subprocess
import glob
import re
import pandas as pd
import duckdb

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CANDS = os.environ.get("CANDS_PATH",
    os.path.join(ROOT, "..", "data_us/phase2/v3/stage2_candidates.parquet"))
OUT_DIR = os.environ.get("OUT_DIR",
    os.path.join(ROOT, "..", "data_us/phase2/v3/poly_batches"))
OUT = os.environ.get("OUT_PATH",
    os.path.join(ROOT, "..", "data_us/phase2/v3/stage2_candidate_polygons.parquet"))
OUT_S3 = os.environ.get("OUT_S3", "").strip()
BATCHES_S3 = os.environ.get("BATCHES_S3", "").strip().rstrip("/")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "50000"))
DUCKDB_MEM = os.environ.get("DUCKDB_MEM", "6GB")

OVERTURE_PARQUET = (
    "s3://overturemaps-us-west-2/release/2026-04-15.0/"
    "theme=buildings/type=building/*.zstd.parquet"
)

CONUS = dict(xmin=-125.0, xmax=-66.5, ymin=24.5, ymax=49.5)

BATCH_NAME_RE = re.compile(r"poly_batch_(\d+)_(\d+)\.parquet$")


def _sync_existing_batches():
    """Pull any pre-existing batch parquets from S3 into OUT_DIR."""
    if not BATCHES_S3:
        return
    print(f"[poly] syncing existing batches from {BATCHES_S3}/", flush=True)
    subprocess.run(
        ["aws", "s3", "sync", BATCHES_S3 + "/", OUT_DIR, "--only-show-errors"],
        check=False,
    )


def _completed_ranges():
    """Return set of (start, end) tuples for batch files already on disk."""
    done = set()
    for p in glob.glob(os.path.join(OUT_DIR, "poly_batch_*.parquet")):
        m = BATCH_NAME_RE.search(p)
        if m:
            done.add((int(m.group(1)), int(m.group(2))))
    return done


def main():
    sys.stdout.reconfigure(line_buffering=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    cands = pd.read_parquet(CANDS, columns=["building_id", "ovt_id"])
    ids = cands["ovt_id"].dropna().unique().tolist()
    n_total = len(ids)
    n_batches = (n_total + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"[poly] candidates: {len(cands):,}   unique ovt_id: {n_total:,}", flush=True)
    print(f"[poly] batch_size={BATCH_SIZE}  → {n_batches} batches  duckdb_mem={DUCKDB_MEM}", flush=True)

    _sync_existing_batches()
    done = _completed_ranges()
    print(f"[poly] resumable: {len(done)} of {n_batches} batches already done on disk", flush=True)

    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("INSTALL spatial; LOAD spatial;")
    con.execute("SET s3_region='us-west-2';")
    con.execute("SET enable_progress_bar=false;")
    con.execute(f"SET memory_limit='{DUCKDB_MEM}';")

    t_start = time.time()
    n_skipped = 0
    n_run = 0
    for bi, start in enumerate(range(0, n_total, BATCH_SIZE)):
        end = min(start + BATCH_SIZE, n_total)
        batch_path = os.path.join(OUT_DIR, f"poly_batch_{start:09d}_{end:09d}.parquet")
        if (start, end) in done or os.path.exists(batch_path):
            n_skipped += 1
            continue

        chunk = ids[start:end]
        t0 = time.time()
        con.execute("CREATE OR REPLACE TEMP TABLE want (id VARCHAR)")
        con.executemany("INSERT INTO want VALUES (?)", [(i,) for i in chunk])
        df = con.execute(f"""
            SELECT b.id AS ovt_id, ST_AsWKB(b.geometry) AS geometry_wkb
            FROM read_parquet('{OVERTURE_PARQUET}', hive_partitioning=0) b
            WHERE b.bbox.xmin >= {CONUS['xmin']} AND b.bbox.xmax <= {CONUS['xmax']}
              AND b.bbox.ymin >= {CONUS['ymin']} AND b.bbox.ymax <= {CONUS['ymax']}
              AND b.id IN (SELECT id FROM want)
        """).fetchdf()
        dt = time.time() - t0
        n_run += 1

        # Persist immediately for resumability.
        df.to_parquet(batch_path, index=False)
        if BATCHES_S3:
            subprocess.run(
                ["aws", "s3", "cp", batch_path, f"{BATCHES_S3}/{os.path.basename(batch_path)}",
                 "--only-show-errors"],
                check=False,
            )

        elapsed = time.time() - t_start
        remain = n_batches - (bi + 1)
        eta = (elapsed / n_run) * remain if n_run else 0
        print(f"[poly] batch {bi+1}/{n_batches}: ids {start:,}-{end:,}  matched {len(df):,}  "
              f"({dt:.0f}s)  written={os.path.basename(batch_path)}  "
              f"elapsed {elapsed:.0f}s  ETA {eta:.0f}s",
              flush=True)
        # Free memory between batches.
        del df

    print(f"[poly] batches done. skipped={n_skipped}  ran={n_run}  concatenating...", flush=True)

    # Concat all batches into final output.
    files = sorted(glob.glob(os.path.join(OUT_DIR, "poly_batch_*.parquet")))
    parts = [pd.read_parquet(f) for f in files]
    out_df = pd.concat(parts, ignore_index=True)
    matched = out_df["ovt_id"].nunique()
    print(f"[poly] total matched: {matched:,}/{n_total:,} ({100*matched/n_total:.1f}%)", flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out_df.to_parquet(OUT, index=False)
    sz_mb = os.path.getsize(OUT) / 1024 / 1024
    print(f"[poly] wrote {OUT}: {len(out_df):,} rows, {sz_mb:.1f} MB", flush=True)

    if OUT_S3:
        print(f"[poly] uploading -> {OUT_S3}", flush=True)
        subprocess.check_call(["aws", "s3", "cp", OUT, OUT_S3])
        print("[poly] upload done", flush=True)


if __name__ == "__main__":
    main()
