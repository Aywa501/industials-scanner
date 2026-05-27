"""Stage 2b v3 — NLCD Annual Fractional Impervious Surface filter.

Per candidate, read 2008 + 2022 NLCD FctImp inside the Overture polygon. Drop
candidates that were already impervious in 2008 (pre-existing sites).

Data: s3://usgs-landcover/annual-nlcd/c1/v0/cu/mosaic/Annual_NLCD_FctImp_{YYYY}_CU_C1V0.tif
  - 30 m UINT8 [0, 100] percent impervious, NoData=250, Albers Equal Area WGS84.
  - Cloud-Optimized GeoTIFF; usgs-landcover is requester-pays, us-west-2.

Reads:
  data_us/phase2/v3/stage3_candidates_v3.parquet            (344K rows)
  data_us/phase2/v3/stage2_candidate_polygons.parquet       (ovt_id, geometry_wkb)
Writes:
  data_us/phase2/v3/stage2b_nlcd_chunks{RUN_TAG}/chunk_XXXXX.parquet

Env knobs:
  STAGE2B_NUM_WORKERS, STAGE2B_CHUNK_SIZE, STAGE2B_S3_BUCKET,
  STAGE2B_INSTANCE_ID, STAGE2B_HEARTBEAT_SEC, STAGE2B_STALL_SEC,
  STAGE2B_CAND_TIMEOUT_SEC, STAGE2B_MIN_PROB, STAGE2B_MAX_PROB,
  STAGE2B_RUN_TAG, STAGE2B_LIMIT, STAGE2B_YEAR_BASELINE, STAGE2B_YEAR_RECENT.
"""
import os
import sys
import time
import json
import socket
import signal
import faulthandler
import traceback
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

# --- Env knobs -------------------------------------------------------------- #

NUM_WORKERS = int(os.environ.get("STAGE2B_NUM_WORKERS", "32"))
CHUNK_SIZE = int(os.environ.get("STAGE2B_CHUNK_SIZE", "500"))
S3_BUCKET = os.environ.get("STAGE2B_S3_BUCKET", "").strip()
INSTANCE_ID = os.environ.get("STAGE2B_INSTANCE_ID", socket.gethostname())
HEARTBEAT_SEC = int(os.environ.get("STAGE2B_HEARTBEAT_SEC", "30"))
STALL_SEC = int(os.environ.get("STAGE2B_STALL_SEC", "300"))
CAND_TIMEOUT_SEC = int(os.environ.get("STAGE2B_CAND_TIMEOUT_SEC", "30"))
MIN_PROB = float(os.environ.get("STAGE2B_MIN_PROB", "0.30"))
MAX_PROB = float(os.environ.get("STAGE2B_MAX_PROB", "1.01"))
RUN_TAG = os.environ.get("STAGE2B_RUN_TAG", "")
LIMIT = int(os.environ.get("STAGE2B_LIMIT", "0"))
YEAR_BASELINE = int(os.environ.get("STAGE2B_YEAR_BASELINE", "2008"))
YEAR_RECENT = int(os.environ.get("STAGE2B_YEAR_RECENT", "2022"))

# GDAL env-var creds only — no AWSSession.
os.environ.setdefault("AWS_REQUEST_PAYER", "requester")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("GDAL_HTTP_MULTIPLEX", "YES")
os.environ.setdefault("GDAL_HTTP_VERSION", "2")
os.environ.setdefault("GDAL_HTTP_TIMEOUT", "30")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "5")
os.environ.setdefault("CPL_VSIL_CURL_USE_HEAD", "NO")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".TIF,.tif")
os.environ.setdefault("VSI_CACHE", "TRUE")
os.environ.setdefault("VSI_CACHE_SIZE", "536870912")  # 512MB per process

# --- Paths ------------------------------------------------------------------ #

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CANDS = os.path.join(ROOT, "..", "data_us/phase2/v3/stage3_candidates_v3.parquet")
POLYS = os.path.join(ROOT, "..", "data_us/phase2/v3/stage2_candidate_polygons.parquet")
CHUNK_DIR = os.path.join(ROOT, "..", f"data_us/phase2/v3/stage2b_nlcd_chunks{RUN_TAG}")
STATS_LOG = os.path.join(CHUNK_DIR, "_stats.jsonl")
HEARTBEAT_PATH = os.path.join(CHUNK_DIR, "_heartbeat.json")

NLCD_URI_TEMPLATE = "/vsis3/usgs-landcover/annual-nlcd/c1/v0/cu/mosaic/Annual_NLCD_FctImp_{year}_CU_C1V0.tif"
URI_BASELINE = NLCD_URI_TEMPLATE.format(year=YEAR_BASELINE)
URI_RECENT = NLCD_URI_TEMPLATE.format(year=YEAR_RECENT)

# --- Algorithm constants ---------------------------------------------------- #

MARGIN_M = 60.0       # ~2 NLCD pixels at 30m around polygon bbox
NODATA = 250
MAX_BBOX_SIDE_M = 5000  # NLCD is robust at any scale; only filter truly absurd polygons

# --- Worker globals (populated in _worker_init) ----------------------------- #

_SRC_BASELINE = None
_SRC_RECENT = None
_TRANSFORMER = None


def _lazy_imports():
    global rasterio, from_bounds, transform_bounds, rasterize, wkb_loads, \
           shapely_transform, Transformer, psutil
    import rasterio
    from rasterio.windows import from_bounds
    from rasterio.warp import transform_bounds
    from rasterio.features import rasterize
    from shapely.wkb import loads as wkb_loads
    from shapely.ops import transform as shapely_transform
    from pyproj import Transformer
    try:
        import psutil
    except ImportError:
        psutil = None


# --- Per-candidate logic --------------------------------------------------- #

def _read_fimp(src, geom_aea):
    """Read FctImp window covering polygon, rasterize polygon as mask, return valid pixel values."""
    minx, miny, maxx, maxy = geom_aea.bounds
    win = from_bounds(minx - MARGIN_M, miny - MARGIN_M, maxx + MARGIN_M, maxy + MARGIN_M,
                     transform=src.transform).round_offsets().round_lengths()
    if win.width <= 0 or win.height <= 0:
        return None, 0, "window empty"
    arr = src.read(1, window=win, boundless=True, fill_value=NODATA)
    wb = rasterio.windows.bounds(win, src.transform)
    target_transform = rasterio.transform.from_origin(wb[0], wb[3], 30, 30)
    mask = rasterize([(geom_aea, 1)], out_shape=arr.shape, transform=target_transform,
                     fill=0, dtype=np.uint8, all_touched=True).astype(bool)
    valid = (arr != NODATA) & mask
    return arr[valid], int(mask.sum()), None


def _process_one(row):
    polygon_wkb = row.get("geometry_wkb")
    if polygon_wkb is None:
        return {"building_id": row["building_id"], "error": "no polygon"}

    try:
        poly = wkb_loads(polygon_wkb)
    except Exception as e:
        return {"building_id": row["building_id"], "error": f"poly: {type(e).__name__}"}

    minx, miny, maxx, maxy = poly.bounds
    side_lat_m = (maxy - miny) * 111_000.0
    side_lon_m = (maxx - minx) * 111_000.0 * np.cos(np.radians((miny + maxy) / 2))
    if max(side_lat_m, side_lon_m) > MAX_BBOX_SIDE_M:
        return {"building_id": row["building_id"], "error": "polygon too large"}

    try:
        geom_aea = shapely_transform(_TRANSFORMER.transform, poly)
    except Exception as e:
        return {"building_id": row["building_id"], "error": f"reproject: {type(e).__name__}"}

    out = {"building_id": row["building_id"]}

    # Baseline year
    t0 = time.time()
    try:
        vals_b, n_mask, err_b = _read_fimp(_SRC_BASELINE, geom_aea)
    except Exception as e:
        return {**out, "error": f"baseline read: {type(e).__name__}: {str(e)[:80]}"}
    out["fetch_baseline"] = time.time() - t0
    out["n_mask_pixels"] = n_mask
    out["baseline_year"] = YEAR_BASELINE
    if err_b:
        return {**out, "error": f"baseline: {err_b}"}
    if len(vals_b) == 0:
        return {**out, "error": "baseline: no valid pixels (all NoData or empty mask)"}
    out["fimp_baseline_n"] = len(vals_b)
    out["fimp_baseline_mean"] = float(vals_b.mean())
    out["fimp_baseline_median"] = float(np.median(vals_b))
    out["fimp_baseline_max"] = int(vals_b.max())
    out["fimp_baseline_p25"] = float(np.percentile(vals_b, 25))
    out["fimp_baseline_p75"] = float(np.percentile(vals_b, 75))

    # Recent year
    t0 = time.time()
    try:
        vals_r, _, err_r = _read_fimp(_SRC_RECENT, geom_aea)
    except Exception as e:
        return {**out, "error": f"recent read: {type(e).__name__}: {str(e)[:80]}"}
    out["fetch_recent"] = time.time() - t0
    out["recent_year"] = YEAR_RECENT
    if err_r:
        return {**out, "error": f"recent: {err_r}"}
    if len(vals_r) == 0:
        return {**out, "error": "recent: no valid pixels"}
    out["fimp_recent_n"] = len(vals_r)
    out["fimp_recent_mean"] = float(vals_r.mean())
    out["fimp_recent_median"] = float(np.median(vals_r))
    out["fimp_recent_max"] = int(vals_r.max())
    out["fimp_recent_p25"] = float(np.percentile(vals_r, 25))
    out["fimp_recent_p75"] = float(np.percentile(vals_r, 75))

    return out


# --- Worker entrypoint ----------------------------------------------------- #

def _worker_init():
    global _SRC_BASELINE, _SRC_RECENT, _TRANSFORMER
    _lazy_imports()
    faulthandler.enable()
    # Albers Equal Area WGS84 (matches NLCD CRS exactly)
    AEA_PROJ = ("+proj=aea +lat_1=29.5 +lat_2=45.5 +lat_0=23 +lon_0=-96 "
                "+x_0=0 +y_0=0 +datum=WGS84 +units=m")
    _TRANSFORMER = Transformer.from_crs("EPSG:4326", AEA_PROJ, always_xy=True)
    _SRC_BASELINE = rasterio.open(URI_BASELINE)
    _SRC_RECENT = rasterio.open(URI_RECENT)


def _process_chunk(chunk_id, rows_dict, out_path):
    t_chunk = time.time()
    results = []
    per_cand_t = []
    err_n = 0

    def _alarm_handler(signum, frame):
        raise TimeoutError(f"candidate exceeded {CAND_TIMEOUT_SEC}s")
    signal.signal(signal.SIGALRM, _alarm_handler)

    rows = pd.DataFrame(rows_dict)
    for _, r in rows.iterrows():
        t0 = time.time()
        signal.alarm(CAND_TIMEOUT_SEC)
        try:
            res = _process_one(r)
            res["lat"] = float(r["lat"])
            res["lon"] = float(r["lon"])
            res["ovt_id"] = r.get("ovt_id")
            res["p_dino_sat493m"] = r.get("p_dino_sat493m")
            res["approx_area_m2"] = r.get("approx_area_m2")
            res["ovt_class"] = r.get("ovt_class")
            if "error" in res:
                err_n += 1
        except Exception as e:
            res = {"building_id": r["building_id"],
                   "error": f"exc: {type(e).__name__}: {str(e)[:160]}"}
            err_n += 1
        finally:
            signal.alarm(0)
        per_cand_t.append(time.time() - t0)
        results.append(res)

    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path, index=False)

    rss_mb = None
    try:
        import psutil
        rss_mb = psutil.Process().memory_info().rss / 1024 / 1024
    except Exception:
        pass

    return {
        "chunk_id": chunk_id,
        "n": len(rows),
        "n_err": err_n,
        "t_total_s": time.time() - t_chunk,
        "t_p50_per_cand_s": float(np.median(per_cand_t)) if per_cand_t else 0.0,
        "t_p90_per_cand_s": float(np.percentile(per_cand_t, 90)) if per_cand_t else 0.0,
        "t_fetch_baseline_mean_s": float(np.mean([r.get("fetch_baseline", 0) for r in results])),
        "t_fetch_recent_mean_s": float(np.mean([r.get("fetch_recent", 0) for r in results])),
        "rss_mb": rss_mb,
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "out_path": out_path,
    }


# --- Telemetry helpers ----------------------------------------------------- #

def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _heartbeat(state):
    elapsed = state["elapsed"]
    cands_done = state["cands_done"]
    cands_total = state["cands_total"]
    rate = cands_done / max(elapsed, 1)
    eta_min = (cands_total - cands_done) / max(rate, 0.01) / 60
    hb = {
        "ts": _now_iso(),
        "instance_id": INSTANCE_ID,
        "elapsed_sec": int(elapsed),
        "chunks_done": state["chunks_done"],
        "chunks_total": state["chunks_total"],
        "cands_done": cands_done,
        "cands_total": cands_total,
        "rate_cand_s": round(rate, 3),
        "eta_min": round(eta_min, 1),
        "err_n": state["err_n"],
        "err_rate": round(state["err_n"] / max(cands_done, 1), 5),
        "mem_rss_gb": round(state.get("rss_gb", 0), 3),
        "mean_per_cand_s": round(state.get("p50_per_cand", 0), 3),
        "mean_fetch_baseline_s": round(state.get("fetch_baseline", 0), 3),
        "mean_fetch_recent_s": round(state.get("fetch_recent", 0), 3),
        "n_workers": NUM_WORKERS,
    }
    with open(HEARTBEAT_PATH, "w") as f:
        json.dump(hb, f, indent=2)
    if S3_BUCKET:
        os.system(
            f"aws s3 cp {HEARTBEAT_PATH} "
            f"s3://{S3_BUCKET}/v3-pt4-artifacts/heartbeat/{INSTANCE_ID}.json "
            f"--only-show-errors 2>/dev/null &"
        )
    return hb


def _dump_stacks(reason):
    path = "/tmp/stage2b_nlcd_stacks.txt"
    with open(path, "w") as f:
        f.write(f"# stage2b-nlcd stack dump @ {_now_iso()}  reason={reason}\n")
        faulthandler.dump_traceback(file=f, all_threads=True)
    print(f"[stage2b-nlcd] stall dump -> {path}", flush=True)
    if S3_BUCKET:
        os.system(
            f"aws s3 cp {path} s3://{S3_BUCKET}/v3-pt4-artifacts/stacks/{INSTANCE_ID}.txt "
            f"--only-show-errors 2>/dev/null"
        )


# --- Main ------------------------------------------------------------------ #

def main():
    print(f"[stage2b-nlcd] start @ {_now_iso()}  host={socket.gethostname()} workers={NUM_WORKERS}")
    print(f"[stage2b-nlcd] year_baseline={YEAR_BASELINE}  year_recent={YEAR_RECENT}")
    print(f"[stage2b-nlcd] chunk_size={CHUNK_SIZE}  s3_bucket={S3_BUCKET or '(none)'}  run_tag={RUN_TAG or '(none)'}")

    for p in (CANDS, POLYS):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}")

    df = pd.read_parquet(CANDS)
    print(f"[stage2b-nlcd] candidates raw: {len(df):,}")
    if "p_dino_sat493m" in df.columns:
        before = len(df)
        df = df[(df["p_dino_sat493m"] >= MIN_PROB) & (df["p_dino_sat493m"] < MAX_PROB)].reset_index(drop=True)
        print(f"[stage2b-nlcd] p_dino in [{MIN_PROB}, {MAX_PROB}): {before:,} -> {len(df):,}")

    polys = pd.read_parquet(POLYS)
    df = df.merge(polys[["ovt_id", "geometry_wkb"]], on="ovt_id", how="left")
    n_poly = df["geometry_wkb"].notna().sum()
    print(f"[stage2b-nlcd] polygons attached: {n_poly:,}/{len(df):,} ({100*n_poly/len(df):.1f}%)")

    if LIMIT > 0:
        df = df.sample(n=min(LIMIT, len(df)), random_state=7).reset_index(drop=True)
        print(f"[stage2b-nlcd] LIMIT={LIMIT} -> sampled {len(df)} candidates")

    # Sort by 1-deg cell for VSI cache locality
    df["_lat_idx"] = np.floor(df["lat"].values).astype(int)
    df["_lon_idx"] = np.floor(df["lon"].values).astype(int)
    df = df.sort_values(["_lat_idx", "_lon_idx"]).reset_index(drop=True)

    os.makedirs(CHUNK_DIR, exist_ok=True)
    n_chunks = (len(df) + CHUNK_SIZE - 1) // CHUNK_SIZE
    existing = set()
    for f in os.listdir(CHUNK_DIR):
        if f.startswith("chunk_") and f.endswith(".parquet"):
            try:
                existing.add(int(f[len("chunk_"):-len(".parquet")]))
            except ValueError:
                pass
    pending = [i for i in range(n_chunks) if i not in existing]
    print(f"[stage2b-nlcd] chunks: total={n_chunks}  done={len(existing)}  pending={len(pending)}")
    if not pending:
        print("[stage2b-nlcd] all chunks already done")
        return

    t_start = time.time()
    cands_done = 0
    err_n = 0
    chunks_done = 0
    last_chunk_t = time.time()
    last_hb = 0.0
    rolling = {"fetch_baseline": [], "fetch_recent": [], "p50_per_cand": [], "rss_gb": 0.0}

    futs = {}
    with ProcessPoolExecutor(max_workers=NUM_WORKERS,
                              initializer=_worker_init) as pool:
        for i in pending:
            start = i * CHUNK_SIZE
            end = min(start + CHUNK_SIZE, len(df))
            chunk = df.iloc[start:end].drop(columns=["_lat_idx", "_lon_idx"]).to_dict(orient="records")
            chunk_path = os.path.join(CHUNK_DIR, f"chunk_{i:05d}.parquet")
            futs[pool.submit(_process_chunk, i, chunk, chunk_path)] = i

        for fut in as_completed(futs):
            chunk_id = futs[fut]
            try:
                tel = fut.result()
            except Exception as e:
                print(f"[stage2b-nlcd] chunk {chunk_id} CRASH: {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
                continue

            chunks_done += 1
            cands_done += tel["n"]
            err_n += tel["n_err"]
            last_chunk_t = time.time()
            rolling["fetch_baseline"].append(tel["t_fetch_baseline_mean_s"])
            rolling["fetch_recent"].append(tel["t_fetch_recent_mean_s"])
            rolling["p50_per_cand"].append(tel["t_p50_per_cand_s"])
            if tel.get("rss_mb"):
                rolling["rss_gb"] = max(rolling["rss_gb"], tel["rss_mb"] / 1024)

            with open(STATS_LOG, "a") as sf:
                sf.write(json.dumps({"ts": _now_iso(), **tel}) + "\n")

            elapsed = time.time() - t_start
            rate = cands_done / max(elapsed, 1)
            remaining = (len(pending) - chunks_done) * CHUNK_SIZE
            eta_min = remaining / max(rate, 0.01) / 60
            print(
                f"[stage2b-nlcd] chunk {chunk_id:>5}  done={chunks_done}/{len(pending)}  "
                f"cand={cands_done:,}  err={err_n}  rate={rate:.1f}/s  eta={eta_min:.0f}m  "
                f"p50={tel['t_p50_per_cand_s']:.2f}s  rss={rolling['rss_gb']:.1f}G",
                flush=True,
            )

            if S3_BUCKET:
                os.system(
                    f"aws s3 cp {tel['out_path']} "
                    f"s3://{S3_BUCKET}/v3-pt4-artifacts/stage2b_nlcd_chunks{RUN_TAG}/ "
                    f"--only-show-errors 2>/dev/null &"
                )

            now = time.time()
            if now - last_hb >= HEARTBEAT_SEC:
                _heartbeat({
                    "elapsed": elapsed, "cands_done": cands_done,
                    "cands_total": len(pending) * CHUNK_SIZE,
                    "chunks_done": chunks_done, "chunks_total": len(pending),
                    "err_n": err_n, "rss_gb": rolling["rss_gb"],
                    "p50_per_cand": float(np.mean(rolling["p50_per_cand"][-32:])) if rolling["p50_per_cand"] else 0,
                    "fetch_baseline": float(np.mean(rolling["fetch_baseline"][-32:])) if rolling["fetch_baseline"] else 0,
                    "fetch_recent": float(np.mean(rolling["fetch_recent"][-32:])) if rolling["fetch_recent"] else 0,
                })
                last_hb = now

            if now - last_chunk_t > STALL_SEC:
                _dump_stacks(f"no chunk completed in {STALL_SEC}s")
                last_chunk_t = now

    print(f"[stage2b-nlcd] DONE  chunks_done={chunks_done}  cand_done={cands_done:,}  err={err_n}  "
          f"elapsed={(time.time() - t_start)/60:.1f}m", flush=True)

    _heartbeat({
        "elapsed": time.time() - t_start, "cands_done": cands_done,
        "cands_total": len(pending) * CHUNK_SIZE,
        "chunks_done": chunks_done, "chunks_total": len(pending),
        "err_n": err_n, "rss_gb": rolling["rss_gb"],
        "p50_per_cand": float(np.mean(rolling["p50_per_cand"])) if rolling["p50_per_cand"] else 0,
        "fetch_baseline": float(np.mean(rolling["fetch_baseline"])) if rolling["fetch_baseline"] else 0,
        "fetch_recent": float(np.mean(rolling["fetch_recent"])) if rolling["fetch_recent"] else 0,
    })


if __name__ == "__main__":
    main()
