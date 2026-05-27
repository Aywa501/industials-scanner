"""Stage 2b change scoring — Landsat 2008 vs 2022 within-year-ratio stability.

Reads:
  - data_us/phase2/v3/stage2_candidates.parquet  (~345K rows)
  - data_us/phase2/v3/landsat_scenes_index.parquet
Writes:
  - data_us/phase2/v3/stage2b_change_chunks/chunk_XXXXX.parquet (resumable)

Per candidate:
  1. Lookup pre-indexed scenes by 1-deg grid cell + year.
  2. Fetch L8 2022 pan median composite over bbox + 1px margin.
  3. Derive footprint mask from 2022 via gradient -> close -> fill -> CC at center.
  4. Fetch L7 2008 pan with same window.
  5. change = |ratio_2022 - ratio_2008|  where ratio = mean(pan[mask]) / mean(pan[~mask]).

Telemetry (all to stdout + per-chunk JSONL + S3 heartbeat):
  - per-chunk: chunk_id, n, n_err, t_total, t_p50_per_cand, t_fetch_2022, t_fetch_2008, t_mask, rss_mb
  - heartbeat (S3, every HEARTBEAT_SEC):
      { ts, instance_id, chunks_done, chunks_total, cands_done, cands_total,
        rate_cand_s, eta_min, err_rate, mem_rss_gb, mean_per_cand_s, mean_fetch_s }
  - stall watchdog: if no chunk completes for STALL_SEC, dump all worker stacks
                    to /tmp/stage2b_stacks.txt -> S3.

Resume:
  - Lists existing chunk parquets in CHUNK_DIR on startup, skips those.

Env vars:
  STAGE2B_NUM_WORKERS         default 32  (oversubscribe vCPU; pure I/O)
  STAGE2B_CHUNK_SIZE          default 200 candidates/chunk
  STAGE2B_S3_BUCKET           optional; if set, syncs chunks + telemetry to S3
  STAGE2B_INSTANCE_ID         optional; tags heartbeat (auto-detect on EC2)
  STAGE2B_HEARTBEAT_SEC       default 30
  STAGE2B_STALL_SEC           default 300
  STAGE2B_MIN_FOOTPRINT_PX    default 10 (below -> ambiguous)
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
from multiprocessing import current_process

import numpy as np
import pandas as pd

# --- Env knobs -------------------------------------------------------------- #

NUM_WORKERS = int(os.environ.get("STAGE2B_NUM_WORKERS", "32"))
CHUNK_SIZE = int(os.environ.get("STAGE2B_CHUNK_SIZE", "200"))
S3_BUCKET = os.environ.get("STAGE2B_S3_BUCKET", "").strip()
INSTANCE_ID = os.environ.get("STAGE2B_INSTANCE_ID", socket.gethostname())
HEARTBEAT_SEC = int(os.environ.get("STAGE2B_HEARTBEAT_SEC", "30"))
STALL_SEC = int(os.environ.get("STAGE2B_STALL_SEC", "300"))
MIN_FOOTPRINT_PX = int(os.environ.get("STAGE2B_MIN_FOOTPRINT_PX", "10"))
CAND_TIMEOUT_SEC = int(os.environ.get("STAGE2B_CAND_TIMEOUT_SEC", "30"))
MIN_PROB = float(os.environ.get("STAGE2B_MIN_PROB", "0.30"))
MAX_PROB = float(os.environ.get("STAGE2B_MAX_PROB", "1.01"))
RUN_TAG = os.environ.get("STAGE2B_RUN_TAG", "")

# GDAL knobs — env-var creds only, NO AWSSession (per memory no-awssession).
os.environ.setdefault("AWS_REQUEST_PAYER", "requester")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("GDAL_HTTP_MULTIPLEX", "YES")
os.environ.setdefault("GDAL_HTTP_VERSION", "2")
os.environ.setdefault("GDAL_HTTP_TIMEOUT", "30")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "5")
os.environ.setdefault("CPL_VSIL_CURL_USE_HEAD", "NO")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".TIF,.tif")
os.environ.setdefault("VSI_CACHE", "TRUE")
os.environ.setdefault("VSI_CACHE_SIZE", "134217728")  # 128 MB / worker; tiny bbox reads, large cache wastes RAM

# --- Paths ------------------------------------------------------------------ #

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CANDS = os.path.join(ROOT, "..", "data_us/phase2/v3/stage2_candidates.parquet")
SCENES = os.path.join(ROOT, "..", "data_us/phase2/v3/landsat_scenes_index.parquet")
CHUNK_DIR = os.path.join(ROOT, "..", f"data_us/phase2/v3/stage2b_change_chunks_v3{RUN_TAG}")
STATS_LOG = os.path.join(CHUNK_DIR, "_stats.jsonl")
HEARTBEAT_PATH = os.path.join(CHUNK_DIR, "_heartbeat.json")

# --- Algorithm constants (locked) ------------------------------------------ #

GRID_DEG = 1.0
MARGIN_M = 15           # 1 pan pixel buffer
GRAD_PERCENTILE = 75
SCENES_INDEX_CACHE = None  # populated lazily per worker


# --- Worker-side imports (lazy to keep main fast) -------------------------- #

def _lazy_imports():
    global rasterio, from_bounds, transform_bounds, transform_from_bounds, \
           rasterize, wkb_loads, binary_dilation, psutil
    import rasterio
    from rasterio.windows import from_bounds
    from rasterio.warp import transform_bounds
    from rasterio.transform import from_bounds as transform_from_bounds
    from rasterio.features import rasterize
    from shapely.wkb import loads as wkb_loads
    from scipy.ndimage import binary_dilation
    try:
        import psutil
    except ImportError:
        psutil = None


# --- Per-candidate logic --------------------------------------------------- #

def _expand_bbox_ll(bbox_ll, margin_m):
    cy = (bbox_ll[1] + bbox_ll[3]) / 2
    dlat = margin_m / 111_000
    dlon = margin_m / (111_000 * np.cos(np.radians(cy)))
    return (bbox_ll[0] - dlon, bbox_ll[1] - dlat, bbox_ll[2] + dlon, bbox_ll[3] + dlat)


def _grid_cell(lat, lon, deg=GRID_DEG):
    return int(np.floor(lat / deg)), int(np.floor(lon / deg))


def _fetch_pan_median(scene_hrefs, bbox_ll):
    stack = []
    with rasterio.Env():
        for href in scene_hrefs:
            try:
                with rasterio.open(href) as src:
                    bbox_utm = transform_bounds("EPSG:4326", src.crs, *bbox_ll)
                    win = from_bounds(*bbox_utm, transform=src.transform).round_offsets().round_lengths()
                    if win.width < 4 or win.height < 4:
                        continue
                    arr = src.read(1, window=win, boundless=True, fill_value=0).astype(np.float32)
                arr[arr == 0] = np.nan
                if np.isnan(arr).all():
                    continue
                stack.append(arr)
            except Exception:
                continue
    if not stack:
        return None, 0
    h = min(a.shape[0] for a in stack)
    w = min(a.shape[1] for a in stack)
    return np.nanmedian(np.stack([a[:h, :w] for a in stack], axis=0), axis=0), len(stack)


def _bbox_mask(pan_shape, bbox_ll, expanded_ll):
    h, w = pan_shape
    ex0, ey0, ex1, ey1 = expanded_ll
    bx0, by0, bx1, by1 = bbox_ll
    px0 = max(0, int(round((bx0 - ex0) / (ex1 - ex0) * w)))
    px1 = min(w, int(round((bx1 - ex0) / (ex1 - ex0) * w)))
    py0 = max(0, int(round((ey1 - by1) / (ey1 - ey0) * h)))
    py1 = min(h, int(round((ey1 - by0) / (ey1 - ey0) * h)))
    if px1 <= px0 or py1 <= py0:
        return None
    mask = np.zeros((h, w), dtype=bool)
    mask[py0:py1, px0:px1] = True
    return mask


def _extract_mask(pan, bbox_ll, expanded_ll, polygon_wkb=None):
    """Polygon-rasterized mask + 1px dilation (handles diagonal/non-rectangular
    buildings — the bbox corners that aren't building get excluded). Falls back to
    uniform bbox mask if no polygon is available."""
    if pan is None or np.isnan(pan).all():
        return None
    h, w = pan.shape
    ex0, ey0, ex1, ey1 = expanded_ll

    if polygon_wkb is not None:
        try:
            poly = wkb_loads(polygon_wkb)
            transform = transform_from_bounds(ex0, ey0, ex1, ey1, w, h)
            poly_mask = rasterize(
                [(poly, 1)], out_shape=(h, w), transform=transform,
                fill=0, dtype=np.uint8, all_touched=True,
            ).astype(bool)
            if poly_mask.any():
                poly_mask = binary_dilation(poly_mask, iterations=1)
                if not poly_mask.all():
                    return poly_mask
        except Exception:
            pass  # fall through to bbox mask

    mask = _bbox_mask(pan.shape, bbox_ll, expanded_ll)
    if mask is None or mask.all():
        return None  # bbox fills window — no outside reference
    return mask


def _ratio(pan, mask):
    if pan is None or mask is None or not mask.any():
        return None
    inside = pan[mask & ~np.isnan(pan)]
    outside = pan[~mask & ~np.isnan(pan)]
    if len(inside) == 0 or len(outside) == 0:
        return None
    out_m = float(outside.mean())
    if out_m <= 0:
        return None
    return float(inside.mean()) / out_m


def _process_one(row, scenes_by_cell):
    """Process a single candidate. Returns dict or {'error': ...}.

    scenes_by_cell: { (lat_idx, lon_idx, year): [s3_href, ...] }
    """
    bbox = (row["xmin"], row["ymin"], row["xmax"], row["ymax"])
    expanded = _expand_bbox_ll(bbox, MARGIN_M)
    cell = _grid_cell(row["lat"], row["lon"])

    t = {}
    t0 = time.time()
    hrefs_22 = scenes_by_cell.get((cell[0], cell[1], 2022), [])
    pan_22, n_22 = _fetch_pan_median(hrefs_22, expanded)
    t["fetch_2022"] = time.time() - t0

    if pan_22 is None:
        return {"building_id": row["building_id"], "error": "no 2022 data",
                "n_scenes_2022": n_22, **t}

    t0 = time.time()
    polygon_wkb = row.get("geometry_wkb")
    mask = _extract_mask(pan_22, bbox, expanded, polygon_wkb=polygon_wkb)
    t["mask"] = time.time() - t0

    if mask is None:
        return {"building_id": row["building_id"], "error": "no 2022 footprint",
                "n_scenes_2022": n_22, **t}

    t0 = time.time()
    hrefs_08 = scenes_by_cell.get((cell[0], cell[1], 2008), [])
    pan_08, n_08 = _fetch_pan_median(hrefs_08, expanded)
    t["fetch_2008"] = time.time() - t0

    if pan_08 is None:
        return {"building_id": row["building_id"], "error": "no 2008 data",
                "n_scenes_2008": n_08, **t}

    h = min(pan_08.shape[0], pan_22.shape[0], mask.shape[0])
    w = min(pan_08.shape[1], pan_22.shape[1], mask.shape[1])
    pan_08, pan_22, mask = pan_08[:h, :w], pan_22[:h, :w], mask[:h, :w]

    r_08 = _ratio(pan_08, mask)
    r_22 = _ratio(pan_22, mask)
    change = abs(r_22 - r_08) if (r_08 is not None and r_22 is not None) else None
    fp_px = int(mask.sum())

    return {
        "building_id": row["building_id"],
        "ratio_2008": r_08, "ratio_2022": r_22, "change": change,
        "footprint_pixels": fp_px,
        "n_scenes_2008": n_08, "n_scenes_2022": n_22,
        **t,
    }


# --- Worker entrypoint ----------------------------------------------------- #

_WORKER_SCENES_INDEX = None


def _worker_init(scenes_path):
    """Initializer for each worker process: load lazy imports + scenes index once."""
    global _WORKER_SCENES_INDEX
    _lazy_imports()
    faulthandler.enable()
    sc = pd.read_parquet(scenes_path)
    idx = {}
    for (lat, lon, yr), grp in sc.groupby(["grid_lat", "grid_lon", "year"], sort=False):
        idx[(int(lat), int(lon), int(yr))] = grp["s3_href"].tolist()
    _WORKER_SCENES_INDEX = idx


def _process_chunk(chunk_id, rows_dict, out_path):
    """Process one chunk of candidates serially in this worker. Returns telemetry."""
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
            res = _process_one(r, _WORKER_SCENES_INDEX)
            res["lat"] = r["lat"]; res["lon"] = r["lon"]
            res["p_dino_sat493m"] = r.get("p_dino_sat493m")
            res["approx_area_m2"] = r.get("approx_area_m2")
            res["ovt_class"] = r.get("ovt_class")
            if "error" in res:
                err_n += 1
        except Exception as e:
            res = {"building_id": r["building_id"], "error": f"exc: {type(e).__name__}: {e}"[:200]}
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
        "t_fetch_2022_mean_s": float(np.mean([r.get("fetch_2022", 0) for r in results])),
        "t_fetch_2008_mean_s": float(np.mean([r.get("fetch_2008", 0) for r in results])),
        "t_mask_mean_s": float(np.mean([r.get("mask", 0) for r in results])),
        "rss_mb": rss_mb,
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "out_path": out_path,
    }


# --- Telemetry-side helpers ------------------------------------------------ #

def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _heartbeat(state):
    """Write heartbeat JSON locally + (if configured) push to S3."""
    elapsed = state["elapsed"]
    cands_done = state["cands_done"]
    cands_total = state["cands_total"]
    chunks_done = state["chunks_done"]
    chunks_total = state["chunks_total"]
    rate = cands_done / max(elapsed, 1)
    eta_min = (cands_total - cands_done) / max(rate, 0.01) / 60

    hb = {
        "ts": _now_iso(),
        "instance_id": INSTANCE_ID,
        "elapsed_sec": int(elapsed),
        "chunks_done": chunks_done,
        "chunks_total": chunks_total,
        "cands_done": cands_done,
        "cands_total": cands_total,
        "rate_cand_s": round(rate, 3),
        "eta_min": round(eta_min, 1),
        "err_n": state["err_n"],
        "err_rate": round(state["err_n"] / max(cands_done, 1), 5),
        "mem_rss_gb": round(state.get("rss_gb", 0), 3),
        "mean_per_cand_s": round(state.get("p50_per_cand", 0), 3),
        "mean_fetch_22_s": round(state.get("fetch_22", 0), 3),
        "mean_fetch_08_s": round(state.get("fetch_08", 0), 3),
        "n_workers": NUM_WORKERS,
    }
    with open(HEARTBEAT_PATH, "w") as f:
        json.dump(hb, f, indent=2)
    if S3_BUCKET:
        os.system(
            f"aws s3 cp {HEARTBEAT_PATH} "
            f"s3://{S3_BUCKET}/v3-pt2-artifacts/heartbeat/{INSTANCE_ID}.json "
            f"--only-show-errors 2>/dev/null &"
        )
    return hb


def _dump_stacks(reason):
    """Dump all thread stacks via faulthandler -> file -> S3."""
    path = "/tmp/stage2b_stacks.txt"
    with open(path, "w") as f:
        f.write(f"# stage2b stack dump @ {_now_iso()}  reason={reason}\n")
        faulthandler.dump_traceback(file=f, all_threads=True)
    print(f"[stage2b] stall dump -> {path}", flush=True)
    if S3_BUCKET:
        os.system(
            f"aws s3 cp {path} s3://{S3_BUCKET}/v3-pt2-artifacts/stacks/{INSTANCE_ID}.txt "
            f"--only-show-errors 2>/dev/null"
        )


# --- Main ------------------------------------------------------------------ #

def main():
    print(f"[stage2b] start @ {_now_iso()}  host={socket.gethostname()} workers={NUM_WORKERS}")
    print(f"[stage2b] chunk_size={CHUNK_SIZE}  s3_bucket={S3_BUCKET or '(none)'}")

    if not os.path.exists(CANDS):
        raise SystemExit(f"missing {CANDS} — run build_candidates.py first")
    if not os.path.exists(SCENES):
        raise SystemExit(f"missing {SCENES} — run build_scenes_index.py first")

    df = pd.read_parquet(CANDS)
    sc = pd.read_parquet(SCENES)
    print(f"[stage2b] candidates raw: {len(df):,}  scenes: {len(sc):,}")
    if "p_dino_sat493m" in df.columns:
        before = len(df)
        df = df[(df["p_dino_sat493m"] >= MIN_PROB) & (df["p_dino_sat493m"] < MAX_PROB)].reset_index(drop=True)
        print(f"[stage2b] p_dino in [{MIN_PROB}, {MAX_PROB}): {before:,} -> {len(df):,}  run_tag={RUN_TAG or '(none)'}")

    # Polygon join (optional — falls back to bbox-mask if missing).
    poly_path = os.path.join(ROOT, "..", "data_us/phase2/v3/stage2_candidate_polygons.parquet")
    if os.path.exists(poly_path):
        polys = pd.read_parquet(poly_path)
        df = df.merge(polys[["ovt_id", "geometry_wkb"]], on="ovt_id", how="left")
        n_poly = df["geometry_wkb"].notna().sum()
        print(f"[stage2b] polygons attached: {n_poly:,}/{len(df):,} ({100*n_poly/len(df):.1f}%)")
    else:
        df["geometry_wkb"] = None
        print(f"[stage2b] WARN: no polygons at {poly_path} — falling back to bbox-only masks")

    # Sort candidates by grid cell -> VSI cache warm across nearby buildings.
    df["_lat_idx"] = np.floor(df["lat"].values / GRID_DEG).astype(int)
    df["_lon_idx"] = np.floor(df["lon"].values / GRID_DEG).astype(int)
    df = df.sort_values(["_lat_idx", "_lon_idx"]).reset_index(drop=True)

    # Chunk plan + resume.
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
    print(f"[stage2b] chunks: total={n_chunks}  done={len(existing)}  pending={len(pending)}")
    if not pending:
        print("[stage2b] all chunks already done")
        return

    t_start = time.time()
    cands_done_resumed = len(existing) * CHUNK_SIZE  # approximate; tolerate last-chunk shortfall
    cands_done = 0
    err_n = 0
    chunks_done = 0
    last_chunk_t = time.time()
    last_hb = 0.0
    rolling = {"fetch_22": [], "fetch_08": [], "mask": [], "p50_per_cand": [], "rss_gb": 0.0}

    # Submit pending chunks.
    futs = {}
    with ProcessPoolExecutor(max_workers=NUM_WORKERS,
                              initializer=_worker_init, initargs=(SCENES,)) as pool:
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
                print(f"[stage2b] chunk {chunk_id} CRASH: {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
                continue

            chunks_done += 1
            cands_done += tel["n"]
            err_n += tel["n_err"]
            last_chunk_t = time.time()
            rolling["fetch_22"].append(tel["t_fetch_2022_mean_s"])
            rolling["fetch_08"].append(tel["t_fetch_2008_mean_s"])
            rolling["mask"].append(tel["t_mask_mean_s"])
            rolling["p50_per_cand"].append(tel["t_p50_per_cand_s"])
            if tel.get("rss_mb"):
                rolling["rss_gb"] = max(rolling["rss_gb"], tel["rss_mb"] / 1024)

            # Stats line per chunk.
            with open(STATS_LOG, "a") as sf:
                sf.write(json.dumps({"ts": _now_iso(), **tel}) + "\n")

            elapsed = time.time() - t_start
            rate = cands_done / max(elapsed, 1)
            remaining_cands = (len(pending) - chunks_done) * CHUNK_SIZE
            eta_min = remaining_cands / max(rate, 0.01) / 60
            print(
                f"[stage2b] chunk {chunk_id:>5}  done={chunks_done}/{len(pending)}  "
                f"cand={cands_done:,}/{len(pending) * CHUNK_SIZE:,}(this-run)  "
                f"err={err_n}  rate={rate:.1f}/s  eta={eta_min:.0f}m  "
                f"p50={tel['t_p50_per_cand_s']:.2f}s  rss={rolling['rss_gb']:.1f}G",
                flush=True,
            )

            # Sync chunk to S3 immediately.
            if S3_BUCKET:
                os.system(
                    f"aws s3 cp {tel['out_path']} "
                    f"s3://{S3_BUCKET}/v3-pt2-artifacts/change_chunks_v3{RUN_TAG}/ "
                    f"--only-show-errors 2>/dev/null &"
                )

            # Heartbeat + stall check.
            now = time.time()
            if now - last_hb >= HEARTBEAT_SEC:
                _heartbeat({
                    "elapsed": elapsed, "cands_done": cands_done,
                    "cands_total": len(pending) * CHUNK_SIZE,
                    "chunks_done": chunks_done, "chunks_total": len(pending),
                    "err_n": err_n,
                    "rss_gb": rolling["rss_gb"],
                    "p50_per_cand": float(np.mean(rolling["p50_per_cand"][-32:])) if rolling["p50_per_cand"] else 0,
                    "fetch_22": float(np.mean(rolling["fetch_22"][-32:])) if rolling["fetch_22"] else 0,
                    "fetch_08": float(np.mean(rolling["fetch_08"][-32:])) if rolling["fetch_08"] else 0,
                })
                last_hb = now

            if now - last_chunk_t > STALL_SEC:
                _dump_stacks(f"no chunk completed in {STALL_SEC}s")
                last_chunk_t = now   # don't dump again immediately

    print(f"[stage2b] DONE  chunks_done={chunks_done}  cand_done={cands_done:,}  err={err_n}  "
          f"elapsed={(time.time() - t_start)/60:.1f}m", flush=True)

    # Final heartbeat.
    _heartbeat({
        "elapsed": time.time() - t_start, "cands_done": cands_done,
        "cands_total": len(pending) * CHUNK_SIZE,
        "chunks_done": chunks_done, "chunks_total": len(pending),
        "err_n": err_n, "rss_gb": rolling["rss_gb"],
        "p50_per_cand": float(np.mean(rolling["p50_per_cand"])) if rolling["p50_per_cand"] else 0,
        "fetch_22": float(np.mean(rolling["fetch_22"])) if rolling["fetch_22"] else 0,
        "fetch_08": float(np.mean(rolling["fetch_08"])) if rolling["fetch_08"] else 0,
    })


if __name__ == "__main__":
    main()
