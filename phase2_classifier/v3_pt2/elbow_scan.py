"""Elbow scan — L7 2008 surface-reflectance 6-band signature per polygon.

Mirrors change_scan.py infrastructure (resumable chunks, ProcessPool, S3 sync,
heartbeat, stall watchdog) but the per-candidate work is:

  1. Lookup L7 2008 scenes by 1-deg grid cell.
  2. Derive L2-SR hrefs from each scene's L1 PAN href:
       /level-1/ -> /level-2/
       _L1TP_    -> _L2SP_           (also _L1GT_ -> _L2SP_ if present)
       _B8.TIF   -> _SR_B{1..7}.TIF
  3. Fetch each band's median composite over (polygon bbox + MARGIN_M).
  4. Mask = rasterize polygon at SR resolution (all_touched=True, NO dilation —
     matches the calibration scripts, not change_scan's 1px buffer).
  5. Per-band mean over mask -> 6 floats per polygon.

Output per chunk: building_id, lat, lon, ovt_id, p_dino_sat493m, ovt_class,
                  approx_area_m2, blue, green, red, nir08, swir16, swir22,
                  footprint_pixels, n_scenes, error

Env vars (same envelope as change_scan.py):
  STAGE2B_NUM_WORKERS, STAGE2B_CHUNK_SIZE, STAGE2B_S3_BUCKET, STAGE2B_INSTANCE_ID,
  STAGE2B_HEARTBEAT_SEC, STAGE2B_STALL_SEC, STAGE2B_MIN_PROB, STAGE2B_MAX_PROB,
  STAGE2B_RUN_TAG
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

NUM_WORKERS = int(os.environ.get("STAGE2B_NUM_WORKERS", "32"))
CHUNK_SIZE = int(os.environ.get("STAGE2B_CHUNK_SIZE", "200"))
S3_BUCKET = os.environ.get("STAGE2B_S3_BUCKET", "").strip()
INSTANCE_ID = os.environ.get("STAGE2B_INSTANCE_ID", socket.gethostname())
HEARTBEAT_SEC = int(os.environ.get("STAGE2B_HEARTBEAT_SEC", "30"))
STALL_SEC = int(os.environ.get("STAGE2B_STALL_SEC", "300"))
CAND_TIMEOUT_SEC = int(os.environ.get("STAGE2B_CAND_TIMEOUT_SEC", "60"))
MIN_PROB = float(os.environ.get("STAGE2B_MIN_PROB", "0.30"))
MAX_PROB = float(os.environ.get("STAGE2B_MAX_PROB", "1.01"))
RUN_TAG = os.environ.get("STAGE2B_RUN_TAG", "")

os.environ.setdefault("AWS_REQUEST_PAYER", "requester")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("GDAL_HTTP_MULTIPLEX", "YES")
os.environ.setdefault("GDAL_HTTP_VERSION", "2")
os.environ.setdefault("GDAL_HTTP_TIMEOUT", "30")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "5")
os.environ.setdefault("CPL_VSIL_CURL_USE_HEAD", "NO")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".TIF,.tif")
os.environ.setdefault("VSI_CACHE", "TRUE")
os.environ.setdefault("VSI_CACHE_SIZE", "134217728")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CANDS = os.path.join(ROOT, "..", "data_us/phase2/v3/stage2_candidates.parquet")
SCENES = os.path.join(ROOT, "..", "data_us/phase2/v3/landsat_scenes_index.parquet")
POLYS = os.path.join(ROOT, "..", "data_us/phase2/v3/stage2_candidate_polygons.parquet")
CHUNK_DIR = os.path.join(ROOT, "..", f"data_us/phase2/v3/elbow_chunks{RUN_TAG}")
STATS_LOG = os.path.join(CHUNK_DIR, "_stats.jsonl")
HEARTBEAT_PATH = os.path.join(CHUNK_DIR, "_heartbeat.json")

GRID_DEG = 1.0
MARGIN_M = 30
BANDS = ("blue", "green", "red", "nir08", "swir16", "swir22")
BAND_SUFFIX = {
    "blue": "_SR_B1.TIF",
    "green": "_SR_B2.TIF",
    "red": "_SR_B3.TIF",
    "nir08": "_SR_B4.TIF",
    "swir16": "_SR_B5.TIF",
    "swir22": "_SR_B7.TIF",
}


def _lazy_imports():
    global rasterio, from_bounds, transform_bounds, transform_from_bounds, \
           rasterize, wkb_loads
    import rasterio
    from rasterio.windows import from_bounds
    from rasterio.warp import transform_bounds
    from rasterio.transform import from_bounds as transform_from_bounds
    from rasterio.features import rasterize
    from shapely.wkb import loads as wkb_loads


def _l1_pan_to_l2_sr(href_l1_pan: str, band: str) -> str:
    """L7 only. Replaces L1 PAN path with L2 SR band path."""
    suffix = BAND_SUFFIX[band]
    return (href_l1_pan
            .replace("/level-1/", "/level-2/")
            .replace("_L1TP_", "_L2SP_")
            .replace("_L1GT_", "_L2SP_")
            .replace("_B8.TIF", suffix))


def _expand_bbox_ll(bbox_ll, margin_m=MARGIN_M):
    cy = (bbox_ll[1] + bbox_ll[3]) / 2
    dlat = margin_m / 111_000
    dlon = margin_m / (111_000 * np.cos(np.radians(cy)))
    return (bbox_ll[0] - dlon, bbox_ll[1] - dlat, bbox_ll[2] + dlon, bbox_ll[3] + dlat)


def _grid_cell(lat, lon, deg=GRID_DEG):
    return int(np.floor(lat / deg)), int(np.floor(lon / deg))


def _fetch_band_median(hrefs, bbox_ll):
    """Fetch one L7 SR band over bbox across multiple scenes, return median (refl)."""
    stack = []
    with rasterio.Env():
        for href in hrefs:
            try:
                with rasterio.open(href) as src:
                    bbox_utm = transform_bounds("EPSG:4326", src.crs, *bbox_ll)
                    win = from_bounds(*bbox_utm, transform=src.transform).round_offsets().round_lengths()
                    if win.width < 2 or win.height < 2:
                        continue
                    arr = src.read(1, window=win, boundless=True, fill_value=0).astype(np.float32)
                refl = arr * 2.75e-5 - 0.2  # L2-SR scaling
                refl[arr == 0] = np.nan
                refl[(refl < 0) | (refl > 1)] = np.nan
                if np.isnan(refl).all():
                    continue
                stack.append(refl)
            except Exception:
                continue
    if not stack:
        return None, 0
    h = min(a.shape[0] for a in stack)
    w = min(a.shape[1] for a in stack)
    return np.nanmedian(np.stack([a[:h, :w] for a in stack], axis=0), axis=0), len(stack)


def _poly_mask(shape, bbox_ll, polygon_wkb):
    h, w = shape
    ex0, ey0, ex1, ey1 = bbox_ll
    try:
        poly = wkb_loads(polygon_wkb)
        transform = transform_from_bounds(ex0, ey0, ex1, ey1, w, h)
        mask = rasterize([(poly, 1)], out_shape=(h, w), transform=transform,
                         fill=0, dtype=np.uint8, all_touched=True).astype(bool)
        return mask if mask.any() else None
    except Exception:
        return None


def _process_one(row, scenes_by_cell):
    bbox = (row["xmin"], row["ymin"], row["xmax"], row["ymax"])
    expanded = _expand_bbox_ll(bbox, MARGIN_M)
    cell = _grid_cell(row["lat"], row["lon"])

    l1_hrefs = scenes_by_cell.get((cell[0], cell[1], 2008), [])
    if not l1_hrefs:
        return {"building_id": row["building_id"], "error": "no 2008 scenes"}

    polygon_wkb = row.get("geometry_wkb")
    if polygon_wkb is None:
        return {"building_id": row["building_id"], "error": "no polygon"}

    t = {}
    medians = {}
    n_scenes = 0
    for b in BANDS:
        t0 = time.time()
        hrefs_b = [_l1_pan_to_l2_sr(h, b) for h in l1_hrefs]
        med, n = _fetch_band_median(hrefs_b, expanded)
        t[f"t_{b}"] = time.time() - t0
        if med is None:
            return {"building_id": row["building_id"], "error": f"no {b} data",
                    "n_scenes": n, **t}
        medians[b] = med
        n_scenes = max(n_scenes, n)

    hs = min(m.shape[0] for m in medians.values())
    ws = min(m.shape[1] for m in medians.values())
    medians = {k: v[:hs, :ws] for k, v in medians.items()}

    mask = _poly_mask((hs, ws), expanded, polygon_wkb)
    if mask is None:
        return {"building_id": row["building_id"], "error": "no mask",
                "n_scenes": n_scenes, **t}

    valid = mask.copy()
    for v in medians.values():
        valid &= ~np.isnan(v)
    if not valid.any():
        return {"building_id": row["building_id"], "error": "no valid pixels",
                "n_scenes": n_scenes, **t}

    out = {"building_id": row["building_id"], "n_scenes": n_scenes,
           "footprint_pixels": int(valid.sum()), **t}
    for b in BANDS:
        out[b] = float(np.nanmean(medians[b][valid]))
    return out


_WORKER_SCENES_INDEX = None


def _worker_init(scenes_path):
    global _WORKER_SCENES_INDEX
    _lazy_imports()
    faulthandler.enable()
    sc = pd.read_parquet(scenes_path)
    sc = sc[(sc["year"] == 2008) & (sc["platform"] == "LANDSAT_7")]
    idx = {}
    for (lat, lon, yr), grp in sc.groupby(["grid_lat", "grid_lon", "year"], sort=False):
        idx[(int(lat), int(lon), int(yr))] = grp["s3_href"].tolist()
    _WORKER_SCENES_INDEX = idx


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
            res = _process_one(r, _WORKER_SCENES_INDEX)
            res["lat"] = r["lat"]; res["lon"] = r["lon"]
            res["ovt_id"] = r.get("ovt_id")
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
        "rss_mb": rss_mb,
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "out_path": out_path,
    }


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _heartbeat(state):
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
    path = "/tmp/elbow_stacks.txt"
    with open(path, "w") as f:
        f.write(f"# elbow stack dump @ {_now_iso()}  reason={reason}\n")
        faulthandler.dump_traceback(file=f, all_threads=True)
    print(f"[elbow] stall dump -> {path}", flush=True)
    if S3_BUCKET:
        os.system(
            f"aws s3 cp {path} s3://{S3_BUCKET}/v3-pt2-artifacts/stacks/{INSTANCE_ID}.txt "
            f"--only-show-errors 2>/dev/null"
        )


def main():
    print(f"[elbow] start @ {_now_iso()}  host={socket.gethostname()} workers={NUM_WORKERS}")
    print(f"[elbow] chunk_size={CHUNK_SIZE}  s3_bucket={S3_BUCKET or '(none)'}  run_tag={RUN_TAG or '(none)'}")

    if not os.path.exists(CANDS):
        raise SystemExit(f"missing {CANDS}")
    if not os.path.exists(SCENES):
        raise SystemExit(f"missing {SCENES}")
    if not os.path.exists(POLYS):
        raise SystemExit(f"missing {POLYS}")

    df = pd.read_parquet(CANDS)
    print(f"[elbow] candidates raw: {len(df):,}")
    if "p_dino_sat493m" in df.columns:
        before = len(df)
        df = df[(df["p_dino_sat493m"] >= MIN_PROB) & (df["p_dino_sat493m"] < MAX_PROB)].reset_index(drop=True)
        print(f"[elbow] p_dino in [{MIN_PROB}, {MAX_PROB}): {before:,} -> {len(df):,}")

    polys = pd.read_parquet(POLYS)
    df = df.merge(polys[["ovt_id", "geometry_wkb"]], on="ovt_id", how="left")
    n_poly = df["geometry_wkb"].notna().sum()
    print(f"[elbow] polygons attached: {n_poly:,}/{len(df):,} ({100*n_poly/len(df):.1f}%)")

    df["_lat_idx"] = np.floor(df["lat"].values / GRID_DEG).astype(int)
    df["_lon_idx"] = np.floor(df["lon"].values / GRID_DEG).astype(int)
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
    print(f"[elbow] chunks: total={n_chunks}  done={len(existing)}  pending={len(pending)}")
    if not pending:
        print("[elbow] all chunks already done")
        return

    t_start = time.time()
    cands_done = 0
    err_n = 0
    chunks_done = 0
    last_chunk_t = time.time()
    last_hb = 0.0
    rolling = {"p50_per_cand": [], "rss_gb": 0.0}

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
                print(f"[elbow] chunk {chunk_id} CRASH: {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
                continue

            chunks_done += 1
            cands_done += tel["n"]
            err_n += tel["n_err"]
            last_chunk_t = time.time()
            rolling["p50_per_cand"].append(tel["t_p50_per_cand_s"])
            if tel.get("rss_mb"):
                rolling["rss_gb"] = max(rolling["rss_gb"], tel["rss_mb"] / 1024)

            with open(STATS_LOG, "a") as sf:
                sf.write(json.dumps({"ts": _now_iso(), **tel}) + "\n")

            elapsed = time.time() - t_start
            rate = cands_done / max(elapsed, 1)
            remaining_cands = (len(pending) - chunks_done) * CHUNK_SIZE
            eta_min = remaining_cands / max(rate, 0.01) / 60
            print(
                f"[elbow] chunk {chunk_id:>5}  done={chunks_done}/{len(pending)}  "
                f"cand={cands_done:,}/{len(pending) * CHUNK_SIZE:,}  "
                f"err={err_n}  rate={rate:.1f}/s  eta={eta_min:.0f}m  "
                f"p50={tel['t_p50_per_cand_s']:.2f}s  rss={rolling['rss_gb']:.1f}G",
                flush=True,
            )

            if S3_BUCKET:
                os.system(
                    f"aws s3 cp {tel['out_path']} "
                    f"s3://{S3_BUCKET}/v3-pt2-artifacts/elbow_chunks{RUN_TAG}/ "
                    f"--only-show-errors 2>/dev/null &"
                )

            now = time.time()
            if now - last_hb >= HEARTBEAT_SEC:
                _heartbeat({
                    "elapsed": elapsed, "cands_done": cands_done,
                    "cands_total": len(pending) * CHUNK_SIZE,
                    "chunks_done": chunks_done, "chunks_total": len(pending),
                    "err_n": err_n,
                    "rss_gb": rolling["rss_gb"],
                    "p50_per_cand": float(np.mean(rolling["p50_per_cand"][-32:])) if rolling["p50_per_cand"] else 0,
                })
                last_hb = now

            if now - last_chunk_t > STALL_SEC:
                _dump_stacks(f"no chunk completed in {STALL_SEC}s")
                last_chunk_t = now

    print(f"[elbow] DONE  chunks_done={chunks_done}  cand_done={cands_done:,}  err={err_n}  "
          f"elapsed={(time.time() - t_start)/60:.1f}m", flush=True)

    _heartbeat({
        "elapsed": time.time() - t_start, "cands_done": cands_done,
        "cands_total": len(pending) * CHUNK_SIZE,
        "chunks_done": chunks_done, "chunks_total": len(pending),
        "err_n": err_n, "rss_gb": rolling["rss_gb"],
        "p50_per_cand": float(np.mean(rolling["p50_per_cand"])) if rolling["p50_per_cand"] else 0,
    })


if __name__ == "__main__":
    main()
