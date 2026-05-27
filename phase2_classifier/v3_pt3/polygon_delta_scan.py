"""Stage 2b v2 — NAIP polygon-delta change scoring.

Per candidate, fetch baseline (~2011-2013) and recent (~2021-2023) NAIP windows
around the Overture polygon bbox. Compute inside-polygon stats per year:
  - mean per band (R, G, B, NIR)
  - std per band
  - NDVI = (NIR-R)/(NIR+R)
  - Sobel-magnitude mean per band (edge density)

Both years are read at the same target ground-resolution (TARGET_GSD_M, default
1 m) so std/Sobel deltas aren't biased by recent NAIP being 30-60 cm vs baseline
NAIP being 1 m. Polygon is reprojected to each COG's CRS for exact mask.

Reads:
  data_us/phase2/v3/stage3_candidates_v3.parquet            (344K rows)
  data_us/phase2/v3/stage2_candidate_polygons.parquet       (ovt_id, geometry_wkb)
  data_us/phase3_naip/naip_tile_index_baseline.parquet
  data_us/phase3_naip/naip_tile_index_recent.parquet
Writes:
  data_us/phase2/v3/stage2b_naip_chunks{RUN_TAG}/chunk_XXXXX.parquet

Env knobs (mirror v3_pt2/change_scan.py):
  STAGE2B_NUM_WORKERS, STAGE2B_CHUNK_SIZE, STAGE2B_S3_BUCKET,
  STAGE2B_INSTANCE_ID, STAGE2B_HEARTBEAT_SEC, STAGE2B_STALL_SEC,
  STAGE2B_CAND_TIMEOUT_SEC, STAGE2B_MIN_PROB, STAGE2B_MAX_PROB,
  STAGE2B_RUN_TAG.
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
CHUNK_SIZE = int(os.environ.get("STAGE2B_CHUNK_SIZE", "200"))
S3_BUCKET = os.environ.get("STAGE2B_S3_BUCKET", "").strip()
INSTANCE_ID = os.environ.get("STAGE2B_INSTANCE_ID", socket.gethostname())
HEARTBEAT_SEC = int(os.environ.get("STAGE2B_HEARTBEAT_SEC", "30"))
STALL_SEC = int(os.environ.get("STAGE2B_STALL_SEC", "300"))
CAND_TIMEOUT_SEC = int(os.environ.get("STAGE2B_CAND_TIMEOUT_SEC", "60"))
MIN_PROB = float(os.environ.get("STAGE2B_MIN_PROB", "0.30"))
MAX_PROB = float(os.environ.get("STAGE2B_MAX_PROB", "1.01"))
RUN_TAG = os.environ.get("STAGE2B_RUN_TAG", "")
LIMIT = int(os.environ.get("STAGE2B_LIMIT", "0"))  # 0 = no limit (full cohort)

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
os.environ.setdefault("VSI_CACHE_SIZE", "134217728")

# --- Paths ------------------------------------------------------------------ #

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CANDS = os.path.join(ROOT, "..", "data_us/phase2/v3/stage3_candidates_v3.parquet")
POLYS = os.path.join(ROOT, "..", "data_us/phase2/v3/stage2_candidate_polygons.parquet")
IDX_BASELINE = os.path.join(ROOT, "..", "data_us/phase3_naip/naip_tile_index_baseline.parquet")
IDX_RECENT = os.path.join(ROOT, "..", "data_us/phase3_naip/naip_tile_index_recent.parquet")
CHUNK_DIR = os.path.join(ROOT, "..", f"data_us/phase2/v3/stage2b_naip_chunks{RUN_TAG}")
STATS_LOG = os.path.join(CHUNK_DIR, "_stats.jsonl")
HEARTBEAT_PATH = os.path.join(CHUNK_DIR, "_heartbeat.json")

# --- Algorithm constants ---------------------------------------------------- #

MARGIN_M = 10.0          # ~10 NAIP pixels at 1 m around polygon bbox
TARGET_GSD_M = 1.0       # read both years at this ground-sample-distance
MAX_BBOX_SIDE_M = 2500   # skip super-huge polygons (>2.5 km on a side)
BANDS = ("R", "G", "B", "NIR")


# --- Worker globals (populated in _worker_init) ----------------------------- #

_TREE_BASELINE = None
_TREE_RECENT = None
_IDX_BASELINE = None
_IDX_RECENT = None
_TRANSFORMERS = None  # cache (src_crs_wkt) -> Transformer


def _lazy_imports():
    global rasterio, from_bounds, transform_bounds, transform_from_bounds, \
           rasterize, wkb_loads, shapely_transform, Transformer, sobel, \
           psutil, shapely_box, STRtree, Resampling
    import rasterio
    from rasterio.windows import from_bounds
    from rasterio.warp import transform_bounds
    from rasterio.transform import from_bounds as transform_from_bounds
    from rasterio.features import rasterize
    from rasterio.enums import Resampling
    from shapely.wkb import loads as wkb_loads
    from shapely.ops import transform as shapely_transform
    from shapely import STRtree, box as shapely_box
    from pyproj import Transformer
    from scipy.ndimage import sobel
    try:
        import psutil
    except ImportError:
        psutil = None


# --- Per-candidate logic --------------------------------------------------- #

def _bbox_around(lon, lat, half_m):
    dlat = half_m / 111_000.0
    dlon = half_m / (111_000.0 * np.cos(np.radians(lat)))
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)


def _expand_bbox_ll(bbox_ll, margin_m):
    cy = (bbox_ll[1] + bbox_ll[3]) / 2
    dlat = margin_m / 111_000.0
    dlon = margin_m / (111_000.0 * np.cos(np.radians(cy)))
    return (bbox_ll[0] - dlon, bbox_ll[1] - dlat, bbox_ll[2] + dlon, bbox_ll[3] + dlat)


def _pick_tile(tree, tile_idx, cand_box, cand_cx, cand_cy):
    """Tree.query(cand_box, intersects). Pick the hit whose centroid is closest to candidate centroid."""
    hits = tree.query(cand_box, predicate="intersects")
    if len(hits) == 0:
        return None
    best = None
    best_d2 = float("inf")
    lon_min = tile_idx["lon_min"].values
    lon_max = tile_idx["lon_max"].values
    lat_min = tile_idx["lat_min"].values
    lat_max = tile_idx["lat_max"].values
    for h in hits.tolist():
        tx = (lon_min[h] + lon_max[h]) / 2
        ty = (lat_min[h] + lat_max[h]) / 2
        d2 = (tx - cand_cx) ** 2 + (ty - cand_cy) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best = h
    return best


def _get_transformer(src_crs_wkt):
    """Cache pyproj Transformer 4326 -> src_crs per worker."""
    if src_crs_wkt in _TRANSFORMERS:
        return _TRANSFORMERS[src_crs_wkt]
    t = Transformer.from_crs("EPSG:4326", src_crs_wkt, always_xy=True)
    _TRANSFORMERS[src_crs_wkt] = t
    return t


def _fetch_and_stats(tile_uri, polygon, expanded_ll):
    """Open the NAIP COG, read a window around expanded_ll at TARGET_GSD_M, rasterize
    the polygon in COG CRS, compute inside-mask stats. Returns dict or {'error': ...}.

    Stats: mean/std per band (R,G,B,NIR), NDVI scalar, Sobel-magnitude mean per band.
    Plus footprint_pixels.
    """
    try:
        with rasterio.open(tile_uri) as src:
            src_crs_wkt = src.crs.to_wkt()
            bbox_utm = transform_bounds("EPSG:4326", src.crs, *expanded_ll, densify_pts=21)
            win = from_bounds(*bbox_utm, transform=src.transform).round_offsets().round_lengths()
            if win.width <= 0 or win.height <= 0:
                return {"error": "window outside tile"}

            # Actual CRS bounds of the rounded window (may differ slightly from bbox_utm)
            win_bounds = rasterio.windows.bounds(win, src.transform)
            extent_x = win_bounds[2] - win_bounds[0]
            extent_y = win_bounds[3] - win_bounds[1]
            target_w = max(8, int(round(extent_x / TARGET_GSD_M)))
            target_h = max(8, int(round(extent_y / TARGET_GSD_M)))

            arr = src.read(
                [1, 2, 3, 4],
                window=win,
                out_shape=(4, target_h, target_w),
                boundless=True,
                fill_value=0,
                resampling=Resampling.average,
            )  # (4, h, w) uint8 — average matters when recent NAIP (30-60cm) is downsampled to 1m

            sx = extent_x / target_w
            sy = extent_y / target_h
            target_transform = rasterio.transform.from_origin(win_bounds[0], win_bounds[3], sx, sy)
    except Exception as e:
        return {"error": f"open/read: {type(e).__name__}: {str(e)[:120]}"}

    if arr.max() == 0:
        return {"error": "no pixel data"}

    arr_f = arr.astype(np.float32) / 255.0

    try:
        transformer = _get_transformer(src_crs_wkt)
        poly_utm = shapely_transform(transformer.transform, polygon)
        mask = rasterize(
            [(poly_utm, 1)],
            out_shape=(target_h, target_w),
            transform=target_transform,
            fill=0, dtype=np.uint8, all_touched=True,
        ).astype(bool)
    except Exception as e:
        return {"error": f"rasterize: {type(e).__name__}: {str(e)[:120]}"}

    if not mask.any():
        return {"error": "empty mask"}

    inside_n = int(mask.sum())
    out = {"footprint_pixels": inside_n}

    # Per-band stats inside polygon
    for i, name in enumerate(BANDS):
        band = arr_f[i]
        vals = band[mask]
        out[f"mean_{name}"] = float(vals.mean())
        out[f"std_{name}"] = float(vals.std())
        # Sobel magnitude inside polygon
        sx = sobel(band, axis=1, mode="nearest")
        sy = sobel(band, axis=0, mode="nearest")
        mag = np.sqrt(sx * sx + sy * sy)
        out[f"edge_{name}"] = float(mag[mask].mean())

    # NDVI
    r = arr_f[0][mask]
    nir = arr_f[3][mask]
    denom = nir + r
    valid = denom > 1e-6
    if valid.any():
        out["ndvi"] = float(((nir - r)[valid] / denom[valid]).mean())
    else:
        out["ndvi"] = float("nan")

    return out


def _process_one(row):
    """Single candidate: fetch baseline + recent, compute stats both years."""
    polygon_wkb = row.get("geometry_wkb")
    if polygon_wkb is None:
        return {"building_id": row["building_id"], "error": "no polygon"}

    try:
        poly = wkb_loads(polygon_wkb)
    except Exception as e:
        return {"building_id": row["building_id"], "error": f"poly: {type(e).__name__}"}

    # Polygon bbox (lon/lat)
    minx, miny, maxx, maxy = poly.bounds
    side_lat_m = (maxy - miny) * 111_000.0
    side_lon_m = (maxx - minx) * 111_000.0 * np.cos(np.radians((miny + maxy) / 2))
    if max(side_lat_m, side_lon_m) > MAX_BBOX_SIDE_M:
        return {"building_id": row["building_id"], "error": "polygon too large"}

    bbox_ll = (minx, miny, maxx, maxy)
    expanded_ll = _expand_bbox_ll(bbox_ll, MARGIN_M)
    cand_box = shapely_box(*expanded_ll)
    cand_cx = (minx + maxx) / 2
    cand_cy = (miny + maxy) / 2

    out = {"building_id": row["building_id"]}
    t = {}

    # Baseline
    t0 = time.time()
    ti = _pick_tile(_TREE_BASELINE, _IDX_BASELINE, cand_box, cand_cx, cand_cy)
    if ti is None:
        return {**out, "error": "no baseline tile"}
    out["baseline_year"] = int(_IDX_BASELINE["naip_year"].iloc[ti])
    out["baseline_state"] = str(_IDX_BASELINE["state"].iloc[ti])
    stats_b = _fetch_and_stats(_IDX_BASELINE["tile_uri"].iloc[ti], poly, expanded_ll)
    t["fetch_baseline"] = time.time() - t0
    if "error" in stats_b:
        return {**out, "error": f"baseline: {stats_b['error']}", **t}
    for k, v in stats_b.items():
        out[f"{k}_baseline"] = v

    # Recent
    t0 = time.time()
    ti = _pick_tile(_TREE_RECENT, _IDX_RECENT, cand_box, cand_cx, cand_cy)
    if ti is None:
        return {**out, "error": "no recent tile", **t}
    out["recent_year"] = int(_IDX_RECENT["naip_year"].iloc[ti])
    out["recent_state"] = str(_IDX_RECENT["state"].iloc[ti])
    stats_r = _fetch_and_stats(_IDX_RECENT["tile_uri"].iloc[ti], poly, expanded_ll)
    t["fetch_recent"] = time.time() - t0
    if "error" in stats_r:
        return {**out, "error": f"recent: {stats_r['error']}", **t}
    for k, v in stats_r.items():
        out[f"{k}_recent"] = v

    out.update(t)
    return out


# --- Worker entrypoint ----------------------------------------------------- #

def _worker_init(idx_baseline_path, idx_recent_path):
    global _TREE_BASELINE, _TREE_RECENT, _IDX_BASELINE, _IDX_RECENT, _TRANSFORMERS
    _lazy_imports()
    faulthandler.enable()
    _IDX_BASELINE = pd.read_parquet(idx_baseline_path).reset_index(drop=True)
    _IDX_RECENT = pd.read_parquet(idx_recent_path).reset_index(drop=True)
    _TREE_BASELINE = STRtree(shapely_box(
        _IDX_BASELINE["lon_min"].to_numpy(),
        _IDX_BASELINE["lat_min"].to_numpy(),
        _IDX_BASELINE["lon_max"].to_numpy(),
        _IDX_BASELINE["lat_max"].to_numpy(),
    ))
    _TREE_RECENT = STRtree(shapely_box(
        _IDX_RECENT["lon_min"].to_numpy(),
        _IDX_RECENT["lat_min"].to_numpy(),
        _IDX_RECENT["lon_max"].to_numpy(),
        _IDX_RECENT["lat_max"].to_numpy(),
    ))
    _TRANSFORMERS = {}


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
            f"s3://{S3_BUCKET}/v3-pt3-artifacts/heartbeat/{INSTANCE_ID}.json "
            f"--only-show-errors 2>/dev/null &"
        )
    return hb


def _dump_stacks(reason):
    path = "/tmp/stage2b_naip_stacks.txt"
    with open(path, "w") as f:
        f.write(f"# stage2b-naip stack dump @ {_now_iso()}  reason={reason}\n")
        faulthandler.dump_traceback(file=f, all_threads=True)
    print(f"[stage2b-naip] stall dump -> {path}", flush=True)
    if S3_BUCKET:
        os.system(
            f"aws s3 cp {path} s3://{S3_BUCKET}/v3-pt3-artifacts/stacks/{INSTANCE_ID}.txt "
            f"--only-show-errors 2>/dev/null"
        )


# --- Main ------------------------------------------------------------------ #

def main():
    print(f"[stage2b-naip] start @ {_now_iso()}  host={socket.gethostname()} workers={NUM_WORKERS}")
    print(f"[stage2b-naip] chunk_size={CHUNK_SIZE}  s3_bucket={S3_BUCKET or '(none)'}  run_tag={RUN_TAG or '(none)'}")

    for p in (CANDS, POLYS, IDX_BASELINE, IDX_RECENT):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}")

    df = pd.read_parquet(CANDS)
    print(f"[stage2b-naip] candidates raw: {len(df):,}")
    if "p_dino_sat493m" in df.columns:
        before = len(df)
        df = df[(df["p_dino_sat493m"] >= MIN_PROB) & (df["p_dino_sat493m"] < MAX_PROB)].reset_index(drop=True)
        print(f"[stage2b-naip] p_dino in [{MIN_PROB}, {MAX_PROB}): {before:,} -> {len(df):,}")

    polys = pd.read_parquet(POLYS)
    df = df.merge(polys[["ovt_id", "geometry_wkb"]], on="ovt_id", how="left")
    n_poly = df["geometry_wkb"].notna().sum()
    print(f"[stage2b-naip] polygons attached: {n_poly:,}/{len(df):,} ({100*n_poly/len(df):.1f}%)")

    if LIMIT > 0:
        df = df.sample(n=min(LIMIT, len(df)), random_state=7).reset_index(drop=True)
        print(f"[stage2b-naip] LIMIT={LIMIT} -> sampled {len(df)} candidates")

    # Sort by 1-deg cell -> VSI cache warm across nearby buildings
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
    print(f"[stage2b-naip] chunks: total={n_chunks}  done={len(existing)}  pending={len(pending)}")
    if not pending:
        print("[stage2b-naip] all chunks already done")
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
                              initializer=_worker_init,
                              initargs=(IDX_BASELINE, IDX_RECENT)) as pool:
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
                print(f"[stage2b-naip] chunk {chunk_id} CRASH: {type(e).__name__}: {e}", flush=True)
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
                f"[stage2b-naip] chunk {chunk_id:>5}  done={chunks_done}/{len(pending)}  "
                f"cand={cands_done:,}  err={err_n}  rate={rate:.1f}/s  eta={eta_min:.0f}m  "
                f"p50={tel['t_p50_per_cand_s']:.2f}s  rss={rolling['rss_gb']:.1f}G",
                flush=True,
            )

            if S3_BUCKET:
                os.system(
                    f"aws s3 cp {tel['out_path']} "
                    f"s3://{S3_BUCKET}/v3-pt3-artifacts/stage2b_naip_chunks{RUN_TAG}/ "
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

    print(f"[stage2b-naip] DONE  chunks_done={chunks_done}  cand_done={cands_done:,}  err={err_n}  "
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
