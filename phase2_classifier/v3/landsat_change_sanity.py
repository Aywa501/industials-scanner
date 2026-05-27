"""Sanity-check Landsat 2008 vs 2022 change via 2022-derived footprint mask.

For each candidate:
  1. Fetch L8 2022 pan (15m) Jun-Aug median composite over bbox + 200m margin.
  2. Derive building footprint mask from 2022 via gradient → close → fill →
     connected component containing bbox center.
  3. Fetch L7 2008 pan with the same window.
  4. ratio_y = mean(pan[mask]) / mean(pan[~mask])   # within-year, sensor-invariant
  5. change = |ratio_2022 - ratio_2008|.

Same spatial region for both years; we avoid noisy gradient extraction on 2008.
"""
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(ROOT, ".env"))
os.environ.setdefault("AWS_REQUEST_PAYER", "requester")

import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
from scipy.ndimage import sobel, binary_closing, binary_fill_holes, label
from pystac_client import Client

STAC_URL = "https://landsatlook.usgs.gov/stac-server"

GDAL_KNOBS = dict(
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    GDAL_HTTP_MULTIPLEX="YES",
    GDAL_HTTP_VERSION="2",
    GDAL_HTTP_TIMEOUT="30",
    GDAL_HTTP_MAX_RETRY="5",
    CPL_VSIL_CURL_USE_HEAD="NO",
    CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".TIF,.tif",
    AWS_REQUEST_PAYER="requester",
    VSI_CACHE="TRUE",
    VSI_CACHE_SIZE="536870912",
)

MANIFEST = os.path.join(ROOT, "..", "data_us/phase2/v3_scan_manifest.parquet")
SCAN = "/tmp/v3_check/scan_results.parquet"
OUT = os.path.join(ROOT, "..", "data_us/phase2/v3/landsat_change_sanity.parquet")

MARGIN_M = 15   # 1 pan pixel buffer — just enough for Sobel kernel at bbox edges
GRAD_PERCENTILE = 75
MAX_SCENES = 12


def _s3_href(asset):
    alt = asset.extra_fields.get("alternate", {}).get("s3", {})
    return alt.get("href") or asset.href


def _expand_bbox_ll(bbox_ll, margin_m):
    cy = (bbox_ll[1] + bbox_ll[3]) / 2
    dlat = margin_m / 111_000
    dlon = margin_m / (111_000 * np.cos(np.radians(cy)))
    return (bbox_ll[0] - dlon, bbox_ll[1] - dlat, bbox_ll[2] + dlon, bbox_ll[3] + dlat)


def fetch_pan_median(bbox_ll, year, platform):
    items = list(Client.open(STAC_URL).search(
        collections=["landsat-c2l1"],
        bbox=list(bbox_ll),
        datetime=f"{year}-06-01/{year}-08-31",
        query={"platform": {"eq": platform}, "eo:cloud_cover": {"lt": 40}},
        max_items=MAX_SCENES,
    ).items())
    if not items:
        return None, 0
    stack = []
    with rasterio.Env(**GDAL_KNOBS):
        for it in items:
            pan = it.assets.get("pan")
            if not pan:
                continue
            try:
                with rasterio.open(_s3_href(pan)) as src:
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


def extract_footprint_mask(pan, bbox_ll, expanded_ll):
    """Gradient → close → fill → connected component at bbox center. Returns binary mask."""
    if pan is None or np.isnan(pan).all():
        return None
    imp = pan.copy()
    if np.isnan(imp).any():
        imp[np.isnan(imp)] = np.nanmean(imp)
    grad = np.hypot(sobel(imp, axis=0), sobel(imp, axis=1))
    g_max = np.percentile(grad, 99)
    if g_max <= 0:
        return None
    grad_n = grad / g_max
    filled = binary_fill_holes(binary_closing(grad_n > np.percentile(grad_n, GRAD_PERCENTILE), iterations=2))
    labels, n = label(filled)
    if n == 0:
        return None

    bx, by = (bbox_ll[0] + bbox_ll[2]) / 2, (bbox_ll[1] + bbox_ll[3]) / 2
    ex0, ey0, ex1, ey1 = expanded_ll
    h_px, w_px = pan.shape
    px = max(0, min(w_px - 1, int(round((bx - ex0) / (ex1 - ex0) * w_px))))
    py = max(0, min(h_px - 1, int(round((ey1 - by) / (ey1 - ey0) * h_px))))
    center_label = labels[py, px]
    if center_label == 0:
        r = 3
        local = labels[max(0, py - r):py + r + 1, max(0, px - r):px + r + 1]
        nz = local[local > 0]
        if nz.size == 0:
            return None
        vals, counts = np.unique(nz, return_counts=True)
        center_label = int(vals[np.argmax(counts)])
    return labels == center_label


def ratio_in_out(pan, mask):
    """mean(pan[mask]) / mean(pan[~mask]). Within-year, sensor-invariant."""
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


def process_row(idx_row):
    i, r = idx_row
    bbox = (r["xmin"], r["ymin"], r["xmax"], r["ymax"])
    expanded = _expand_bbox_ll(bbox, MARGIN_M)
    try:
        pan_22, n_22 = fetch_pan_median(expanded, 2022, "LANDSAT_8")
        mask = extract_footprint_mask(pan_22, bbox, expanded)
        if mask is None:
            return i, {"error": "no footprint in 2022"}

        pan_08, n_08 = fetch_pan_median(expanded, 2008, "LANDSAT_7")
        if pan_08 is None:
            return i, {"error": "no 2008 data"}

        h = min(pan_08.shape[0], pan_22.shape[0], mask.shape[0])
        w = min(pan_08.shape[1], pan_22.shape[1], mask.shape[1])
        pan_08, pan_22, mask = pan_08[:h, :w], pan_22[:h, :w], mask[:h, :w]

        r_08 = ratio_in_out(pan_08, mask)
        r_22 = ratio_in_out(pan_22, mask)
        change = abs(r_22 - r_08) if (r_08 is not None and r_22 is not None) else None
        return i, {
            "ratio_2008": r_08, "ratio_2022": r_22, "change": change,
            "footprint_pixels": int(mask.sum()),
            "n_scenes_2008": n_08, "n_scenes_2022": n_22,
        }
    except Exception as e:
        return i, {"error": str(e)}


def main():
    df = pd.read_parquet(SCAN).dropna(subset=["p_dino_sat493m"])
    mf = pd.read_parquet(MANIFEST)[["building_id", "xmin", "xmax", "ymin", "ymax"]]
    df = df.merge(mf, on="building_id")

    high = df[df["p_dino_sat493m"] >= 0.90].sample(30, random_state=7)
    mid = df[(df["p_dino_sat493m"] >= 0.50) & (df["p_dino_sat493m"] < 0.90)].sample(30, random_state=7)
    sample = pd.concat([high.assign(band="high"), mid.assign(band="mid")], ignore_index=True)

    out = [None] * len(sample)
    with ThreadPoolExecutor(max_workers=6) as pool:
        futs = {pool.submit(process_row, (i, r)): i for i, r in sample.iterrows()}
        done = 0
        for fut in as_completed(futs):
            i, result = fut.result()
            r = sample.iloc[i]
            out[i] = {**r.to_dict(), **result}
            done += 1
            if result.get("error"):
                print(f"{done:3d}/{len(sample)} ERROR i={i}: {result['error'][:120]}")
            else:
                print(f"{done:3d}/{len(sample)} {r['band']} p={r['p_dino_sat493m']:.3f} "
                      f"r2008={result['ratio_2008']:.3f} r2022={result['ratio_2022']:.3f} "
                      f"change={result['change']:.3f} fp_px={result['footprint_pixels']}")

    res = pd.DataFrame(out)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    res.to_parquet(OUT, index=False)
    print(f"\nwrote {OUT}: {len(res)} rows")

    valid = res.dropna(subset=["change"]).copy()
    print(f"\nvalid: {len(valid)}/{len(res)}")
    print(f"\nchange distribution:")
    print(valid["change"].describe())
    print("\ndrop fraction by threshold (drop iff change < T):")
    for T in [0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50]:
        drop = (valid["change"] < T).sum()
        print(f"  T={T:.2f}: drop {drop}/{len(valid)} ({100*drop/len(valid):.0f}%)")

    print("\n--- MOST STABLE (smallest change) ---")
    print(valid.nsmallest(15, "change")[["band", "p_dino_sat493m", "lat", "lon", "ratio_2008", "ratio_2022", "change", "footprint_pixels"]].to_string(index=False))
    print("\n--- MOST CHANGED ---")
    print(valid.nlargest(10, "change")[["band", "p_dino_sat493m", "lat", "lon", "ratio_2008", "ratio_2022", "change", "footprint_pixels"]].to_string(index=False))


if __name__ == "__main__":
    main()
