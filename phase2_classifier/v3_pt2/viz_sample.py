"""Visualize a random 25-polygon sample: 2008 L7 | 2022 L8 | recent NAIP — one
3-image set per PDF page, polygon outline in yellow, ~500m window.

L7 2008 SR  -> SR_B3 (R), SR_B2 (G), SR_B1 (B)
L8 2022 SR  -> SR_B4 (R), SR_B3 (G), SR_B2 (B)
NAIP        -> tile selected from naip_tile_index.parquet (most recent year that
               covers the bbox), bands 1/2/3 = R/G/B (RGBIR COG).
Derive Landsat SR paths from existing L1 PAN scene index via:
   level-1 -> level-2 ;  _L1TP_ -> _L2SP_ ;  _B8.TIF -> _SR_B{N}.TIF
"""
import os, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
from shapely.wkb import loads as wkb_loads
from shapely.geometry import mapping
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ.setdefault("AWS_REQUEST_PAYER", "requester")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_USE_HEAD", "NO")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CANDS = os.path.join(ROOT, "..", "data_us/phase2/v3/stage2_candidates.parquet")
POLYS = os.path.join(ROOT, "..", "data_us/phase2/v3/stage2_candidate_polygons.parquet")
SCENES = os.path.join(ROOT, "..", "data_us/phase2/v3/landsat_scenes_index.parquet")
NAIP_IDX = os.path.join(ROOT, "..", "data_us/phase3_naip/naip_tile_index.parquet")
OUT_PDF = os.path.join(ROOT, "..", "data_us/phase2/v3/viz_sample_2008_2022_naip.pdf")

WINDOW_M = 500            # half-side of the displayed window
N_SAMPLE = 25
SEED = 7

L7_BAND_NUM = {"red": 3, "green": 2, "blue": 1}
L8_BAND_NUM = {"red": 4, "green": 3, "blue": 2}


def l1_pan_to_sr(href, band_num):
    return (href
            .replace("/level-1/", "/level-2/")
            .replace("_L1TP_", "_L2SP_")
            .replace("_L1GT_", "_L2SP_")
            .replace("_B8.TIF", f"_SR_B{band_num}.TIF"))


def fetch_rgb(scene_hrefs, bbox_ll, band_map):
    """Median RGB over a list of L1-PAN hrefs, mapped to SR bands. Returns (H,W,3) float [0,1]."""
    chans = {b: [] for b in ("red", "green", "blue")}
    with rasterio.Env():
        for href in scene_hrefs:
            for b, num in band_map.items():
                sr = l1_pan_to_sr(href, num)
                try:
                    with rasterio.open(sr) as src:
                        bbox_utm = transform_bounds("EPSG:4326", src.crs, *bbox_ll)
                        win = from_bounds(*bbox_utm, transform=src.transform).round_offsets().round_lengths()
                        if win.width < 4 or win.height < 4:
                            continue
                        arr = src.read(1, window=win, boundless=True, fill_value=0).astype(np.float32)
                    refl = arr * 2.75e-5 - 0.2
                    refl[arr == 0] = np.nan
                    refl[(refl < 0) | (refl > 1)] = np.nan
                    chans[b].append(refl)
                except Exception:
                    continue
    out = {}
    for b in ("red", "green", "blue"):
        if not chans[b]:
            return None
        h = min(a.shape[0] for a in chans[b])
        w = min(a.shape[1] for a in chans[b])
        out[b] = np.nanmedian(np.stack([a[:h, :w] for a in chans[b]], axis=0), axis=0)
    h = min(out[b].shape[0] for b in out)
    w = min(out[b].shape[1] for b in out)
    rgb = np.stack([out["red"][:h, :w], out["green"][:h, :w], out["blue"][:h, :w]], axis=-1)
    return rgb


def stretch(rgb, p_lo=2, p_hi=98):
    """Per-channel percentile stretch, clip to [0,1]. NaNs -> 0."""
    if rgb is None:
        return None
    out = np.zeros_like(rgb)
    for c in range(3):
        ch = rgb[..., c]
        finite = ch[np.isfinite(ch)]
        if len(finite) < 10:
            continue
        lo, hi = np.percentile(finite, [p_lo, p_hi])
        if hi <= lo:
            hi = lo + 1e-3
        out[..., c] = np.clip((ch - lo) / (hi - lo), 0, 1)
    out[np.isnan(out)] = 0
    return out


def bbox_around_centroid(cy, cx, half_m=WINDOW_M):
    dlat = half_m / 111_000
    dlon = half_m / (111_000 * np.cos(np.radians(cy)))
    return (cx - dlon, cy - dlat, cx + dlon, cy + dlat)


def fetch_naip(naip_idx, bbox_ll):
    """Pick the most recent NAIP tile fully or partially covering bbox_ll, fetch RGB.
    Returns (H,W,3) float [0,1] or None."""
    lon_lo, lat_lo, lon_hi, lat_hi = bbox_ll
    # Tiles intersecting the bbox
    cand = naip_idx[
        (naip_idx["lon_min"] <= lon_hi) & (naip_idx["lon_max"] >= lon_lo) &
        (naip_idx["lat_min"] <= lat_hi) & (naip_idx["lat_max"] >= lat_lo)
    ]
    if len(cand) == 0:
        return None, None
    # Prefer most recent acquisition; pick tile whose center is closest to bbox center
    cy = (lat_lo + lat_hi) / 2; cx = (lon_lo + lon_hi) / 2
    cand = cand.copy()
    cand["d"] = ((cand["lon_min"] + cand["lon_max"]) / 2 - cx) ** 2 + \
                ((cand["lat_min"] + cand["lat_max"]) / 2 - cy) ** 2
    cand = cand.sort_values(["naip_year", "naip_acq_date", "d"], ascending=[False, False, True])
    for tile in cand.itertuples():
        try:
            with rasterio.Env():
                with rasterio.open(tile.tile_uri) as src:
                    bbox_utm = transform_bounds("EPSG:4326", src.crs, *bbox_ll)
                    win = from_bounds(*bbox_utm, transform=src.transform).round_offsets().round_lengths()
                    if win.width < 8 or win.height < 8:
                        continue
                    arr = src.read([1, 2, 3], window=win, boundless=True, fill_value=0).astype(np.float32)
            if arr.max() == 0:
                continue
            rgb = np.transpose(arr, (1, 2, 0)) / 255.0
            return rgb, f"NAIP {tile.naip_year}"
        except Exception:
            continue
    return None, None


def main():
    print("loading data...")
    cands = pd.read_parquet(CANDS).set_index("building_id")
    polys = pd.read_parquet(POLYS).set_index("ovt_id")
    scenes = pd.read_parquet(SCENES)
    naip_idx = pd.read_parquet(NAIP_IDX)
    print(f"  cands={len(cands):,}  polys={len(polys):,}  scenes={len(scenes):,}  naip_tiles={len(naip_idx):,}")

    scenes_idx = {}
    for (lat, lon, yr, plat), grp in scenes.groupby(["grid_lat", "grid_lon", "year", "platform"], sort=False):
        scenes_idx[(int(lat), int(lon), int(yr), plat)] = grp["s3_href"].tolist()

    rng = np.random.default_rng(SEED)
    valid = cands.dropna(subset=["lat", "lon"]).copy()
    valid = valid[valid["ovt_id"].isin(polys.index)]
    sample = valid.sample(n=N_SAMPLE, random_state=SEED)
    print(f"sample of {len(sample)}")

    def _do(i, bid, row):
        lat, lon = float(row["lat"]), float(row["lon"])
        bbox_ll = bbox_around_centroid(lat, lon)
        cell = (int(np.floor(lat)), int(np.floor(lon)))
        h08 = scenes_idx.get((cell[0], cell[1], 2008, "LANDSAT_7"), [])
        h22 = scenes_idx.get((cell[0], cell[1], 2022, "LANDSAT_8"), [])
        rgb_08 = stretch(fetch_rgb(h08, bbox_ll, L7_BAND_NUM)) if h08 else None
        rgb_22 = stretch(fetch_rgb(h22, bbox_ll, L8_BAND_NUM)) if h22 else None
        rgb_naip, naip_label = fetch_naip(naip_idx, bbox_ll)
        return i, bid, row, bbox_ll, rgb_08, rgb_22, rgb_naip, naip_label

    t0 = time.time()
    items = list(enumerate(sample.itertuples()))
    results = [None] * N_SAMPLE
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(_do, i, c.Index, sample.loc[c.Index]) for i, c in items]
        for done, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            results[r[0]] = r[1:]
            print(f"  [{done}/{N_SAMPLE}]  {r[1]}  elapsed={time.time()-t0:.0f}s", flush=True)

    print(f"all fetches done in {time.time()-t0:.0f}s — rendering PDF...")

    with PdfPages(OUT_PDF) as pdf:
        for i, (bid, row, bbox_ll, rgb_08, rgb_22, rgb_naip, naip_label) in enumerate(results):
            ovt_id = row["ovt_id"]
            poly = wkb_loads(polys.loc[ovt_id]["geometry_wkb"])
            geoms = list(poly.geoms) if poly.geom_type == "MultiPolygon" else [poly]

            fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
            panels = [
                (rgb_08, "2008 L7 SR"),
                (rgb_22, "2022 L8 SR"),
                (rgb_naip, naip_label or "NAIP (no tile)"),
            ]
            for j, (rgb, label) in enumerate(panels):
                ax = axes[j]
                if rgb is None:
                    ax.text(0.5, 0.5, f"{label}\n(no data)", ha="center", va="center", transform=ax.transAxes)
                    ax.set_xticks([]); ax.set_yticks([])
                    continue
                ax.imshow(rgb, extent=(bbox_ll[0], bbox_ll[2], bbox_ll[1], bbox_ll[3]), origin="upper")
                for g in geoms:
                    x, y = g.exterior.xy
                    ax.plot(x, y, color="yellow", linewidth=1.5)
                    for interior in g.interiors:
                        xi, yi = interior.xy
                        ax.plot(xi, yi, color="yellow", linewidth=1.0)
                ax.set_xlim(bbox_ll[0], bbox_ll[2])
                ax.set_ylim(bbox_ll[1], bbox_ll[3])
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(label, fontsize=10)

            p_d = row.get("p_dino_sat493m")
            area = row.get("approx_area_m2")
            cls = row.get("ovt_class")
            fig.suptitle(
                f"{bid}   lat={float(row['lat']):.4f} lon={float(row['lon']):.4f}   "
                f"p_dino={p_d:.2f}   area~{int(area)}m²   class={cls}",
                fontsize=10,
            )
            plt.tight_layout(rect=(0, 0, 1, 0.95))
            pdf.savefig(fig)
            plt.close(fig)
    print(f"wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
