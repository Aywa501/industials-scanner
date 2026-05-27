"""Fetch and visualize Landsat 2008 + 2022 pan for one candidate lat/lon.

Usage: python inspect_one.py <lat> <lon> [<building_id_for_bbox>]

If building_id given, uses Overture bbox; otherwise builds a small square
(40m × 40m) around lat/lon for visualization.
"""
import os
import sys
import numpy as np
import pandas as pd
from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(ROOT, ".env"))
os.environ.setdefault("AWS_REQUEST_PAYER", "requester")

import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
from pystac_client import Client
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

STAC_URL = "https://landsatlook.usgs.gov/stac-server"
GDAL_KNOBS = dict(
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    GDAL_HTTP_MULTIPLEX="YES",
    GDAL_HTTP_VERSION="2",
    GDAL_HTTP_TIMEOUT="30",
    GDAL_HTTP_MAX_RETRY="5",
    CPL_VSIL_CURL_USE_HEAD="NO",
    AWS_REQUEST_PAYER="requester",
)
MARGIN_M = 250


def _s3_href(a):
    return a.extra_fields.get("alternate", {}).get("s3", {}).get("href") or a.href


def _expand(bbox_ll, m):
    cy = (bbox_ll[1] + bbox_ll[3]) / 2
    dlat = m / 111_000
    dlon = m / (111_000 * np.cos(np.radians(cy)))
    return (bbox_ll[0] - dlon, bbox_ll[1] - dlat, bbox_ll[2] + dlon, bbox_ll[3] + dlat)


def fetch_pan(bbox_ll, year, platform):
    items = list(Client.open(STAC_URL).search(
        collections=["landsat-c2l1"],
        bbox=list(bbox_ll),
        datetime=f"{year}-06-01/{year}-08-31",
        query={"platform": {"eq": platform}, "eo:cloud_cover": {"lt": 40}},
        max_items=12,
    ).items())
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
                    arr = src.read(1, window=win, boundless=True, fill_value=0).astype(np.float32)
                arr[arr == 0] = np.nan
                if not np.isnan(arr).all():
                    stack.append(arr)
            except Exception:
                pass
    if not stack:
        return None, 0
    h = min(a.shape[0] for a in stack)
    w = min(a.shape[1] for a in stack)
    return np.nanmedian(np.stack([a[:h, :w] for a in stack], axis=0), axis=0), len(stack)


def main():
    lat, lon = float(sys.argv[1]), float(sys.argv[2])
    building_id = sys.argv[3] if len(sys.argv) > 3 else None

    if building_id:
        mf = pd.read_parquet(os.path.join(ROOT, "..", "data_us/phase2/v3_scan_manifest.parquet"))
        row = mf[mf["building_id"] == building_id].iloc[0]
        bbox = (row["xmin"], row["ymin"], row["xmax"], row["ymax"])
    else:
        d = 20 / 111_000
        bbox = (lon - d, lat - d, lon + d, lat + d)

    expanded = _expand(bbox, MARGIN_M)
    print(f"lat,lon = {lat}, {lon}")
    print(f"bbox = {bbox}")
    print(f"window = {expanded}")
    print(f"\nFetching L7 2008 ...")
    pan_08, n_08 = fetch_pan(expanded, 2008, "LANDSAT_7")
    print(f"  {n_08} scenes")
    print(f"Fetching L8 2022 ...")
    pan_22, n_22 = fetch_pan(expanded, 2022, "LANDSAT_8")
    print(f"  {n_22} scenes")

    # Compute bbox mask and ratios
    def ratio(pan):
        if pan is None or np.isnan(pan).all():
            return None, None, None, None
        ex0, ey0, ex1, ey1 = expanded
        bx0, by0, bx1, by1 = bbox
        h, w = pan.shape
        px0 = max(0, int(round((bx0 - ex0) / (ex1 - ex0) * w)))
        px1 = min(w, int(round((bx1 - ex0) / (ex1 - ex0) * w)))
        py0 = max(0, int(round((ey1 - by1) / (ey1 - ey0) * h)))
        py1 = min(h, int(round((ey1 - by0) / (ey1 - ey0) * h)))
        bbox_mask = np.zeros_like(pan, dtype=bool)
        bbox_mask[py0:py1, px0:px1] = True
        bv = pan[bbox_mask & ~np.isnan(pan)]
        rv = pan[~bbox_mask & ~np.isnan(pan)]
        if len(bv) == 0 or len(rv) == 0:
            return None, None, None, (px0, py0, px1, py1)
        return float(bv.mean()), float(rv.mean()), float(bv.mean()) / float(rv.mean()), (px0, py0, px1, py1)

    b_08, r_08, ratio_08, rect_08 = ratio(pan_08)
    b_22, r_22, ratio_22, rect_22 = ratio(pan_22)

    print(f"\n2008: bbox_mean={b_08!r}  ring_mean={r_08!r}  ratio={ratio_08!r}")
    print(f"2022: bbox_mean={b_22!r}  ring_mean={r_22!r}  ratio={ratio_22!r}")
    if ratio_08 and ratio_22:
        print(f"change = |{ratio_22:.3f} - {ratio_08:.3f}| = {abs(ratio_22 - ratio_08):.3f}")

    # Visualize
    out_dir = os.path.join(ROOT, "..", "data_us/phase2/v3/landsat_viz")
    os.makedirs(out_dir, exist_ok=True)
    fname = f"inspect_{lat:.4f}_{lon:.4f}.png"
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    for ax, pan, rect, year, ratio_v in [
        (axes[0], pan_08, rect_08, 2008, ratio_08),
        (axes[1], pan_22, rect_22, 2022, ratio_22),
    ]:
        if pan is None:
            ax.set_title(f"{year}: NO DATA"); continue
        im = ax.imshow(pan, cmap="gray")
        if rect:
            px0, py0, px1, py1 = rect
            ax.add_patch(mpatches.Rectangle((px0, py0), px1 - px0, py1 - py0, fill=False, edgecolor="red", linewidth=2))
        ax.set_title(f"{year} pan median  ratio(bbox/ring) = {ratio_v:.3f}" if ratio_v else f"{year}")
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"{lat:.4f}, {lon:.4f}")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, fname), dpi=90)
    plt.close(fig)
    print(f"\nwrote {os.path.join(out_dir, fname)}")
    print(f"\nGoogle Earth Web (historical imagery slider available): https://earth.google.com/web/@{lat},{lon},300a,1500d,35y,0h,0t,0r")
    print(f"Google Maps satellite: https://www.google.com/maps/@{lat},{lon},18z/data=!3m1!1e3")


if __name__ == "__main__":
    main()
