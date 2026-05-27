"""Sample 30 buildings from each of 3 probability bands of the 10K validation scan.
Adds Google Maps + NAIP-tile-list URLs so the user can eyeball each."""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data_us"
SCAN_RESULTS = DATA / "phase2" / "v3" / "scan_validate" / "scan_results.parquet"
SCENES = DATA / "phase2" / "v3_scan_scenes_index.parquet"
OUT = DATA / "phase2" / "v3" / "scan_validate" / "sample_bands.csv"

BANDS = [
    ("low_0.50_0.55", 0.50, 0.55),
    ("mid_0.70_0.75", 0.70, 0.75),
    ("high_0.95_1.00", 0.95, 1.00),
]
N_PER_BAND = 30
SEED = 11


def maps_url(lat: float, lon: float, z: int = 18) -> str:
    return f"https://www.google.com/maps/@{lat:.6f},{lon:.6f},{z}z/data=!3m1!1e3"


def main() -> None:
    df = pd.read_parquet(SCAN_RESULTS)
    scenes = pd.read_parquet(SCENES, columns=["building_id", "naip_uris"])
    df = df.merge(scenes, on="building_id", how="left")
    print(f"loaded scan_results: {len(df):,}")
    print(f"columns: {list(df.columns)}")
    print()
    print("=== p_mean distribution ===")
    for t in (0.3, 0.5, 0.7, 0.9, 0.95):
        print(f"  >={t}: {int((df['p_mean'] >= t).sum()):,}")
    print()
    print("=== ovt_class breakdown of high-confidence flags (p_mean>=0.95) ===")
    print(df[df["p_mean"] >= 0.95]["ovt_class"].value_counts(dropna=False).head(15).to_string())
    print()

    rng = np.random.default_rng(SEED)
    parts = []
    for name, lo, hi in BANDS:
        bucket = df[(df["p_mean"] >= lo) & (df["p_mean"] < hi)]
        n_take = min(N_PER_BAND, len(bucket))
        if n_take == 0:
            continue
        sel = bucket.sample(n=n_take, random_state=int(rng.integers(0, 10**9))).copy()
        sel["band"] = name
        parts.append(sel)
        print(f"band {name}: pool={len(bucket):,} sampled={n_take}")

    out = pd.concat(parts, ignore_index=True)
    out["gmaps"] = [maps_url(la, lo) for la, lo in zip(out["lat"], out["lon"])]
    out["naip_uri_first"] = out["naip_uris"].map(
        lambda u: u[0] if isinstance(u, (list, np.ndarray)) and len(u) > 0 else ""
    )

    cols = ["band", "building_id", "lat", "lon", "approx_area_m2", "ovt_class",
            "p_dino_sat493m", "p_dino_vitb", "p_mean", "gmaps", "naip_uri_first"]
    out[cols].to_csv(OUT, index=False)
    print(f"\nwrote {len(out)} rows -> {OUT}")


if __name__ == "__main__":
    main()
