"""Filter v3 scan results to Stage 2b candidate set @ p_dino_sat493m >= 0.30.

91% recall on industrial>=5000m2 set (n=32,983), 50% cut on the full scan.
Output: data_us/phase2/v3/stage2_candidates.parquet
"""
import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCAN_LOCAL = "/tmp/v3_check/scan_results.parquet"
SCAN_S3 = "s3://industrials-scanner-us-west-2/v3-artifacts/v3/scan_chunks/_scores/"
MANIFEST = os.path.join(ROOT, "..", "data_us/phase2/v3_scan_manifest.parquet")
OUT = os.path.join(ROOT, "..", "data_us/phase2/v3/stage2_candidates.parquet")

THR = 0.30


def main():
    if not os.path.exists(SCAN_LOCAL):
        raise SystemExit(
            f"missing {SCAN_LOCAL}. Sync first:\n"
            f"  aws s3 sync {SCAN_S3} /tmp/v3_check/scan_chunks/ "
            f"&& python -c \"import pandas as pd, glob; "
            f"pd.concat([pd.read_parquet(f) for f in glob.glob('/tmp/v3_check/scan_chunks/*.parquet')])"
            f".to_parquet('{SCAN_LOCAL}', index=False)\""
        )

    df = pd.read_parquet(SCAN_LOCAL).dropna(subset=["p_dino_sat493m"])
    print(f"scored:       {len(df):,}")

    kept = df[df["p_dino_sat493m"] >= THR].copy()
    print(f"kept @ {THR}:  {len(kept):,} ({len(kept)/len(df):.1%})")

    mf = pd.read_parquet(MANIFEST)[["building_id", "xmin", "xmax", "ymin", "ymax"]]
    out = kept.merge(mf, on="building_id", how="inner")
    print(f"with bbox:    {len(out):,}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"wrote {OUT}: {len(out):,} candidates, {len(out.columns)} cols")
    print(f"cols: {list(out.columns)}")


if __name__ == "__main__":
    main()
