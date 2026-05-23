"""Profile what features distinguish Overture-missed sites from Overture-found.

Compare the 200 sites Overture missed (>500m to nearest industrial building)
vs the 76 sites Overture found across categorical features in the
manufacturing announcements CSV.
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parents[3]
df = pd.read_csv(ROOT / "data_us" / "phase3_joint_coverage.csv")

df["overture_found"] = df["nearest_overture_m"] <= 500

ann = pd.read_csv(ROOT / "data_us" / "manufacturing_announcements_geocoded.csv")
df = df.merge(ann[["canonical_project_name", "sector", "site_type", "status_current", "announced_completion_date", "announcement_date"]],
              on="canonical_project_name", how="left", suffixes=("", "_a"))

print(f"total positives: {len(df)}")
print(f"overture-found (<=500m): {df['overture_found'].sum()}")
print(f"overture-missed: {(~df['overture_found']).sum()}")
print()


def crosstab(col):
    print(f"=== {col} ===")
    ct = pd.crosstab(df[col].fillna("NULL"), df["overture_found"], margins=True)
    ct.columns = ["missed", "found", "total"] if False else ct.columns
    ct["miss_rate"] = (ct[False] / (ct[False] + ct[True])).round(2) if False in ct.columns else None
    if False in ct.columns and True in ct.columns:
        ct = ct.assign(miss_rate=(ct[False] / (ct[False] + ct[True])).round(3))
        ct = ct.sort_values("miss_rate", ascending=False)
    print(ct.to_string())
    print()


for c in ["sector", "site_type", "status_current", "state"]:
    crosstab(c)

# parent_company top frequencies among misses
print("=== parent_company among Overture-missed (top 15) ===")
miss_co = df.loc[~df["overture_found"], "parent_company"].value_counts().head(15)
print(miss_co.to_string())
print()

# completion dates among misses
df["completion_year"] = pd.to_datetime(df["announced_completion_date"], errors="coerce").dt.year
print("=== announced completion year distribution ===")
ct = pd.crosstab(df["completion_year"].fillna("NULL"), df["overture_found"])
ct = ct.assign(miss_rate=(ct[False] / (ct[False] + ct[True])).round(3))
print(ct.sort_index().to_string())
