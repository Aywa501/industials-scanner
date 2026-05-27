"""Assemble the Stage 1 industrial classifier training pool.

Reads:
- gs://{GCS_BUCKET}/manifest/s2_chip_manifest.parquet  (all S2 chips)
- data_us/labels/manufacturing_announcements_geocoded.csv     (anchor metadata, raw)
- data_us/labels/manual_labels.parquet                        (greenfield labels)
- data_us/labels/manual_site_notes.parquet                    (IMPORTANT overrides + bad-geocode notes)

Writes:
- data_us/phase1/stage1_dataset.parquet  columns: site_id, year, tile_uri, label,
                                            site_type, source

Rules (see sites_us/phase2_classifier/PLAN.md):
- Drop sites with city-level geocodes (min decimal places of lat/lng < 4).
- IMPORTANT notes override CSV site_type. "demolish" / "cut off" -> drop site.
- Positives:
    brownfield × all years
    expansion_existing × all years
    greenfield: only chips manually labeled complete or partial
    (overrides count: a greenfield reclassified extension/brownfield becomes
     full × all years.)
- Negative candidate pool (no NN filter applied here; that's filter_negatives.py):
    site_type=='negative' × all years
    + manually-labeled not_a_site chips
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

GCS_BUCKET = os.getenv("GCS_BUCKET", "")
MANIFEST_URI = f"gs://{GCS_BUCKET}/manifest/s2_chip_manifest.parquet"

DATA_US = ROOT.parent / "data_us"
ANCHORS_CSV = DATA_US / "labels" / "manufacturing_announcements_geocoded.csv"
LABELS_PATH = DATA_US / "labels" / "manual_labels.parquet"
NOTES_PATH = DATA_US / "labels" / "manual_site_notes.parquet"
OUT_PATH = DATA_US / "phase1" / "stage1_dataset.parquet"

MIN_GEOCODE_DP = 4
POSITIVE_GREENFIELD_LABELS = {"complete", "partial"}
FULL_HISTORY_TYPES = {"brownfield", "expansion_existing"}


def parse_overrides(notes: pd.DataFrame) -> tuple[dict[str, str], set[str]]:
    """Parse IMPORTANT-prefixed notes.

    Returns:
        type_overrides: site_id -> new site_type ('brownfield'|'expansion_existing')
        drop_sites:    site_ids to drop entirely (demolition, bad geocode, etc.)
    """
    type_overrides: dict[str, str] = {}
    drop_sites: set[str] = set()
    for _, r in notes.iterrows():
        note = (r.note or "").strip()
        nl = note.lower()
        if "demolish" in nl:
            drop_sites.add(r.site_id)
            continue
        if "cut off" in nl or "not centered" in nl or "bottom right corner" in nl:
            drop_sites.add(r.site_id)
            continue
        if not nl.startswith("important"):
            continue
        if "extention" in nl or "extension" in nl or "expansion" in nl:
            type_overrides[r.site_id] = "expansion_existing"
        elif "brownfield" in nl:
            type_overrides[r.site_id] = "brownfield"
        elif "greenfield" in nl:
            type_overrides[r.site_id] = "greenfield"
    return type_overrides, drop_sites


def low_precision_names(csv_path: Path) -> set[str]:
    raw = pd.read_csv(csv_path, dtype={"lat": str, "lng": str})

    def _dp(s):
        return len(s.split(".", 1)[1]) if isinstance(s, str) and "." in s else 0

    raw["min_dp"] = raw.apply(lambda r: min(_dp(r.lat), _dp(r.lng)), axis=1)
    return set(raw.loc[raw.min_dp < MIN_GEOCODE_DP, "canonical_project_name"])


def main() -> int:
    if not GCS_BUCKET:
        print("error: GCS_BUCKET not set", file=sys.stderr)
        return 1

    print(f"reading manifest from {MANIFEST_URI}")
    manifest = pd.read_parquet(MANIFEST_URI)
    completed = manifest[manifest.export_status == "COMPLETED"].copy()
    print(f"  {len(manifest)} rows, {len(completed)} completed")

    anchors_csv = pd.read_csv(ANCHORS_CSV)
    type_by_name = dict(
        zip(anchors_csv.canonical_project_name, anchors_csv.site_type)
    )

    low_prec_names = low_precision_names(ANCHORS_CSV)
    print(f"  {len(low_prec_names)} canonical_project_names have city-level geocodes")

    notes = pd.read_parquet(NOTES_PATH) if NOTES_PATH.exists() else pd.DataFrame(
        columns=["site_id", "note"]
    )
    type_overrides, drop_sites = parse_overrides(notes)
    print(f"  IMPORTANT overrides: {type_overrides}")
    print(f"  notes-driven drops: {sorted(drop_sites)}")

    labels = pd.read_parquet(LABELS_PATH)
    print(f"  manual labels: {len(labels)} rows / {labels.site_id.nunique()} sites")

    # ---- Anchor view: site_type per (site_id, canonical_project_name).
    # Effective site_type = override > CSV type.
    anchors = completed[completed.site_type == "anchor"].copy()
    anchors["csv_type"] = anchors.canonical_project_name.map(type_by_name)
    anchors["eff_type"] = anchors.apply(
        lambda r: type_overrides.get(r.site_id, r.csv_type), axis=1
    )

    # Apply drops + low-precision filter to anchors only.
    anchors = anchors[~anchors.site_id.isin(drop_sites)]
    anchors = anchors[~anchors.canonical_project_name.isin(low_prec_names)]
    print(f"  anchors after drops + geocode filter: "
          f"{anchors.site_id.nunique()} sites, {len(anchors)} chips")

    # ---- Positives.
    pos_full = anchors[anchors.eff_type.isin(FULL_HISTORY_TYPES)].copy()
    pos_full["label"] = "industrial"
    pos_full["source"] = "type_full_history"

    # Greenfield: keep only chips with manual label complete/partial.
    pos_label_keys = set(
        labels[labels.label.isin(POSITIVE_GREENFIELD_LABELS)]
        .apply(lambda r: (r.site_id, int(r.year)), axis=1)
        .tolist()
    )
    green = anchors[anchors.eff_type == "greenfield"].copy()
    green["key"] = list(zip(green.site_id, green.year.astype(int)))
    pos_green = green[green.key.isin(pos_label_keys)].drop(columns=["key"]).copy()
    pos_green["label"] = "industrial"
    pos_green["source"] = "greenfield_labeled"

    pos = pd.concat([pos_full, pos_green], ignore_index=True)
    print(f"  positives: {len(pos)} chips "
          f"({len(pos_full)} full-history + {len(pos_green)} greenfield-labeled)")
    print(f"    by eff_type:\n{pos.eff_type.value_counts().to_string()}")

    # ---- Negative candidate pool (NN filter applied later).
    # Drop any random-CONUS site where the user manually confirmed industrial
    # activity (complete/partial labels) — common in the relabel-shortlist pass
    # where we discovered real industrial sites in the random pool.
    confirmed_industrial_sites = set(
        labels[(labels.label.isin(POSITIVE_GREENFIELD_LABELS))
               & (labels.site_id.str.startswith("n_"))]
        .site_id.unique()
    )
    if confirmed_industrial_sites:
        print(f"  dropping random-CONUS sites confirmed-industrial by user: "
              f"{sorted(confirmed_industrial_sites)}")
    neg_random = completed[
        (completed.site_type == "negative")
        & (~completed.site_id.isin(confirmed_industrial_sites))
    ].copy()
    neg_random["eff_type"] = "random_conus"
    neg_random["label"] = "candidate_negative"
    neg_random["source"] = "random_conus"

    # Manually-confirmed not_a_site chips: take from anchors+random where label==not_a_site.
    # If a site ended up as a positive (e.g. via IMPORTANT-note override flipping
    # greenfield→brownfield), the type_full_history rule wins — drop those (site_id, year)
    # pairs from the manual_negative pool to avoid double-counting.
    pos_keys = set(zip(pos.site_id, pos.year.astype(int)))
    not_site_keys = set(
        labels[labels.label == "not_a_site"]
        .apply(lambda r: (r.site_id, int(r.year)), axis=1)
        .tolist()
    )
    not_site_keys = not_site_keys - pos_keys
    cand = completed.copy()
    cand["key"] = list(zip(cand.site_id, cand.year.astype(int)))
    confirmed_neg = cand[cand.key.isin(not_site_keys)].drop(columns=["key"]).copy()
    # Drop any that ended up dropped above.
    confirmed_neg = confirmed_neg[~confirmed_neg.site_id.isin(drop_sites)]
    confirmed_neg = confirmed_neg[
        ~confirmed_neg.canonical_project_name.isin(low_prec_names)
        | confirmed_neg.canonical_project_name.isna()
    ]
    confirmed_neg["eff_type"] = "manual_negative"
    confirmed_neg["label"] = "manual_negative"
    confirmed_neg["source"] = "manual_not_a_site"
    print(f"  random-CONUS candidates: {len(neg_random)}")
    print(f"  manual not_a_site chips: {len(confirmed_neg)}")

    # ---- Drop manual_negative chips from random pool to avoid duplicates.
    confirmed_keys = set(
        zip(confirmed_neg.site_id, confirmed_neg.year.astype(int))
    )
    neg_random = neg_random[
        ~neg_random.apply(
            lambda r: (r.site_id, int(r.year)) in confirmed_keys, axis=1
        )
    ]

    # ---- Assemble.
    cols = ["site_id", "year", "tile_uri", "label", "eff_type", "source",
            "canonical_project_name"]
    out = pd.concat(
        [pos[cols], neg_random[cols], confirmed_neg[cols]], ignore_index=True
    ).rename(columns={"eff_type": "site_type"})
    out["year"] = out.year.astype(int)
    out = out.drop_duplicates(["site_id", "year"]).reset_index(drop=True)

    print(f"\nfinal dataset: {len(out)} chips, {out.site_id.nunique()} sites")
    print(out.label.value_counts().to_string())
    print()
    print("by site_type:")
    print(out.groupby(["label", "site_type"]).size().to_string())

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    print(f"\nwrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
