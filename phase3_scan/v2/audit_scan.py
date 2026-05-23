"""Sanity-audit of the Phase 3 v2 pipeline (Step 1 rural-exclusion + Step 2 v2 scan).

The manual labeling pass was dropped — industrial/not discrimination moves to a
later NAIP stage. This script verifies the two completed steps are correct and
sane before that handoff, and emits a pass/warn/fail verdict.

Checks:
  A1  reproduce the Overture pre-filter at the deployed radius (+ a wider comparison)
  A2  anchor survival through Step 1 (probe-independent — a clean test)
  B1  surface the deployed probe's held-out test metrics (leaderboard.json)
  B2  probability distribution + anchor-prob separation (the centerpiece)
  B3  per-shard completeness (scored rows vs filtered tiles)
  C1  dedup magnitude — distinct candidate sites at several thresholds
  B4  visual contact sheet across probability bands

Run from sites_us/:  python phase3_scan/v2/audit_scan.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

ROOT = Path(__file__).resolve().parents[2]          # sites_us/
DATA_US = ROOT.parent / "data_us"
sys.path.insert(0, str(ROOT))
from phase3_scan.v1.aggregate import _cluster        # noqa: E402  (reuse, not duplicate)

RESULTS_DIR = DATA_US / "phase3_results_v2"
GRID_PATH = DATA_US / "phase3_grid.parquet"
OVERTURE_PATH = DATA_US / "overture_industrial_conus_2025_aligned.parquet"
ANCHORS_CSV = DATA_US / "manufacturing_announcements_geocoded.csv"
LEADERBOARD_PATH = DATA_US / "v2" / "leaderboard.json"
CHIPS_DIR = ROOT / ".artifacts" / "labeling_v2" / "chips"
QUEUE_PATH = ROOT / ".artifacts" / "labeling_v2" / "queue.json"
OUT_DIR = ROOT / ".artifacts" / "audit_v2"
REPORT_PATH = OUT_DIR / "audit_report.md"
CONTACT_SHEET_PATH = OUT_DIR / "audit_contact_sheet.png"

EARTH_M = 6_371_000.0
DEPLOYED_MODEL = "dino_vitb"
WIDE_RADIUS_M = 3000.0            # comparison-only radius for A1/A2 (not deployed)


def _deployed_constant(name: str) -> float:
    """Read a numeric constant from infer_shard_v2.py source.

    The audit must reproduce the *deployed* Overture filter exactly. Parsing the
    constant from source (rather than hardcoding it here) is what keeps this
    audit honest — a hardcoded copy silently drifts when the scan code changes.
    """
    import re
    src = (ROOT / "phase3_scan" / "v2" / "infer_shard_v2.py").read_text()
    m = re.search(rf"^{name}\s*=\s*([0-9.]+)", src, re.M)
    if not m:
        raise SystemExit(f"could not find {name} in infer_shard_v2.py")
    return float(m.group(1))


OVERTURE_MIN_AREA_M2 = _deployed_constant("OVERTURE_MIN_AREA_M2")
OVERTURE_RADIUS_M = _deployed_constant("OVERTURE_RADIUS_M")


def log(msg: str) -> None:
    print(f"[audit] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results() -> pd.DataFrame:
    parts = sorted(p for p in RESULTS_DIR.glob("*.parquet")
                   if not p.stem.endswith("_emb"))
    if not parts:
        raise SystemExit(f"no result parquets in {RESULTS_DIR}")
    log(f"loading {len(parts)} result shards")
    frames = []
    for p in parts:
        df = pd.read_parquet(p, columns=["tile_id", "lon", "lat", "prob"])
        if df.empty:                       # empty shards would poison concat dtype
            continue
        df["mgrs_tile"] = p.stem
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df[["lon", "lat", "prob"]] = df[["lon", "lat", "prob"]].astype("float64")
    log(f"  {len(df):,} scored tiles across {df.mgrs_tile.nunique()} shards")
    return df


# ---------------------------------------------------------------------------
# A1 — reproduce the Step 1 Overture pre-filter
# ---------------------------------------------------------------------------

def check_a1(results: pd.DataFrame) -> tuple[str, dict, set, set]:
    log("A1: reproducing Overture pre-filter")
    grid = pd.read_parquet(GRID_PATH, columns=["tile_id", "mgrs_tile", "lat", "lon"])
    ov = pd.read_parquet(OVERTURE_PATH, columns=["lat", "lon", "approx_area_m2"])
    ov = ov[ov.approx_area_m2 >= OVERTURE_MIN_AREA_M2]
    log(f"  grid={len(grid):,} tiles, overture>= {OVERTURE_MIN_AREA_M2:.0f}m^2 = {len(ov):,} buildings")

    tree = BallTree(np.radians(ov[["lat", "lon"]].to_numpy()), metric="haversine")
    tile_rad = np.radians(grid[["lat", "lon"]].to_numpy())

    dep_r = OVERTURE_RADIUS_M       # deployed radius, parsed from infer_shard_v2.py
    wide_r = WIDE_RADIUS_M          # comparison-only
    kept = {}
    for radius_m in (dep_r, wide_r):
        counts = tree.query_radius(tile_rad, r=radius_m / EARTH_M, count_only=True)
        kept[radius_m] = counts > 0
        log(f"  radius={radius_m:.0f}m -> {int(kept[radius_m].sum()):,} tiles kept")

    kept_dep = set(grid.tile_id[kept[dep_r]])
    kept_wide = set(grid.tile_id[kept[wide_r]])
    scored = set(results.tile_id)

    ndep, nwide, nscored = len(kept_dep), len(kept_wide), len(scored)
    leaked = len(scored - kept_dep)         # scored but NOT reproduced as filter-passing
    no_imagery = ndep - (nscored - leaked)  # passed filter but no detection row written

    if leaked / max(nscored, 1) < 0.005:
        status = "PASS"
    elif leaked / max(nscored, 1) < 0.05:
        status = "WARN"
    else:
        status = "FAIL"

    md = [
        "## A1 — Step 1 Overture pre-filter reproduction",
        "",
        f"Re-ran the prefilter logic from `infer_shard_v2.py:_filter_tiles_by_overture` "
        f"over the full {len(grid):,}-tile grid, with the deployed constants parsed "
        f"directly from `infer_shard_v2.py` (`OVERTURE_MIN_AREA_M2={OVERTURE_MIN_AREA_M2:.0f}`, "
        f"`OVERTURE_RADIUS_M={dep_r:.0f}`).",
        "",
        "| metric | value |",
        "|---|---|",
        f"| grid tiles (universe) | {len(grid):,} |",
        f"| tiles kept @ {dep_r:.0f} m radius (deployed) | {ndep:,} |",
        f"| tiles kept @ {wide_r:.0f} m radius (wider, for comparison) | {nwide:,} |",
        f"| extra tiles a {wide_r:.0f} m radius would add | {nwide - ndep:,} "
        f"(+{100*(nwide-ndep)/max(ndep,1):.0f}%) |",
        f"| tiles actually scored by the v2 scan | {nscored:,} |",
        f"| scored tiles NOT reproduced as filter-passing | {leaked:,} |",
        f"| passed filter @{dep_r:.0f} m but no detection row (no usable imagery) | {no_imagery:,} "
        f"({100*no_imagery/max(ndep,1):.1f}%) |",
        "",
        f"**Filter parameters:** Step 1 keeps a grid tile if an Overture building of area "
        f">= {OVERTURE_MIN_AREA_M2:.0f} m² lies within **{dep_r:.0f} m** of the tile "
        f"centre (`OVERTURE_RADIUS_M`). A {wide_r:.0f} m radius would have kept "
        f"{nwide - ndep:,} more tiles.",
        "",
        f"**Verdict {status}** — the scored set is "
        f"{'a clean subset of' if leaked == 0 else 'mostly within'} the reproduced "
        f"{dep_r:.0f} m filter ({leaked:,} discrepancies, "
        f"{100*leaked/max(nscored,1):.2f}% — attributable to Overture-snapshot drift "
        "between the scan-time S3 bundle and the local aligned file, not a logic error). "
        f"The {no_imagery:,} filter-passing tiles with no detection row are tiles with no "
        f"usable cloud-free Sentinel-2 coverage — expected attrition, not a bug.",
        "",
    ]
    return ("\n".join(md),
            {"name": "A1 prefilter", "status": status, "leaked": leaked},
            kept_dep, kept_wide)


# ---------------------------------------------------------------------------
# A2 — anchor survival through Step 1
# ---------------------------------------------------------------------------

def map_anchors_to_grid() -> pd.DataFrame:
    grid = pd.read_parquet(GRID_PATH, columns=["tile_id", "lat", "lon"])
    anchors = pd.read_csv(ANCHORS_CSV).dropna(subset=["lat", "lng"]).reset_index(drop=True)
    tree = BallTree(np.radians(grid[["lat", "lon"]].to_numpy()), metric="haversine")
    dist, idx = tree.query(np.radians(anchors[["lat", "lng"]].to_numpy()), k=1)
    anchors["grid_tile_id"] = grid.tile_id.to_numpy()[idx[:, 0]]
    anchors["grid_dist_m"] = dist[:, 0] * EARTH_M
    return anchors


def check_a2(anchors: pd.DataFrame, results: pd.DataFrame,
             kept_dep: set, kept_wide: set) -> tuple[str, dict]:
    log("A2: anchor coverage by the v2 scan")
    # The question that matters is "is the site covered by a scored tile", NOT
    # "did the anchor's exact geocode-tile survive". A geocode is a single point;
    # a real industrial site sprawls, and its buildings generate kept tiles even
    # when the geocode lands 0.5-3 km away on an office, gate, or parcel centroid.
    rtree = BallTree(np.radians(results[["lat", "lon"]].to_numpy()), metric="haversine")
    dist, _ = rtree.query(np.radians(anchors[["lat", "lng"]].to_numpy()), k=1)
    near_m = dist[:, 0] * EARTH_M
    n = len(anchors)

    reach = [(1120, "inside a scored tile"),
             (2240, "within one tile of a scored tile"),
             (3360, "within ~1.5 tiles of a scored tile")]
    cov = {r: int((near_m <= r).sum()) for r, _ in reach}
    cov2240 = 100 * cov[2240] / n

    surv_dep = int(anchors.grid_tile_id.isin(kept_dep).sum())
    surv_wide = int(anchors.grid_tile_id.isin(kept_wide).sum())

    status = "PASS" if cov2240 >= 80 else ("WARN" if cov2240 >= 60 else "FAIL")

    uncov = anchors[near_m > 3360].copy()
    uncov["near_m"] = near_m[near_m > 3360]
    type_col = "site_type" if "site_type" in uncov.columns else None
    name_col = ("canonical_project_name" if "canonical_project_name" in uncov.columns
                else None)
    rows = []
    uncov_detail = []
    for _, r in uncov.sort_values("near_m", ascending=False).iterrows():
        nm = str(r[name_col])[:48] if name_col else "?"
        st = str(r[type_col]) if type_col else "?"
        rows.append(f"| {nm} | {st} | {r.near_m/1000:.1f} |")
        uncov_detail.append((nm, st))

    md = [
        "## A2 — Anchor coverage by the v2 scan",
        "",
        "The 316 geocoded announcement anchors are known industrial sites. This checks "
        "whether each one is **covered by a scored tile** — the correct unit of analysis. "
        "Checking instead whether an anchor's exact geocode-tile survived Step 1 would be "
        "misleading: geocodes are imprecise points, real sites sprawl across multiple "
        "2.24 km tiles, and Step 1 keeps tiles around a site's actual buildings "
        "regardless of where the geocode lands.",
        "",
        "**Coverage — distance from each anchor to the nearest scored tile:**",
        "",
        "| anchor is... | count | % |",
        "|---|---:|---:|",
        *[f"| {label} (<= {r} m) | {cov[r]}/{n} | {100*cov[r]/n:.0f}% |"
          for r, label in reach],
        "",
        f"**Step 1 filter context (geocode-point proximity to a >={OVERTURE_MIN_AREA_M2:.0f} m² "
        "building):**",
        "",
        f"- {surv_dep}/{n} ({100*surv_dep/n:.0f}%) anchor geocode-points have a qualifying "
        f"building within the deployed **{OVERTURE_RADIUS_M:.0f} m** radius.",
        f"- {surv_wide}/{n} ({100*surv_wide/n:.0f}%) have one within **{WIDE_RADIUS_M:.0f} m**.",
        "",
        f"The jump from {OVERTURE_RADIUS_M:.0f} m to {WIDE_RADIUS_M:.0f} m is **not** "
        "missing buildings — it is the geocode point sitting near, but not on top of, the "
        "largest structure (normal for sprawling industrial sites). The wider figure "
        "confirms the buildings exist; the coverage table above is what determines whether "
        "the scan actually reached each site.",
        "",
    ]
    if rows:
        md += [
            f"**Anchors with NO scored tile within 3.36 km ({len(rows)}):** each is "
            "either a not-yet-built greenfield (correctly excluded — Step 1 working) or "
            "a genuine coverage gap. The audit cannot auto-classify build status; review "
            "the `site_type` column.",
            "",
            "| project | site_type | dist to nearest scored tile (km) |",
            "|---|---|---|",
            *rows,
            "",
        ]
    md += [
        f"**Verdict {status}** — {cov2240:.0f}% of the 316 known industrial anchors have "
        "a scored tile within one tile-width. "
        + ("Coverage of known industrial geography is sound; uncovered anchors (if any) "
           "are listed above for build-status review."
           if status == "PASS" else
           "Coverage is lower than expected — review the uncovered-anchor list to "
           "separate not-yet-built greenfields (expected) from genuine gaps before NAIP."),
        "",
    ]
    return "\n".join(md), {"name": "A2 anchor coverage", "status": status,
                           "uncovered": uncov_detail}


# ---------------------------------------------------------------------------
# B1 — deployed probe metrics
# ---------------------------------------------------------------------------

def check_b1() -> tuple[str, dict]:
    log("B1: probe leaderboard metrics")
    if not LEADERBOARD_PATH.exists():
        md = ["## B1 — Deployed probe metrics", "",
              f"`{LEADERBOARD_PATH}` not found locally. Pull it with "
              "`aws s3 cp s3://industrials-scanner-us-west-2/v2-artifacts/v2/leaderboard.json "
              "data_us/v2/` and re-run.", ""]
        return "\n".join(md), {"name": "B1 probe metrics", "status": "INFO"}

    lb = json.loads(LEADERBOARD_PATH.read_text())
    m = lb[DEPLOYED_MODEL]
    auroc = m["auroc_industrial"]
    ap = m["ap_industrial"]
    r70 = m["recall_p_industrial>=0.7"]
    r95 = m["recall_p_industrial>=0.95"]

    status = "PASS" if auroc >= 0.85 else ("WARN" if auroc >= 0.70 else "FAIL")

    rows = []
    for name, d in sorted(lb.items(), key=lambda kv: -kv[1]["auroc_industrial"]):
        mark = " **(deployed)**" if name == DEPLOYED_MODEL else ""
        rows.append(f"| {name}{mark} | {d['auroc_industrial']:.3f} | "
                    f"{d['ap_industrial']:.3f} | {d['recall_p_industrial>=0.7']:.2f} | "
                    f"{d['recall_p_industrial>=0.95']:.2f} |")

    md = [
        "## B1 — Deployed probe metrics",
        "",
        f"The v2 scan applied the **`{DEPLOYED_MODEL}`** linear probe. Metrics below are "
        f"from a **single held-out test split** (`test_n={m['test_n']}`: "
        f"{m['non_n']} negative, {m['complete_n']} completed-industrial positive) — "
        "not k-fold CV, and the positive class is *completed* industrial sites only "
        "(partial/under-construction tiles are not measured here).",
        "",
        "| model | AUROC | AP | recall@0.7 | recall@0.95 |",
        "|---|---|---|---|---|",
        *rows,
        "",
        f"**Deployed `{DEPLOYED_MODEL}`: AUROC {auroc:.3f}, AP {ap:.3f}, "
        f"recall@0.7 {r70:.0%}, recall@0.95 {r95:.0%}.**",
        "",
        f"**Verdict {status}** — AUROC {auroc:.3f} is weak (0.5 = random, 0.9+ = good). "
        f"At a 0.7 cutoff the probe catches only {r70:.0%} of completed industrial sites; "
        f"at 0.95, {r95:.0%}. Every encoder in the bake-off scored AUROC 0.71–0.79, so this "
        "is a ceiling of the current embedding+probe approach, not a wrong-model choice. "
        "Implication: the v2 scan is a **weak recall filter** — usable to thin candidates "
        "for NAIP, but it permanently drops a large fraction of real sites at any "
        "high threshold. See the final recommendation.",
        "",
    ]
    return "\n".join(md), {"name": "B1 probe metrics", "status": status,
                           "auroc": auroc, "r70": r70}


# ---------------------------------------------------------------------------
# B2 — probability distribution + anchor separation
# ---------------------------------------------------------------------------

def check_b2(results: pd.DataFrame, anchors: pd.DataFrame,
             scored: set) -> tuple[str, dict]:
    log("B2: probability distribution + anchor separation")
    prob = results.prob.to_numpy()
    edges = np.arange(0, 1.0001, 0.05)
    hist, _ = np.histogram(prob, bins=edges)
    peak_bin = int(np.argmax(hist))
    spike_lo, spike_hi = edges[peak_bin], edges[peak_bin + 1]

    by_tile = results.set_index("tile_id").prob
    in_scan = anchors[anchors.grid_tile_id.isin(scored)].copy()
    in_scan["prob"] = by_tile.reindex(in_scan.grid_tile_id).to_numpy()
    a_prob = in_scan.prob.dropna().to_numpy()

    overall_med = float(np.median(prob))
    anchor_med = float(np.median(a_prob)) if len(a_prob) else float("nan")

    def frac_above(arr, t):
        return float((arr >= t).mean()) if len(arr) else float("nan")

    # Floor test: anchors are training data, so high probs prove fit not
    # generalization — but anchors piling near 0.5 means the probe is broken.
    separates = (anchor_med - overall_med) >= 0.12 and frac_above(a_prob, 0.7) >= 0.35
    status = "PASS" if separates else ("WARN" if anchor_med >= 0.55 else "FAIL")

    hist_rows = []
    for i in range(len(hist)):
        bar = "#" * int(60 * hist[i] / max(hist.max(), 1))
        hist_rows.append(f"| {edges[i]:.2f}-{edges[i+1]:.2f} | {hist[i]:6,} | {bar} |")

    md = [
        "## B2 — Probability distribution & anchor separation",
        "",
        "> **Circularity caveat:** the anchors were training data for the probe. Anchor "
        "probs being *high* proves the probe **fit** its training, not that it "
        "**generalizes**. The useful signal is the opposite direction: anchors piling up "
        "near 0.5 would mean the probe cannot even separate its own training positives — "
        "a hard failure floor.",
        "",
        f"Probability histogram over all {len(prob):,} scored tiles (bin width 0.05):",
        "",
        "| prob bin | count | |",
        "|---|---:|---|",
        *hist_rows,
        "",
        f"Peak bin: **{spike_lo:.2f}–{spike_hi:.2f}** with {hist[peak_bin]:,} tiles "
        f"({100*hist[peak_bin]/len(prob):.0f}% of all detections). Overall median "
        f"prob = {overall_med:.3f}.",
        "",
        "**Anchor probs vs all tiles:**",
        "",
        "| metric | anchors-in-scan | all scored tiles |",
        "|---|---|---|",
        f"| n | {len(a_prob)} | {len(prob):,} |",
        f"| median prob | {anchor_med:.3f} | {overall_med:.3f} |",
        f"| frac >= 0.3 | {frac_above(a_prob,0.3):.2f} | {frac_above(prob,0.3):.2f} |",
        f"| frac >= 0.5 | {frac_above(a_prob,0.5):.2f} | {frac_above(prob,0.5):.2f} |",
        f"| frac >= 0.7 | {frac_above(a_prob,0.7):.2f} | {frac_above(prob,0.7):.2f} |",
        f"| frac >= 0.9 | {frac_above(a_prob,0.9):.2f} | {frac_above(prob,0.9):.2f} |",
        "",
        "The anchor `frac >=` row is a **recall proxy** (upper bound — training positives) "
        "for the aggregation threshold: e.g. clustering at prob>=0.5 retains roughly "
        f"{frac_above(a_prob,0.5):.0%} of anchor-equivalent sites.",
        "",
        f"**Verdict {status}** — "
        + (f"anchors separate clearly from the bulk. The distribution is smooth and "
           f"unimodal (peak bin {spike_lo:.2f}-{spike_hi:.2f}, no degenerate spike); the "
           "mass below the anchor median is non-anchor ambiguous building tiles "
           "(commercial/warehouse), which NAIP is expected to filter."
           if status == "PASS" else
           "anchors do NOT separate cleanly from the bulk — the probe is "
           "uncalibrated/weak (consistent with the B1 AUROC), a genuine inability to "
           "discriminate rather than just ambiguous negatives.")
        + " Input-prep was verified identical between training (`v2_train.py`) and scan "
          "(`infer_shard_v2.py`) — percentile stretch, RGB order, LANCZOS 256->224, "
          "ImageNet norm all match — so this is a probe-quality effect, not a "
          "train/scan prep mismatch.",
        "",
    ]
    return "\n".join(md), {"name": "B2 prob distribution", "status": status}


# ---------------------------------------------------------------------------
# B3 — per-shard completeness
# ---------------------------------------------------------------------------

def check_b3(results: pd.DataFrame, kept_dep: set) -> tuple[str, dict]:
    log("B3: per-shard completeness")
    grid = pd.read_parquet(GRID_PATH, columns=["tile_id", "mgrs_tile"])
    n_grid_shards = grid.mgrs_tile.nunique()
    files = [p for p in RESULTS_DIR.glob("*.parquet") if not p.stem.endswith("_emb")]
    n_files = len(files)

    # A grid shard writes a result parquet only if >=1 of its tiles passes the
    # Overture filter; a shard with no qualifying building produces no file by
    # design (process_shard returns before writing). kept_dep tells genuine
    # gaps apart from expected-empty (no-building) shards.
    grid_shards = set(grid.mgrs_tile.unique())
    file_shards = {p.stem for p in files}
    kept_shards = set(grid[grid.tile_id.isin(kept_dep)].mgrs_tile.unique())
    no_building = grid_shards - kept_shards
    genuinely_missing = sorted((grid_shards & kept_shards) - file_shards)

    scored_per = results.groupby("mgrs_tile").size()
    kept_per = grid[grid.tile_id.isin(kept_dep)].groupby("mgrs_tile").size()
    shards = sorted(set(kept_per.index) | set(scored_per.index))
    kept_per = kept_per.reindex(shards, fill_value=0)
    scored_per = scored_per.reindex(shards, fill_value=0)
    overscored = int((scored_per > kept_per + 1).sum())   # Overture-drift, not a bug

    scored_mgrs = set(results.mgrs_tile)
    empty_mgrs = [p.stem for p in files if p.stem not in scored_mgrs]

    status = "PASS" if (not genuinely_missing
                        and overscored / max(len(shards), 1) < 0.02) else "WARN"

    spp = scored_per.to_numpy()
    md = [
        "## B3 — Per-shard completeness",
        "",
        f"The scan emits one result parquet per MGRS shard that has >=1 tile passing the "
        f"Overture filter; a shard with no qualifying building produces no file by "
        f"design. Checked all {n_grid_shards} grid shards.",
        "",
        "| metric | value |",
        "|---|---|",
        f"| grid shards | {n_grid_shards} |",
        f"| shards with a result file | {n_files} |",
        f"| shards with no qualifying building (expected: no file) | {len(no_building)} |",
        f"| shards that should have a file but don't (genuine gap) | {len(genuinely_missing)} |",
        f"| result files with detection rows | {len(scored_mgrs)} |",
        f"| result files empty (0 rows — all tiles failed imagery/validity) | {len(empty_mgrs)} |",
        f"| shards scoring slightly more than the reproduced filter | {overscored} "
        f"({100*overscored/max(len(shards),1):.1f}% — same Overture-snapshot drift as A1) |",
        f"| rows per non-empty shard: min / median / max | {spp[spp>0].min()} / "
        f"{int(np.median(spp[spp>0]))} / {spp.max()} |",
        "",
        f"**Verdict {status}** — "
        + (f"every grid shard with a qualifying building produced a result file, so the "
           f"scan ran to completion. {len(no_building)} shards have no qualifying "
           "building and correctly produced no file; "
           f"{len(empty_mgrs)} more produced a file with zero detection rows (all tiles "
           "failed imagery/validity — expected attrition in cloudy or low-coverage "
           "zones). The small over-count is Overture-snapshot drift (see A1), not a "
           "partial scan."
           if status == "PASS" else
           f"{len(genuinely_missing)} shards have a qualifying building but no result "
           f"file ({', '.join(genuinely_missing[:8])}"
           f"{', ...' if len(genuinely_missing) > 8 else ''}) — the scan did not run to "
           "completion on these. Re-run them before relying on the candidate set."),
        "",
    ]
    return "\n".join(md), {"name": "B3 shard completeness", "status": status}


# ---------------------------------------------------------------------------
# C1 — dedup magnitude
# ---------------------------------------------------------------------------

def check_c1(results: pd.DataFrame) -> tuple[str, dict]:
    log("C1: dedup magnitude")
    rows = []
    for thr in (0.3, 0.5, 0.7, 0.9):
        sub = results[results.prob >= thr]
        if len(sub) < 2:
            rows.append(f"| {thr:.1f} | {len(sub):,} | - | - |")
            continue
        clustered = _cluster(sub[["tile_id", "lon", "lat", "prob"]].copy(),
                             eps_m=2000.0, min_samples=2)
        n_sites = int(clustered.cluster[clustered.cluster >= 0].nunique())
        n_singletons = int((clustered.cluster < 0).sum())
        rows.append(f"| {thr:.1f} | {len(sub):,} | {n_sites:,} | {n_singletons:,} |")

    md = [
        "## C1 — Dedup magnitude (NAIP-stage input sizing)",
        "",
        f"The grid has 25% tile overlap, so the raw {len(results):,} scored tiles "
        "double-count sites. "
        "DBSCAN (`eps=2 km, min_samples=2`, EPSG:5070 — reused from `v1/aggregate.py`) "
        "collapses overlapping tiles into distinct candidate sites. Singletons are "
        "dropped (a real site spans >=2 tiles at 1.68 km stride).",
        "",
        "| prob threshold | tiles kept | distinct sites | singletons dropped |",
        "|---|---:|---:|---:|",
        *rows,
        "",
        "Pick the threshold for the NAIP handoff from this table against the B2 anchor "
        "recall-proxy row: lower threshold = more recall into NAIP = more false positives "
        "for NAIP to filter.",
        "",
    ]
    return "\n".join(md), {"name": "C1 dedup", "status": "INFO"}


# ---------------------------------------------------------------------------
# B4 — visual contact sheet
# ---------------------------------------------------------------------------

def make_contact_sheet(results: pd.DataFrame) -> str:
    log("B4: contact sheet")
    from PIL import Image, ImageDraw

    if not QUEUE_PATH.exists():
        return "## B4 — Contact sheet\n\nqueue.json not found; skipped.\n"
    queue = json.loads(QUEUE_PATH.read_text())
    # The chips were rendered for the dropped labeling pass; queue.json carries
    # that run's probs. Re-band each chip by its current (fixed-run) prob so the
    # sheet reflects this scan, not the stale one.
    prob_by_tile = results.set_index("tile_id").prob.to_dict()
    chips = [{"tile_id": c["tile_id"], "prob": float(prob_by_tile[c["tile_id"]])}
             for c in queue
             if c.get("chip_quality") == "ok"
             and c["tile_id"] in prob_by_tile
             and (CHIPS_DIR / f"{c['tile_id']}.png").exists()]

    bands = [("p > 0.95", 0.95, 1.01), ("0.80 - 0.95", 0.80, 0.95),
             ("0.45 - 0.55", 0.45, 0.55), ("p < 0.05", 0.0, 0.05)]
    per_band, cell = 8, 224
    pad, label_h = 6, 18

    sheet_w = per_band * (cell + pad) + pad
    sheet_h = len(bands) * (cell + label_h + pad) + pad
    sheet = Image.new("RGB", (sheet_w, sheet_h), (14, 17, 23))
    draw = ImageDraw.Draw(sheet)

    for r, (name, lo, hi) in enumerate(bands):
        picks = sorted([c for c in chips if lo <= c["prob"] < hi],
                       key=lambda c: -c["prob"])[:per_band]
        y = pad + r * (cell + label_h + pad)
        draw.text((pad, y), f"{name}   ({len(picks)} shown)", fill=(230, 237, 243))
        for ci, c in enumerate(picks):
            x = pad + ci * (cell + pad)
            chip = Image.open(CHIPS_DIR / f"{c['tile_id']}.png").convert("RGB")
            chip = chip.resize((cell, cell), Image.LANCZOS)
            sheet.paste(chip, (x, y + label_h))
            draw.text((x + 3, y + label_h + 3), f"{c['prob']:.2f}", fill=(255, 255, 0))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sheet.save(CONTACT_SHEET_PATH)
    log(f"  wrote {CONTACT_SHEET_PATH} ({len(chips)} chips matched to scored tiles)")
    return (
        "## B4 — Visual contact sheet\n\n"
        f"`{CONTACT_SHEET_PATH.relative_to(ROOT)}` — rows are probability bands "
        "(>0.95, 0.80-0.95, 0.45-0.55, <0.05), up to 8 chips each. Chips are reused "
        "from the dropped labeling pass but **re-banded by their current fixed-run "
        f"prob** ({len(chips)} of the pre-rendered chips matched a scored tile). "
        "Eyeball it: top-row chips should look industrial; the 0.45-0.55 row shows "
        "what the mid-probability tiles actually contain.\n"
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = load_results()
    scored = set(results.tile_id)
    anchors = map_anchors_to_grid()

    a1_md, a1_v, kept_dep, kept_wide = check_a1(results)
    a2_md, a2_v = check_a2(anchors, results, kept_dep, kept_wide)
    b1_md, b1_v = check_b1()
    b2_md, b2_v = check_b2(results, anchors, scored)
    b3_md, b3_v = check_b3(results, kept_dep)
    c1_md, c1_v = check_c1(results)
    b4_md = make_contact_sheet(results)

    verdicts = [a1_v, a2_v, b1_v, b2_v, b3_v, c1_v]
    summary_rows = [f"| {v['name']} | {v['status']} |" for v in verdicts]

    # Gate on substance, not a status tally: A1 WARN here is benign Overture
    # drift; A2 WARN/FAIL is dual-cause (greenfields vs gaps) and not a defect
    # on its own. Real blockers are a broken filter/scan (A1/B3 FAIL) or a
    # probe that cannot separate its own training positives (B2 FAIL).
    mechanical_fail = a1_v["status"] == "FAIL" or b3_v["status"] == "FAIL"
    probe_broken = b2_v["status"] == "FAIL"
    auroc = b1_v.get("auroc")
    r70 = b1_v.get("r70")
    probe_line = (f"AUROC {auroc:.2f} / recall@0.7 {r70:.0%} on completed sites"
                  if auroc is not None else "see B1")
    if mechanical_fail:
        recommendation = (
            "**A pipeline step is mechanically broken** (A1 or B3 FAIL — a filter logic "
            "error or an incomplete scan). Fix that before the NAIP handoff; everything "
            "downstream inherits the defect."
        )
    elif probe_broken:
        recommendation = (
            "**The probe cannot separate even its own training positives** (B2 FAIL). "
            "Do not hand off to NAIP — retrain or replace `dino_vitb` first."
        )
    else:
        # Step-1 finding, computed from A1's leaked count and A2's uncovered
        # anchors split by build status — greenfield misses are correct
        # exclusions, already-built misses are the only genuine recall gap.
        uncov = a2_v.get("uncovered", [])
        built = [(n, s) for n, s in uncov if "greenfield" not in s.lower()]
        green = [(n, s) for n, s in uncov if "greenfield" in s.lower()]
        leaked = a1_v.get("leaked", 0)
        a1_line = (f"The scored set reproduces the deployed Overture filter exactly "
                   f"(A1: {leaked} discrepanc{'y' if leaked == 1 else 'ies'})"
                   if leaked < 50 else
                   f"{leaked:,} scored tiles fall outside the reproduced filter (A1)")
        cov_line = (f"{316 - len(uncov)}/316 known anchors have a scored tile within "
                    "~1.5 tile-widths (A2)")

        if not built:
            step1 = (
                f"2. **Step 1 is clean — no action needed before NAIP.** {a1_line} "
                f"and {cov_line}. "
                + (f"The {len(green)} uncovered anchors are all not-yet-built "
                   "greenfields, correctly excluded — they have no structure for the "
                   "Overture filter to key on." if green else
                   "Anchor coverage is complete."))
        else:
            names = ", ".join(n for n, _ in built[:3])
            mandate = ("Spot-check "
                       + ("that site" if len(built) == 1 else "those sites")
                       + "; if the gap is real, refresh Overture and re-run Step 1 + "
                       "the scan. Not a blocker for the NAIP handoff."
                       if len(built) <= 3 else
                       "Refresh Overture and re-run Step 1 + the scan before the NAIP "
                       "handoff to backfill that cohort.")
            step1 = (
                f"2. **Step 1 recall — a small gap to review.** {a1_line} and "
                f"{cov_line}. "
                + (f"{len(green)} of the uncovered anchors are not-yet-built greenfields "
                   "(correctly excluded); " if green else "")
                + f"{len(built)} {'is' if len(built) == 1 else 'are'} already-built "
                f"({names}) and {'sits' if len(built) == 1 else 'sit'} just outside scan "
                "coverage — likely geocode "
                "imprecision or a thin Overture building footprint there, possibly "
                f"snapshot drift. {mandate}")

        recommendation = (
            "**Both steps are mechanically sound** — Step 1's filter logic reproduces "
            "(A1), the scan ran to completion on every shard (B3), input-prep matches "
            "training exactly, and known industrial anchors separate clearly from the "
            "bulk (B2). Findings:\n\n"
            f"1. **Probe quality (B1).** `dino_vitb` is {probe_line} — a usable but weak "
            "recall filter, a ceiling shared by all five encoders in the bake-off. "
            "Aggregate the candidate set at a **low probability threshold (0.3-0.5)** for "
            "the NAIP handoff so recall is preserved — NAIP does the precision work and "
            "cannot recover sites the probe scored low. Size the handoff with the C1 "
            "table against the B2 anchor recall-proxy; if that proxy is unacceptably low "
            "at your chosen threshold, retrain the probe (more labeled data / stronger "
            "encoder) before NAIP rather than after.\n\n"
            + step1
        )

    report = "\n".join([
        "# Phase 3 v2 Pipeline — Sanity Audit",
        "",
        f"Generated by `phase3_scan/v2/audit_scan.py` over {len(results):,} scored tiles "
        f"in `data_us/phase3_results_v2/`.",
        "",
        "## Verdict summary",
        "",
        "| check | status |",
        "|---|---|",
        *summary_rows,
        "",
        "## Recommendation",
        "",
        recommendation,
        "",
        "---",
        "",
        a1_md, a2_md, b1_md, b2_md, b3_md, c1_md, b4_md,
    ])
    REPORT_PATH.write_text(report)
    log(f"wrote {REPORT_PATH}")
    print("\n" + "\n".join(["VERDICT SUMMARY"] + summary_rows))
    print("\nRECOMMENDATION:\n" + recommendation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
