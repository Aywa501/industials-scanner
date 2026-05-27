# Industrial Site Detection — Imagery Pipeline

## What this is

A remote-sensing pipeline that detects industrial-scale sites across the US, seeded from a labeled dataset of 316 manufacturing investment announcements (2015–2025). The 316 set is the *training seed* — the goal is a continuously-updated canonical table of every detected industrial site in the US, tagged by NAICS code, with multi-source timelines per site.

The pipeline is **staged single-image classification, not change detection.** Stage 1 is a cheap binary classifier (`industrial` vs `not`) that scans CONUS at S2 resolution as a recall-first sieve. Stage 2 (a per-candidate lifecycle classifier) reconstructs the construction timeline by running on each historical year of a surviving candidate. NAIP imagery refines confirmed industrial candidates into building footprints, and worker agents enrich each verified site with operator / project / NAICS information.

Negatives for Stage 1 come from heuristic-filtered random CONUS, with embedding-distance pruning to remove industrial leaks (random-CONUS samples that happened to land on real industry). The hard negatives Stage 1 lets through become free training data for Stage 2 and for any intermediate filter we slot between them.

Runs on AWS (us-west-2): EC2 spot for embed + scan + Stage 2b, S3 (`sentinel-cogs`, `usgs-landsat`, NAIP) for all imagery reads. Earth Engine was used for Phase 1 anchor prep only; everything downstream is AWS COG + rasterio. BigQuery is the canonical site table.

## Current state (2026-05-25)

- **v3 detector trained.** 5-encoder bake-off (DINOv3 SAT-493M, DINOv2, ResNet-50, Prithvi 300M/600M). `dino_sat493m` is the production probe. Artifacts at `data_us/phase2/v3/`.
- **CONUS scan complete.** 696,417 Overture-industrial buildings scored. Manifest 697,589 rows. Output at `s3://industrials-scanner-us-west-2/v3-artifacts/v3/scan_chunks/_scores/`.
- **Threshold locked at `p_dino_sat493m ≥ 0.70`** → ~156K candidates (91% recall on a 1000-row Overture-industrial validation sample).
- **Stage 2b temporal filter (current work).** Landsat 2008 vs 2022 panchromatic; drops candidates that look identical AND have no changed neighbor within 2 km. In calibration on a 60-row sample; about to scale.
- **Phase 3 NAIP refinement** scaffolded at `sites_us/phase3_naip/`, currently on hold pending Stage 2b completion.

## Pipeline phases

### Phase 0 — Announcement crawling (complete)

Produced `data_us/labels/manufacturing_announcements_geocoded.csv`: 316 geocoded manufacturing investments (2015–2025). Phase 0 is closed; the dataset doesn't grow from new announcement research.

### Phase 1 — Anchor imagery + feature prep (complete)

- **NAIP archive** of the 316 anchor sites: 2242 GeoTIFFs at `gs://{bucket}/tiles/{site_id}/{year}_{image_id}.tif`, manifest at `gs://{bucket}/manifest/tile_manifest.parquet`. Used as a labeled validation set for the eventual NAIP-stage classifiers.
- **Sentinel-2 anchor chips**: yearly summer (Jun–Aug) median composites, B4/B3/B2/B8 at 10 m, 256×256 px (1.28 km buffer), for 316 anchors + ~500 random CONUS sites × 9 years (2017–2025). Manifest at `gs://{bucket}/manifest/s2_chip_manifest.parquet`. Pulled via `phase1_prep/pull_s2.py` against GEE's high-volume endpoint (`ee.data.computePixels`).
- **Anchor feature analysis** (`phase1_prep/anchor_features.py`): NLCD land-cover class, slope, elevation, distance to road, distance to nearest developed pixel — sampled at each anchor. Empirical findings inform the Phase 2 heuristic.

### Phase 2 — Manual labeling + v3 industrial detector (complete)

**Step 2a — Exclusion heuristic.** Define features that drop ~zero anchors but eliminate large fractions of CONUS as definitely-not-industrial. Apply during candidate-region sampling in Phase 3 step 1 to bound inference cost.

Confirmed safe filters from the 316 anchors:
- **Slope > 15°** — drops 0/316 anchors, eliminates ~15–20% of CONUS (mountain terrain). Strongest single filter.
- **NLCD = ice/snow** — drops 0/316. Trivial CONUS area but free.
- **NLCD = open water / wetland** with surrounding-area majority vote — drops 0/316 with adequate buffer; ~8 anchors hit these classes at the bare lat/lng but are geocoding artifacts. Eliminates large CONUS fractions.

**Step 2b — Manual labeling round (complete).** Built a keyboard-driven webapp (`phase2_classifier/labeling_webapp/`) and produced ground-truth labels on a stratified ~700-chip sample.

Output (under `data_us/`):
- `manual_labels.parquet` — 682 labels across 77 sites (52 anchor + 25 random-CONUS negatives, soon 50). Distribution: 326 not_a_site / 161 partial / 99 complete / 96 unsure.
- `manual_site_outlines.parquet` — per-site polygons (normalized [0,1] coords, same outline applies to all 9 years per site since chips are co-registered).
- `manual_site_notes.parquet` — free-text notes; entries prefixed `IMPORTANT:` flag CSV `site_type` mislabels (5 found, all on greenfields that look like extension/brownfield/demolition in S2).
- `manual_site_flags.parquet` — `bad_geocode` flag for sites where the chip is off-center.

Empirical findings:
- ~14% of greenfields in the CSV are mis-classified per S2 inspection. Don't treat CSV `site_type` as ground truth — apply IMPORTANT-note overrides.
- ~14% of labels are `unsure`. A meaningful fraction are temporal mismatches: announcements made for already-built sites (no S2 transition observable). `ann_year` is unreliable as a completion marker for these.
- 40 of 316 anchors had city-level geocodes (lat/lng with <4 decimal places); excluded from labeling and downstream training.

**Step 2c — v3 industrial detector (complete).** Cheap recall-first binary classifier; ran across Overture-industrial-CONUS × multiple years.

- **Backbone shipped:** DINOv3 ViT-L/16 SAT-493M (`facebook/dinov3-vitl16-pretrain-sat493m`), frozen, 1024-dim CLS, linear probe head.
- **Encoder bake-off:** also trained probes on `dino_vitb`, `resnet50`, `prithvi_300m`, `prithvi_600m`. SAT-493M wins. All artifacts at `data_us/phase2/v3/`.
- **Training data:** Overture-industrial buildings as positives (much larger than v2's 682 manual labels), OSM-derived non-industrial as negatives. IMPORTANT-flagged sites override CSV `site_type`.
- **Input:** B4/B3/B2 only, 1–99 percentile stretch, model-specific normalization, resize 256→224.
- **Operating point:** `p_dino_sat493m ≥ 0.70` → ~156K candidates from 696K scored buildings (91% recall on 1000-row Overture-industrial validation).
- **Code:** `phase2_classifier/v3/v3_train.py` (EC2-driven), `phase2_classifier/v3/v3_scan_infer.py` (CONUS scan), bootstrap + launch scripts alongside.

**Stage 2b — temporal stability filter (current).** Slotted between v3 and Phase 3 NAIP. Drops candidates that look identical in 2008 and 2022 AND have no changed neighbor within 2 km. Reasoning: pre-existing static sites have no news event for the verification agent to find.

- **Imagery:** Landsat C2 L1 panchromatic (15 m). L7 ETM+ for 2008 (post-SLC-off, median over Jun–Aug to fill gaps); L8 OLI for 2022. STAC at `landsatlook.usgs.gov`, COGs at `s3://usgs-landsat` (requester-pays).
- **Method:** 2022-derived footprint mask via Sobel gradient + connected components; within-year ratio `mean(pan[mask]) / mean(pan[~mask])` cancels L7/L8 sensor offsets; `change = |ratio_2022 − ratio_2008|`.
- **Calibration in progress.** 60-row sample: median change=0.119, T=0.10 drops 42%. Open items: T against 316 known-changed announcements + pre-2008 negatives; `min_footprint_pixels` threshold for `ambiguous` flag; 2 km proximity rescue logic.
- **Code:** `phase2_classifier/v3/landsat_change_sanity.py` (sanity script), `inspect_one.py` (per-candidate viz).

**Lifecycle classifier — deferred / superseded.** The original 3-class { complete, partial, not_a_site } classifier was deprioritized: Stage 2b's coarser "was this static for 14 years?" filter is cheaper and matches the downstream need (agent verification needs *something to find*, not a construction date). A lifecycle classifier may return for fine-grained dating after Phase 3 NAIP.

**Note on prior failed v0 detector.** Earlier work trained a siamese ResNet-18 on chip-pair labels for binary change detection. Three structural revisions plateaued at val AP ≈ 0.21 — data-bound. Replaced by the staged-classifier formulation. Don't propose returning to pair-based change detection without new data.

### Phase 3 — Per-site refinement + verification

Linear pipeline. No iterative loop.

1. **CONUS scan (complete).** v3 detector ran over Overture-industrial-CONUS × multiple years. Output: `scan_results.parquet` (696K rows). Reads Sentinel-2 from `sentinel-cogs` directly; no GEE.
2. **v3 threshold filter (complete).** `p_dino_sat493m ≥ 0.70` → ~156K candidates.
3. **Stage 2b temporal stability filter (current).** Landsat 2008 vs 2022 panchromatic change detection per candidate; drop static sites unless rescued by a changed neighbor within 2 km. Output: `data_us/phase2/v3/stage3_candidates.parquet`.
4. **NAIP epoch selection + refinement (scaffolded, on hold).** SAM 3 text-prompted segmentation + LLM reasoning groups buildings into discrete sites. Pipeline at `sites_us/phase3_naip/`. See `phase3_naip/NAIP_STAGE_NOTES.md`.
5. **Agent verification (not yet built).** Per-candidate worker agents search news, identify operator/project/NAICS, build multi-source timeline. Prototype dirs `prototypes/orchestrator/` + `prototypes/worker_agent/`.
6. **Aggregate** — verified records into the canonical BigQuery site table.

### Phase 4 — Steady state (later)

Periodic rescans on new S2 imagery; ongoing agent verification queue; classifier retraining as the verified site set grows. Iteration here is *additive labels*, not *bootstrap from noisy outputs*.

## Datasets

| Source | Resolution | Role |
| --- | --- | --- |
| **AWS `sentinel-cogs`** (S2 L2A) | 10 m, B4/B3/B2/B8 | v3 training + CONUS scan |
| **AWS `usgs-landsat`** (C2 L1) | 15 m pan, requester-pays | Stage 2b 2008-vs-2022 stability filter |
| **AWS NAIP** | 0.6–1 m, RGBN | Phase 3 NAIP refinement |
| **`COPERNICUS/S2_SR_HARMONIZED`** (GEE) | 10 m | Phase 1 anchor chips only (frozen) |
| **`USGS/NLCD_RELEASES/2019_REL/NLCD/2019`** (GEE) | 30 m | Phase 2 heuristic land-cover (frozen) |
| **`USGS/3DEP/10m_collection`** (GEE) | 10 m | Phase 2 heuristic slope/elevation (frozen) |

Phase 1 anchor chips were pulled once via GEE; everything downstream reads AWS COGs directly via rasterio (no quota). NAIP fires only on Stage-2b-surviving candidates.

## Output

One canonical BigQuery site table — one row per detected industrial site:

```
lat, lng, naics_code, completion_year, confidence,
timeline (multi-source chronology, not collapsed),
+ standard metadata
```

Multi-source timelines are preserved as structured records; the worker agent doesn't reduce N news stories to one description.

## Scope

US only. Global is a future concern; the architecture leans US-only because NAIP is US-only and the heuristic features (NLCD, 3DEP) are CONUS coverage. Going global means swapping NAIP for Planet/Maxar and adapting the heuristics to global land-cover data.

## Important context

- **GEE for prep only, AWS for everything else.** Phase 1 anchor chips were pulled via GEE; v3 training, CONUS scan, and Stage 2b read AWS COGs directly via rasterio. Cloud-masking and seasonal compositing happen locally.
- **Staged single-image classification, not pair-based change detection.** v3 detector (industrial-vs-not) is the cheap CONUS-wide sieve. Stage 2b (Landsat 2008-vs-2022 stability) drops pre-existing static sites. Stage 3 (NAIP refinement + agent verification) closes the loop. Each stage is tuned for recall; precision comes from agent verification.
- **CSV `site_type` is noisy.** ~14% of greenfields mislabeled per manual S2 inspection. `IMPORTANT:` notes in `manual_site_notes.parquet` carry the user's overrides — apply during dataset construction.
- **The 316 are training + validation, not the production cohort.** The pipeline finds sites that *aren't* in the 316.
- **Recall over precision at every imagery stage.** False negatives are permanent; FPs get filtered by the next stage or by agents.

## Tech stack

- **AWS S3 + EC2 (us-west-2):** `sentinel-cogs` + `usgs-landsat` + NAIP for imagery; g4dn spot GPUs for embed/scan; S3 for artifact staging.
- **DINOv3 ViT-L/16 SAT-493M** (frozen) — v3 detector backbone. Linear probe over Overture-industrial positives + OSM negatives.
- **Landsat C2 L1 pan** — Stage 2b temporal stability filter (L7 ETM+ for 2008, L8 OLI for 2022).
- **SAM 3** (planned, on-hold) — Phase 3 NAIP building-level segmentation.
- **Google Earth Engine** — Phase 1 anchor imagery only (frozen).
- **BigQuery** — canonical site table + per-detection candidate records.
- Python: `rasterio`, `pystac-client`, `shapely`, `pyproj`, `torch`/`transformers`, `scikit-learn`, `scipy.ndimage`.

## Running

Copy `.env.example` to `.env` and `.env.agent-profile` for IAM-user AWS creds (local). Then:

```bash
pip install -r requirements.txt

# Phase 1 — frozen; included for reproducibility
python phase1_prep/pull_s2.py --workers 50           # GEE anchor chips
python phase1_prep/anchor_features.py                # NLCD / slope features

# Phase 2c — v3 detector (trained; rebuild artifacts only if needed)
python phase2_classifier/v3/v3_build_dataset.py
python phase2_classifier/v3/v3_build_scenes_index.py
./phase2_classifier/v3/push_v3_bundle.sh             # → s3://bucket/v3-bundle/
./phase2_classifier/v3/launch_v3.sh                  # EC2 spot launch (g4dn.8xlarge/g6.8xlarge)

# Phase 3 scan — CONUS inference (complete; rebuild if model changes)
python phase2_classifier/v3/v3_build_scan_manifest.py
./phase2_classifier/v3/push_v3_scan_bundle.sh
./phase2_classifier/v3/launch_v3_scan.sh             # g4dn.4xlarge primary, g4dn.8xlarge fallback

# Stage 2b — Landsat temporal filter (in calibration)
python phase2_classifier/v3/landsat_change_sanity.py # 60-row validation
python phase2_classifier/v3/inspect_one.py <lat> <lon>  # per-candidate viz

# Phase 3 NAIP (scaffolded, on hold)
# See sites_us/phase3_naip/NAIP_STAGE_NOTES.md
```

Scan outputs live in S3 (`s3://industrials-scanner-us-west-2/v3-artifacts/`) and are the source of truth — local copies under `data_us/phase2/v3/` are working caches.
