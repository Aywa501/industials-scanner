# Industrial Site Detection — Imagery Pipeline

## What this is

A remote-sensing pipeline that detects industrial-scale sites across the US, seeded from a labeled dataset of 316 manufacturing investment announcements (2015–2025). The 316 set is the *training seed* — the goal is a continuously-updated canonical table of every detected industrial site in the US, tagged by NAICS code, with multi-source timelines per site.

The pipeline is **single-image classification, not change detection**. A lifecycle classifier (3-class: complete / partial / not_a_site) runs on each Sentinel-2 chip; aggregating its outputs across years for a candidate location reconstructs that site's construction timeline. NAIP imagery refines confirmed industrial candidates into building footprints, and worker agents enrich each verified site with operator / project / NAICS information.

Negatives are mass-produced from CONUS by applying simple exclusionary heuristics derived from the 316 anchors (e.g., slope, land-cover impossibilities). No iterative bootstrap loop, no separate negative-class research effort.

Runs on Google Cloud — Earth Engine for imagery prep, Cloud Storage for staging, AWS Open Data Sentinel-2 COGs for national inference, Vertex AI / Cloud Run with GPU for SAM, BigQuery for the canonical site table.

## Pipeline phases

### Phase 0 — Announcement crawling (complete)

Produced `data_us/manufacturing_announcements_geocoded.csv`: 316 geocoded manufacturing investments (2015–2025). Phase 0 is closed; the dataset doesn't grow from new announcement research.

### Phase 1 — Anchor imagery + feature prep (complete)

- **NAIP archive** of the 316 anchor sites: 2242 GeoTIFFs at `gs://{bucket}/tiles/{site_id}/{year}_{image_id}.tif`, manifest at `gs://{bucket}/manifest/tile_manifest.parquet`. Used as a labeled validation set for the eventual NAIP-stage classifiers.
- **Sentinel-2 anchor chips**: yearly summer (Jun–Aug) median composites, B4/B3/B2/B8 at 10 m, 256×256 px (1.28 km buffer), for 316 anchors + ~500 random CONUS sites × 9 years (2017–2025). Manifest at `gs://{bucket}/manifest/s2_chip_manifest.parquet`. Pulled via `phase1_prep/pull_s2.py` against GEE's high-volume endpoint (`ee.data.computePixels`).
- **Anchor feature analysis** (`phase1_prep/anchor_features.py`): NLCD land-cover class, slope, elevation, distance to road, distance to nearest developed pixel — sampled at each anchor. Empirical findings inform the Phase 2 heuristic.

### Phase 2 — Exclusion heuristic + lifecycle classifier (current)

**Step 2a — Exclusion heuristic.** Define features that drop ~zero anchors but eliminate large fractions of CONUS as definitely-not-industrial. Apply during candidate-region sampling (Phase 3 step 1) to bound inference cost, and during negative-class sampling (Phase 2 step 2c) to harvest clean negatives instead of noisy random CONUS.

Confirmed safe filters from the 316 anchors:
- **Slope > 15°** — drops 0/316 anchors, eliminates ~15–20% of CONUS (mountain terrain). Strongest single filter.
- **NLCD = ice/snow** — drops 0/316. Trivial CONUS area but free.
- **NLCD = open water / wetland** with surrounding-area majority vote — drops 0/316 with adequate buffer; ~8 anchors hit these classes at the bare lat/lng but are geocoding artifacts. Eliminates large CONUS fractions.

Distance-to-road, population-density, and other features are candidates for further empirical testing.

**Step 2b — Lifecycle classifier (3-class).** Single-image classifier. Input: one S2 chip. Output: { `complete`, `partial`, `not_a_site` }.

Label derivation per `(site, year)`:
- Anchor, `year < ann_year` → `not_a_site` (visually nothing yet built)
- Anchor, `ann_year ≤ year < ann_year + 2` → `partial`
- Anchor, `year ≥ ann_year + 2` → `complete`
- Heuristic-filtered random CONUS, any year → `not_a_site`

Architectural baseline: ImageNet-pretrained image backbone, single-image head. **Strongly preferred:** remote-sensing-pretrained foundation model (Prithvi-100M, Clay, SatMAE) or VLM with semantic prompts (RemoteCLIP, SkyCLIP). Open-vocabulary text prompts also future-enable sub-class queries ("data center" vs "battery factory") without retraining.

Why this formulation works where the previous siamese pair-based change classifier didn't:
- Single-image classification is a much smaller-data problem than learned change detection.
- Every (site, year) becomes a labeled example — ~2800 anchor-years instead of 316 sites.
- Heuristic-filtered CONUS produces effectively unlimited clean negatives.
- Timeline reconstruction (Phase 3) emerges from year-by-year inference; no separate change-detection model needed.

**Note on prior failed v0 detector.** Earlier work in this phase trained a siamese ResNet-18 on chip-pair labels for binary change detection. Three structural revisions (label tightening, summer compositing, diff-feature head) all plateaued at val AP ≈ 0.21 — data-bound. Replaced by the lifecycle-classifier formulation above.

### Phase 3 — National scan + per-site timeline reconstruction (not yet built)

Linear pipeline. No iterative loop.

1. **Heuristic-filtered national grid** — at MGRS-tile granularity over CONUS, exclude grid cells that fail the Phase 2a heuristics. Reads Sentinel-2 directly from AWS Open Data COGs (the public `sentinel-cogs` S3 bucket); no GEE quota involved.
2. **Lifecycle classifier inference on most-recent year (2025)** — for each surviving grid cell, infer { complete, partial, not_a_site }. Output: candidate locations classified as `complete` or `partial`.
3. **Spatial cluster** — group adjacent positives into site-level candidates.
4. **Per-candidate timeline reconstruction** — for each candidate location, run the lifecycle classifier on each historical year (2017→2024). The transition `not_a_site → partial → complete` dates the construction. Output: per-site timeline.
5. **NAIP epoch selection** — using the timeline, pick the NAIP survey covering the post-completion period for each site.
6. **NAIP refinement + site-type classification** — fetch the selected NAIP, run SAM for building polygons, run a downstream classifier (industrial vs not) on segmented polygons. (Builds on `phase3_refinement/pull_naip.py` / `sam_inference.py` / `mask_diff.py`.)
7. **Agent verification** — `prototypes/orchestrator/` dispatches per-candidate worker agents that search news, identify operator/project/sector, and build a multi-source timeline. Output: structured site record + NAICS tag.
8. **Aggregate** — verified records into the canonical BigQuery site table.

### Phase 4 — Steady state (later)

Periodic rescans on new S2 imagery; ongoing agent verification queue; classifier retraining as the verified site set grows. Iteration here is *additive labels*, not *bootstrap from noisy outputs*.

## Datasets

| Source | Resolution | Role |
| --- | --- | --- |
| **`COPERNICUS/S2_SR_HARMONIZED`** (GEE) | 10 m, B4/B3/B2/B8 | Phase 1 anchor chips (training data) |
| **AWS `sentinel-cogs`** | 10 m | Phase 3 national-scale inference reads |
| **`USDA/NAIP/DOQQ`** | 0.6–1 m, RGBN | Phase 3 step 6 footprint refinement |
| **`USGS/NLCD_RELEASES/2019_REL/NLCD/2019`** (GEE) | 30 m | Phase 2 heuristic land-cover filter |
| **`USGS/3DEP/10m_collection`** (GEE) | 10 m | Phase 2 heuristic slope/elevation filter |

S2 anchor chips are pulled once via GEE; national inference reads from AWS COGs (no quota). NAIP fires only on candidates surfaced by Phase 3 steps 2–4.

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

- **GEE for prep, AWS for inference.** GEE batch quota (3000-task queue, ~10/min) and even the high-volume endpoint don't scale to a million-chip national pass. Phase 1 chips were pulled via GEE; Phase 3 national inference reads `sentinel-cogs` directly from S3. Cloud-masking (SCL band) and seasonal compositing are done locally in inference code, not server-side on GEE.
- **Single-image, not pair-based.** The Phase 2 classifier sees one chip at a time and outputs a lifecycle label. Construction dating emerges from running it across each year of a candidate location, not from a learned change-detection function.
- **Recall over precision.** Every imagery stage (heuristic, lifecycle classifier, SAM) is tuned for recall. The agent verification step is the final precision gate. False negatives are permanent.
- **The 316 are training + validation, not the production cohort.** The pipeline finds sites that *aren't* in the 316. The 316 ground-truth detections.
- **Negatives are heuristic-derived, not bootstrap-derived.** No iterative loop where v1 trains on v0's mistakes. The training set is fixed at the start of Phase 2: 316 anchors × years (label by date) + heuristic-filtered random CONUS as not_a_site.

## Tech stack

- **Google Earth Engine** — Phase 1 anchor imagery + Phase 2 anchor feature sampling
- **AWS Open Data Sentinel-2 COGs** (`sentinel-cogs`) — Phase 3 national inference reads, no quota
- **Google Cloud Storage** — imagery + manifests + masks + intermediate staging
- **Foundation models / VLMs** for the lifecycle classifier — candidates: Prithvi-100M, Clay, SatMAE, RemoteCLIP, SkyCLIP. Backbone choice TBD when Phase 2b is implemented.
- **Vertex AI / Cloud Run with GPU** — SAM inference and lifecycle classifier inference
- **BigQuery** — canonical site table + per-detection candidate records
- Python orchestration: `earthengine-api`, `google-cloud-storage`, `google-cloud-bigquery`, `rasterio`, `shapely`, `pyproj`, `torch`/`transformers`

## Running

Copy `.env.example` to `.env` and fill in `GCP_PROJECT`, `GCS_BUCKET`, and (if not using ADC) `GEE_SERVICE_ACCOUNT` / `GEE_KEY_FILE`. Then:

```bash
pip install -r requirements.txt

# Phase 1 — already complete; included for reproducibility
python phase1_prep/pull_s2.py --workers 50           # S2 anchor + negative chips via high-vol endpoint
python phase1_prep/anchor_features.py                # NLCD / slope / road dist / dev dist per anchor

# Phase 2 — heuristic + lifecycle classifier (in progress)
# (heuristic mask builder + classifier training scripts to be added under phase2_*/)

# Phase 3 step 6 tooling — NAIP + SAM (existing pipeline, slots in late-stage)
python phase3_refinement/pull_naip.py --dry-run      # NAIP fetch for a site list
python phase3_refinement/pull_naip.py --poll-wait 60 # schedule + poll-loop
cd phase3_refinement && docker build -f Dockerfile.sam -t REGION-docker.pkg.dev/PROJECT/sam:latest .
python phase3_refinement/sam_inference.py            # batch SAM over manifest
python phase3_refinement/mask_diff.py                # mask processing → BigQuery
```

Manifests in GCS are the source of truth — re-runs only process work not yet recorded there.
