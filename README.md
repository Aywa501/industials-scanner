# Industrial Site Detection — Imagery Pipeline

## What this is

A remote-sensing pipeline that detects industrial-scale sites across the US, seeded from a labeled dataset of 316 manufacturing investment announcements (2015–2025). The 316 set is the *training seed* — the goal is a continuously-updated canonical table of every detected industrial site in the US, tagged by NAICS code, with multi-source timelines per site.

The pipeline is **staged single-image classification, not change detection.** Stage 1 is a cheap binary classifier (`industrial` vs `not`) that scans CONUS at S2 resolution as a recall-first sieve. Stage 2 (a per-candidate lifecycle classifier) reconstructs the construction timeline by running on each historical year of a surviving candidate. NAIP imagery refines confirmed industrial candidates into building footprints, and worker agents enrich each verified site with operator / project / NAICS information.

Negatives for Stage 1 come from heuristic-filtered random CONUS, with embedding-distance pruning to remove industrial leaks (random-CONUS samples that happened to land on real industry). The hard negatives Stage 1 lets through become free training data for Stage 2 and for any intermediate filter we slot between them.

Runs on Google Cloud — Earth Engine for imagery prep, Cloud Storage for staging, AWS Open Data Sentinel-2 COGs for national inference, Vertex AI / Cloud Run with GPU for SAM, BigQuery for the canonical site table.

## Pipeline phases

### Phase 0 — Announcement crawling (complete)

Produced `data_us/manufacturing_announcements_geocoded.csv`: 316 geocoded manufacturing investments (2015–2025). Phase 0 is closed; the dataset doesn't grow from new announcement research.

### Phase 1 — Anchor imagery + feature prep (complete)

- **NAIP archive** of the 316 anchor sites: 2242 GeoTIFFs at `gs://{bucket}/tiles/{site_id}/{year}_{image_id}.tif`, manifest at `gs://{bucket}/manifest/tile_manifest.parquet`. Used as a labeled validation set for the eventual NAIP-stage classifiers.
- **Sentinel-2 anchor chips**: yearly summer (Jun–Aug) median composites, B4/B3/B2/B8 at 10 m, 256×256 px (1.28 km buffer), for 316 anchors + ~500 random CONUS sites × 9 years (2017–2025). Manifest at `gs://{bucket}/manifest/s2_chip_manifest.parquet`. Pulled via `phase1_prep/pull_s2.py` against GEE's high-volume endpoint (`ee.data.computePixels`).
- **Anchor feature analysis** (`phase1_prep/anchor_features.py`): NLCD land-cover class, slope, elevation, distance to road, distance to nearest developed pixel — sampled at each anchor. Empirical findings inform the Phase 2 heuristic.

### Phase 2 — Manual labeling + Stage 1 industrial classifier (current)

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

**Step 2c — Stage 1 industrial classifier (next).** Cheap recall-first binary classifier, runs on every CONUS grid cell in Phase 3 step 2. See plan in `phase2_classifier/PLAN.md`.

- **Backbone:** DINOv3 ViT-L/16 with SAT-493M satellite weights (`facebook/dinov3-vitl16-pretrain-sat493m`), 300M params, frozen. 1024-dim CLS features, pretrained on a 493M-image satellite corpus — direct domain match to S2.
- **Head:** linear probe (Linear(1024, 2)), trained with cross-entropy and class-balanced sampling.
- **Input:** B4/B3/B2 only (drop B8 NIR), 1–99 percentile stretch to [0,1], SAT-493M-specific normalization (mean 0.430/0.411/0.296, std 0.213/0.156/0.143), resize 256→224.
- **Positive set:** all brownfield + extension sites × all years (industrial pre-existing); greenfield sites only on years labeled `complete` or `partial` in `manual_labels.parquet`. IMPORTANT-flagged sites override CSV `site_type` to user-stated type.
- **Negative set:** random-CONUS sites + manual-confirmed negatives, with an embedding-distance filter. Embed all candidates, drop any whose nearest neighbor is a positive (likely industrial leak in the random sampler). Surfaces a relabel shortlist for review before locking.
- **Tradeoff (accepted):** the embedding filter prunes hard negatives along with leaks. Stage 1 is therefore high-recall, lower-precision — quarries, solar farms, big-box distribution will pass through. Refinement is deferred to Stage 2 (lifecycle) and Phase 3 step 7 (agent verification), which use the harder candidates as free training data later.

**Lifecycle classifier — deferred to Phase 3 step 4.** The 3-class lifecycle classifier ({ complete, partial, not_a_site }) is no longer in Phase 2; it runs as part of per-candidate timeline reconstruction on the smaller candidate volume Stage 1 surfaces. This is also where it makes most sense — lifecycle is genuinely hard for brownfield/extension (already-industrial surroundings), so running it on a Stage-1-filtered candidate pool is more tractable than on raw CONUS.

**Note on prior failed v0 detector.** Earlier work in this phase trained a siamese ResNet-18 on chip-pair labels for binary change detection. Three structural revisions (label tightening, summer compositing, diff-feature head) all plateaued at val AP ≈ 0.21 — data-bound. Replaced by the staged-classifier formulation above.

### Phase 3 — National scan + per-site refinement (not yet built)

Linear pipeline. No iterative loop.

1. **Heuristic-filtered national grid** — at MGRS-tile granularity over CONUS, exclude grid cells that fail the Phase 2a heuristics. Reads Sentinel-2 directly from AWS Open Data COGs (the public `sentinel-cogs` S3 bucket); no GEE quota involved.
2. **Stage 1 industrial classifier inference on most-recent year (2025)** — surfaces candidate locations classified as `industrial`.
3. **Spatial cluster** — group adjacent positives into site-level candidates.
4. **Per-candidate lifecycle timeline** — for each candidate, run the lifecycle classifier on each historical year (2017→2024). The transition `not_a_site → partial → complete` dates the construction. Output: per-site timeline with completion year + confidence.
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
- **Staged classification, not pair-based change detection.** Stage 1 (industrial vs not) is the cheap CONUS-wide sieve. Stage 2 (lifecycle) runs on the candidate volume Stage 1 surfaces, dating construction by year-by-year inference. Each stage is tuned for recall; precision comes from the agent verification step.
- **CSV `site_type` is noisy.** ~14% of greenfields are mislabeled per manual S2 inspection. The `IMPORTANT:` notes in `manual_site_notes.parquet` carry the user's overrides — apply them during dataset construction.
- **The 316 are training + validation, not the production cohort.** The pipeline finds sites that *aren't* in the 316.
- **Recall over precision at every imagery stage.** False negatives are permanent; false positives get filtered by the next stage or by agents.

## Tech stack

- **Google Earth Engine** — Phase 1 anchor imagery + Phase 2 anchor feature sampling
- **AWS Open Data Sentinel-2 COGs** (`sentinel-cogs`) — Phase 3 national inference reads, no quota
- **Google Cloud Storage** — imagery + manifests + masks + intermediate staging
- **DINOv3 ViT-L/16 SAT-493M** (frozen) — Stage 1 industrial classifier backbone. Pretrained on satellite imagery, no domain-gap mitigation needed. Linear probe head trained on the 682-chip manual-label set.
- **Vertex AI / Cloud Run with GPU** — SAM inference and (later) batch lifecycle classifier inference
- **BigQuery** — canonical site table + per-detection candidate records
- Python orchestration: `earthengine-api`, `google-cloud-storage`, `google-cloud-bigquery`, `rasterio`, `shapely`, `pyproj`, `torch`/`transformers`

## Running

Copy `.env.example` to `.env` and fill in `GCP_PROJECT`, `GCS_BUCKET`, and (if not using ADC) `GEE_SERVICE_ACCOUNT` / `GEE_KEY_FILE`. Then:

```bash
pip install -r requirements.txt

# Phase 1 — already complete; included for reproducibility
python phase1_prep/pull_s2.py --workers 50           # S2 anchor + negative chips via high-vol endpoint
python phase1_prep/anchor_features.py                # NLCD / slope / road dist / dev dist per anchor

# Phase 2b — manual labeling webapp (used to produce manual_labels.parquet)
python -m sites_us.phase2_classifier.labeling_webapp.prep_data
python -m uvicorn sites_us.phase2_classifier.labeling_webapp.server:app --reload --port 8765

# Phase 2c — Stage 1 industrial classifier (next; scripts to be added under phase2_classifier/)

# Phase 3 step 6 tooling — NAIP + SAM (existing pipeline, slots in late-stage)
python phase3_refinement/pull_naip.py --dry-run      # NAIP fetch for a site list
python phase3_refinement/pull_naip.py --poll-wait 60 # schedule + poll-loop
cd phase3_refinement && docker build -f Dockerfile.sam -t REGION-docker.pkg.dev/PROJECT/sam:latest .
python phase3_refinement/sam_inference.py            # batch SAM over manifest
python phase3_refinement/mask_diff.py                # mask processing → BigQuery
```

Manifests in GCS are the source of truth — re-runs only process work not yet recorded there.
