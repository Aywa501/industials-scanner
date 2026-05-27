# Phase 3 NAIP Stage — Pipeline Design (2026-05-22)

> **STATUS — DEAD END (2026-05-25).** This iteration is abandoned in favor of a different Stage 2 (v3) approach worked on separately, which produces a different candidate set and changes what the right Stage 3 looks like. The SAM 3 + NAIP architecture below works (benchmarked 134.6s / 100 clusters on g6e.2xlarge, ~32 hr / ~$25 for the full 113,966-cluster filtered set), but the per-cluster compute is too expensive for the scale we're now targeting. Pieces likely worth carrying into the next iteration:
> - **SAM 3 text-prompted segmentation** with a small prompt set (5: industrial building, warehouse, parking lot, storage tank, silo). Encoder cost dominates (~0.34s/cluster vs ~0.035s/cluster/prompt) so trimming prompts has diminishing returns.
> - **VRAM-budget batch admission** (`probe_batch_n()` in `naip_sam.py`) — measures peak alloc at N=1 and N=2, fits batch size to GPU budget. Auto-tunes across hardware.
> - **Batched SAM 3 forward** — one decoder call per N-image × M-prompt batch (vision FPN repeated M times, prompt tokens tiled N times). Replaces the original per-prompt loop.
> - **ProcessPool postproc** with `mp_context=spawn` for the mask→polygon→reproject path (GIL-bound under shapely + pyproj).
> - **EC2 sweet-spot finding**: g6e.2xlarge (8 vCPU, L40S 48 GB) is the right size for this workload — postproc drain just exceeds SAM production, no wasted CPU. Verified via telemetry, NOT just stage-summed numbers (steady-state GPU 100% + CPU 82% on probe samples).
> - **OSM-cut clustering** (`cluster_osm_cut.py`) for breaking 155 km mega-blobs into road-bounded sub-clusters. Stable across NC/CONUS tests.
> - Stage-summed timing is misleading when stages parallelize across pools — verify with a system telemetry sample (cpu_pct / gpu_util_pct) BEFORE concluding which stage is the bottleneck.
>
> Do **NOT** launch the full prod run. The benchmarks at shards 19 and 531 are the most that should be spent on this architecture.

---


## Purpose

Stage 3 Part 2 of the industrials detection pipeline. Takes the 22,629 S2 DBSCAN candidates in `data_us/phase3_scan/phase3_candidates_v2.parquet` and refines them into discrete industrial **sites**, using NAIP aerial imagery + SAM 3 text-prompted segmentation + OSM/Overture/location context + a model/LLM reasoning step. Output feeds the web verification agent (operator / NAICS resolution, multi-source timeline).

Supersedes the SAM 1 AMG + hand-labeled GBT classifier design (Design A in the earlier notes; the approved plan at `.claude/plans/eventual-stirring-pillow.md`). The SAM 3 + reasoning architecture eliminates the manual-labeling sub-task and gives the downstream step actual category labels instead of having to derive them from heuristics on raw masks.

## Pipeline

### Step 1 — Overture retrieval + drop obvious non-industrial

For each S2 candidate, retrieve every Overture building within candidate bbox + retrieval buffer (~1500 m; half the S2 tile-width plus context). **Drop** (not flag) buildings that are confidently non-industrial.

Drop set (calibration items in parentheses):
- `class` ∈ {house, detached, terrace, semidetached_house, apartments, bungalow, dormitory, residential, garage, garages, shed, hut, cabin, houseboat, static_caravan} *(residential — locked from v1)*
- *plus, after widening:* `class` ∈ {commercial, retail, school, university, college, church, mosque, synagogue, religious, hotel, stadium, grandstand, civic, government, hospital} *(needs sample inspection before locking)*
- `subtype == "residential"`
- `area_m2 < AREA_FLOOR_M2` *(v1 default 100 m²; calibration item)*

Kept: `industrial / warehouse / hangar / NaN`-class buildings above the area floor.

A building retrieved by multiple S2 candidates is assigned to the one with highest `max_prob` (no duplication).

Drop candidates with zero kept buildings (the Overture gate).

### Step 2 — Buffer-merge into loose clusters

Project kept buildings to EPSG:5070. Expand each building's bbox by the merge buffer and union overlapping into connected components → clusters. **Over-merge is preferred** — Step 3's OSM cut bounds the chains, and downstream stages partition with richer evidence.

Merge buffer: ~300-500 m (calibration item — can be more generous than v1 since OSM cut bounds extent).

Output: per-cluster summary (cluster_id, member buildings, bbox, originating candidate_id(s), max S2 prob).

### Step 3 — OSM-cut clusters conservatively

For each cluster, query OSM roads (static national extract — TIGER/Line is the in-repo precedent at `phase1_prep/anchor_features.py`; OSM Geofabrik US is an alternative) intersecting the cluster bbox. Cut the cluster wherever a road of a "cut class" separates two building groups.

Cut classes (calibration item; v1 default):
- *Cut*: residential, tertiary, secondary, primary, motorway, trunk.
- *Do not cut*: service, unclassified, track, footway, rail, internal driveways.

Implementation: build a graph where building polygons are nodes and edges exist between two buildings within the merge buffer AND not separated by a cut-class road. Connected components → final clusters. The graph-with-OSM-blockers replaces the v1 pure-buffer-merge.

Output: bounded clusters with member buildings. The 155 km buffer-merge blob (`c_0000000`) becomes O(100-1000) road-bounded sub-clusters.

### Step 4 — NAIP fetch + SAM 3 segmentation

Per cluster: fetch NAIP COGs at native resolution within cluster bbox + fetch buffer (~100-200 m). Read in-region us-west-2 from `s3://naip-analytic/{state}/{year}/{res}/rgbir_cog/`, requester-pays. Mosaic via `WarpedVRT` → EPSG:5070 (implemented in `naip_sam.py:read_naip_mosaic`).

Run **SAM 3** with text prompts to get **labeled** masks. Prompt set (calibration item; v1 candidates):
- "industrial building"
- "warehouse"
- "office building"
- "parking lot"
- "loading dock"
- "tank" / "silo"
- "vegetation" / "tree"
- "road"

Output per cluster (parquet, S3-cached at `s3://industrials-scanner-us-west-2/phase3-naip-sam3/{cluster_id}/masks.parquet`):
- mask_id, label, label_score
- polygon (lon/lat, WKT)
- area_m2, area_px
- mean R/G/B/NIR
- shape stats (rectangularity, aspect, elongation)

Resumable: skip clusters whose output already exists in S3.

**Fallback if SAM 3 unavailable**: Grounded SAM (Grounding-DINO box prompts from text → SAM 1/2 mask refinement). Same output contract.

### Step 5 — Site selection (model / LLM reasoning)

Per cluster, combine SAM 3 labels + polygons + Overture metadata + OSM context + location → identify discrete sites within the cluster.

Two-tier approach (architecture is a calibration item):
- **Bulk path**: embed each cluster's feature vector (SAM 3 label distribution + polygon shape stats + Overture summaries + OSM proximities + S2 max_prob) → small learned classifier or rule set picks site groupings + filters non-industrial false-positive clusters.
- **Tail path**: for ambiguous clusters, or where shared infrastructure suggests merging (shared parking lot, loading apron straddling buildings), invoke an LLM with the spatial + label context to reason.

May extend cluster bboxes via shared infrastructure (e.g., a SAM 3 "parking lot" polygon adjacent to two building groups → those groups are one site).

Errs **conservative** — over-grouping is preferred to over-splitting; agent verification can still split.

Output: per-site polygon set + feature vector + selection score + provenance back to cluster + originating S2 candidate.

### Step 6 — Hand-off to web verification agent

Sites pass to the agent step with: polygon set, NAIP imagery, SAM 3 labels, feature vector, selection score. Agent does operator resolution, NAICS classification, multi-source timeline construction, single/multi-tenant resolution.

## Current build status

| Module | Status | Notes |
|---|---|---|
| `overture_groups.py` | **done** | Prune A drops outright + 100 m² area floor; widened drop set (education / religious / medical / hospitality / sports / agricultural / office); 22,629 → 21,965 candidates |
| `osm_cut_test.py` | done | single-candidate harness; go/no-go ran on `c_0000000` |
| `cluster_osm_cut.py` | **done** | Step 2+3 combined; per-state Geofabrik download + STRtree cut; NC tested (818 cands → 44,774 clusters; 2 > 5 km in Charlotte distribution corridor — accepted as over-merge) |
| `build_naip_manifest.py` | done | now consumes `clusters.parquet`; the 2 km sub-cluster grid is gone |
| `naip_sam.py` | **done (SAM 3)** | `facebook/sam3` via HF transformers; prompt set env-configurable (`SAM3_PROMPTS`); labelled masks; S3 prefix `phase3-naip-sam3/{cluster_id}/`. **Open**: transformers pin audit — 4.56.0 (current pin) likely predates SAM 3; expected min ≥ 4.57. NAIP-read path verified locally on a NC cluster. |
| `overlay_inspect.py` | done | visual debug; updated for cluster_id + cluster_buildings.parquet |
| `select_sites.py` | **new** | Step 5 reasoning |
| EC2 deploy scripts | pending | bootstrap_naip.sh, push_naip_bundle.sh, launch_naip.sh |

## Carried over (don't rebuild)

- `data_us/phase3_scan/phase3_candidates_v2.parquet` — input.
- `data_us/external/overture_industrial_conus_2025_aligned.parquet` — Overture (bbox-only).
- `data_us/phase3_naip/naip_tile_index.parquet` — 216k NAIP COG tiles indexed (47 states, naming convention per state-year).
- `data_us/phase3_naip/candidate_buildings.parquet`, `candidates_with_buildings.parquet` — Step 1-2 output.
- NAIP COG read pattern (`naip_sam.py:read_naip_mosaic` — WarpedVRT to EPSG:5070).
- AWS auth (`.env` IAM user locally; instance role on EC2; requester-pays for NAIP).
- EC2 spot launch pattern at `sites_us/phase3_scan/v2/bootstrap_v2_scan.sh`.

## Go / no-go gate — **PASSED 2026-05-22**

OSM-cut on `c_0000000` (24,044 survived_a buildings, 155 km extent) using Geofabrik NC roads filtered to cut-classes (motorway/trunk/primary/secondary/tertiary + their `_link`s + residential):

| merge buffer | clusters | median size | max size | max extent | clusters > 3 km | clusters > 5 km |
|---|---|---|---|---|---|---|
| 300 m | 7,712 | 2 buildings | 90 | 2,452 m | 0 | 0 |
| 500 m | 6,778 | 2 buildings | 93 | 2,668 m | 0 | 0 |

Top clusters by extent inspected (Goldsboro / Snow Hill / Princeton NC commercial corridors) are road-bounded mixed-use districts at 2-2.5 km. Over-merge of commercial corridors into one cluster is acceptable — downstream Step 5 + agent verification break it down further. The 155 km chain is decisively broken.

Implementation: `sites_us/phase3_naip/osm_cut_test.py`. Single-candidate harness; the production cluster step (#29) lifts the same logic across all candidates.

Roads source: `data_us/external/osm/nc/north-carolina-latest-free.shp.zip` (Geofabrik, ~776 MB; `gis_osm_roads_free_1.shp`). Production needs the per-state extracts for the full CONUS sweep.

## Build order

1. **Go/no-go**: OSM-cut verification on `c_0000000`.
2. Edit `overture_groups.py`: flag → drop, widen drop set + area floor.
3. Build `cluster_buffer_merge.py` + `cluster_osm_cut.py`.
4. Update `build_naip_manifest.py` to per-cluster manifest.
5. Swap `naip_sam.py` to SAM 3 (or Grounded SAM fallback if SAM 3 not ready).
6. Complete EC2 deploy.
7. Benchmark SAM 3 cost on a sample, size the full run.
8. Build `select_sites.py` (Step 5).
9. Define hand-off contract for the agent step.

## Open calibration items

- Step 1 drop set widening (commercial / retail / etc.) + area floor.
- Step 2 buffer-merge buffer size.
- Step 3 OSM cut-class list.
- Step 4 SAM 3 prompt set + fetch buffer.
- Step 5 reasoning architecture (embeddings + classifier vs LLM split).

## Out of scope (this stage)

- Lifecycle / multi-NAIP-year reasoning.
- Manufacturing vs warehouse vs data-center typing (agent step).
- Single- vs multi-tenant resolution (agent step).
- New-construction recall — Overture-gated by design (sites without Overture footprints not in scope; this stage feeds training data, the next iteration may extend).

## References

- Memory: `phase3-naip-stage` (status), `evidence-based-cluster-bounds` (Step 3 rationale), `prefer-overmerge-in-coarse-stages` (Steps 1-3 ethos), `naip-s3-us-west-2` (bucket details), `sentinel-cogs-anonymous` (AWS auth gotchas).
- Earlier plan: `.claude/plans/eventual-stirring-pillow.md` — the SAM 1 + GBT design; **superseded** by this doc.
- Code reuse: `sites_us/phase3_scan/v2/infer_shard_v2.py:_setup_rasterio_env` (requester-pays AWSSession pattern), `sites_us/phase3_refinement/sam_inference.py` (SAM 1 pattern, kept as reference; SAM 3 needs new wiring), `sites_us/phase3_scan/v2/bootstrap_v2_scan.sh` (EC2 spot launch + EXIT-trap termination + IMDSv2 guard pattern).
