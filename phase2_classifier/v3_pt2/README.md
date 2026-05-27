# v3 pt 2 — Stage 2b temporal stability filter

Drops v3 detector candidates that look identical in Landsat 2008 vs 2022 AND have no changed neighbor within 2 km. Slots between v3 scan and Phase 3 NAIP.

**Separate from the v3 CONUS scan** (`phase2_classifier/v3/v3_scan_infer.py`) — different inputs, different worker shape (CPU-only, lighter), separate bundle and bucket prefix. Do not bolt onto v3_scan_infer as a config.

## Pipeline

```
scan_results.parquet (696K)
    │
    ▼  build_candidates.py (local, instant)
stage2_candidates.parquet (345K @ p≥0.30)
    │
    ▼  build_scenes_index.py (local, ~10 min STAC)
landsat_scenes_index.parquet
    │
    ▼  push_bundle.sh → s3://bucket/v3-pt2-bundle/
    ▼  launch.sh → spot c6a.4xlarge in us-west-2
change_scan.py (EC2, ~1–2 h)
    │
    ▼  per-chunk parquet → s3://bucket/v3-pt2-artifacts/change_chunks/
    ▼  proximity_rescue.py (local, seconds)
stage3_candidates.parquet  →  Phase 3 NAIP entry
```

## Run

```bash
# Local
python3 sites_us/phase2_classifier/v3_pt2/build_candidates.py
python3 sites_us/phase2_classifier/v3_pt2/build_scenes_index.py
./sites_us/phase2_classifier/v3_pt2/push_bundle.sh

# EC2 — spot launcher (c6a.4xlarge primary, c6i.2xlarge fallback)
./sites_us/phase2_classifier/v3_pt2/launch.sh

# Tail
./sites_us/phase2_classifier/v3_pt2/monitor.sh        # uses /tmp/stage2b_instance_id.txt

# After EC2 finishes (auto-terminates via EXIT trap)
aws s3 sync s3://${BUCKET}/v3-pt2-artifacts/change_chunks/ data_us/phase2/v3/stage2b_change_chunks/
python3 sites_us/phase2_classifier/v3_pt2/proximity_rescue.py
```

## Telemetry

Three concurrent signals from the EC2 worker, all visible from local Mac:

1. **`v3-pt2-artifacts/logs/<instance>.log`** — stdout tail, synced every 30s (first 5 min) then 60s. One line per chunk: `chunk=X done=Y/Z cand=N err=M rate=R/s eta=Tm p50=Ps rss=Gg`.
2. **`v3-pt2-artifacts/heartbeat/<instance>.json`** — overall progress JSON, synced every `STAGE2B_HEARTBEAT_SEC` (default 30s). Includes elapsed, rate, ETA, error rate, mem RSS, mean per-cand time, mean fetch times.
3. **`v3-pt2-artifacts/logs/<instance>.stats.jsonl`** — per-chunk JSONL with full timing breakdown (`t_fetch_2022_mean_s`, `t_fetch_2008_mean_s`, `t_mask_mean_s`, p50/p90 per-cand) — post-run analysis.

**Stall detection:** if no chunk completes for `STAGE2B_STALL_SEC` (default 300s), `change_scan.py` dumps thread stacks via `faulthandler` to `s3://bucket/v3-pt2-artifacts/stacks/<instance>.txt`.

**CloudWatch:** the bootstrap pushes a 1-minute `stage2b/alive` metric per memory `monitor-heartbeat` — log-line silence is no longer ambiguous.

## Env knobs (change_scan.py)

| var | default | notes |
| --- | --- | --- |
| `STAGE2B_NUM_WORKERS` | `min(4×vCPU, 64)` | I/O-bound; oversubscribe |
| `STAGE2B_CHUNK_SIZE` | 200 | candidates per parquet chunk |
| `STAGE2B_HEARTBEAT_SEC` | 30 | heartbeat sync cadence |
| `STAGE2B_STALL_SEC` | 300 | stall watchdog → stack dump |
| `STAGE2B_MIN_FOOTPRINT_PX` | 10 | below → `ambiguous` |
| `STAGE2B_CHANGE_T` | 0.10 | rescue script — change threshold |
| `STAGE2B_PROX_KM` | 2.0 | rescue script — proximity radius |

## Resume

Bootstrap pulls existing chunks from S3 on start. `change_scan.py` scans `data_us/phase2/v3/stage2b_change_chunks/` for existing `chunk_NNNNN.parquet` and skips those. Safe across spot reclaims.

## Constraints (locked by user)

- p≥0.30 threshold (91% recall on industrial≥5000m², 50% cut on full scan)
- 15 m (1 pan pixel) bbox margin — no more
- 2022-derived mask applied to both years
- Within-year ratio (`mean[mask] / mean[~mask]`) for sensor-invariance
- Spot only (memory `never-on-demand`)
- GDAL env-var creds, no `AWSSession` (memory `no-awssession`)
