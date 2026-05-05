# Stage 1 Industrial Classifier — Plan

## Goal

Cheap, recall-first binary classifier deciding `industrial` vs `not` on a single Sentinel-2 chip (B4/B3/B2/B8, 10 m, 256×256). Runs on every CONUS grid cell at Phase 3 step 2 as the load-bearing sieve. Recall > precision; quarries / solar / big-box are *expected* to pass through and get filtered downstream.

## Architecture

- **Backbone:** **DINOv3 ViT-L/16 with SAT-493M satellite weights** (`facebook/dinov3-vitl16-pretrain-sat493m`), 300M params, **frozen.** Outputs 1024-dim CLS-token features. Pretrained on a 493M-image satellite corpus — direct domain match to S2, no domain-gap mitigation needed.
- **Head:** linear probe — `nn.Linear(1024, 2)`. Cross-entropy loss, class-balanced sampling. Trained on the manual labels (~682 chips, currently growing as the 25 new negatives + relabel shortlist get labeled).
- **Input adapter:** S2 chips are B4/B3/B2/B8 — **drop B8 (NIR), use B4/B3/B2 only.** Apply the same 1–99 percentile stretch used in the labeling webapp to map to [0,1], then apply SAT-493M-specific normalization: mean `(0.430, 0.411, 0.296)`, std `(0.213, 0.156, 0.143)`. Resize 256 → 224 (ViT-L/16 native input).
- **Optimizer:** AdamW on head only, lr=1e-3, 30 epochs, early-stop on val F1.
- **Augmentation:** horizontal flip, ±10° rotation. **No color jitter** — would invalidate the SAT-493M normalization assumptions.

**Fallback ladder if linear probe < 0.85 val F1:**
1. Small MLP head: Linear(1024, 128) + ReLU + Dropout(0.2) + Linear(128, 2).
2. Concat features from multiple transformer blocks (e.g. last 4 CLS tokens) into a 4096-dim input to the head.
3. Drop to a smaller DINOv3 satellite variant (vitb16 / vits16) with last-block fine-tune — sometimes a smaller frozen backbone with a tiny amount of unfreezing beats a frozen 300M model on small label sets.

## Data

### Positive class

For each candidate site:
1. If `manual_site_notes.parquet` has an `IMPORTANT:` note for this site_id, override `site_type` to user-stated type from the note.
2. Drop sites with `bad_geocode` flag in `manual_site_flags.parquet`.
3. Drop sites with city-level geocodes (lat or lng has < 4 decimal places in source CSV).

Then:
- **All `brownfield` sites × all years** in the S2 manifest. (~38 sites in CSV, less low-precision drops.)
- **All `expansion_existing` sites × all years.** (~128 sites, less drops.)
- **Greenfield sites: only chips labeled `complete` or `partial`** in `manual_labels.parquet`. (~260 chips from manual labels.)

Estimated positive pool: ~1100–1500 chips.

### Negative class — embedding-distance filter

1. **Candidate pool:** all ~500 random-CONUS sites in the S2 manifest × all years (~4500 chips), plus the 50 random-CONUS negatives in the manual queue × all years labeled `not_a_site`.
2. **Embed everything** (positives + candidate negatives + manual-confirmed negatives) once with the frozen DINOv3 ViT-L/16 SAT-493M backbone — same model used downstream for the linear probe, so embeddings are reused for training and inference.
3. **For each candidate negative**, compute cosine distance to:
   - nearest positive embedding
   - nearest manual-confirmed-negative embedding
4. **Filter:** if NN is a positive → discard (likely industrial leak). Else keep.
5. **Relabel shortlist:** the ~30–50 candidates with the *smallest* distance to a positive (i.e., most positive-leaning) get surfaced for user review in the labeling webapp before they're discarded — confirms the filter is doing the right thing and rescues any false leaks.

Locked negative pool after filter: estimated ~3500 chips.

### Train/val/test splits

5-fold cross-validation **by site_id**, not by chip. A given site's chips never split across folds — prevents leakage from co-located years sharing visual signature.

Hold out a small chunk of unfiltered random-CONUS chips (no NN-filter applied) as a "wild" test set for false-positive bucketing.

## Eval

- Primary: by-site CV F1 / AUC on the locked training pool.
- **By-category FP analysis on the wild test set** — manually bucket false positives into:
  - `quarry`, `solar_farm`, `dense_suburban`, `large_parking`, `data_center`, `mine`, `landfill`, `airport`, `agricultural_processing`, `other`
  - Goal: instrument what Stage 2 will need to handle. Doesn't change training; informs design of any intermediate filter and/or lifecycle-classifier training data.
- Don't overfit to F1 — recall is the headline metric. Target: ≥ 0.95 recall on the held-out positive set, F1 secondary.

## Inference cost (sanity check)

- Heuristic-filtered CONUS at 10 m, 256×256 tiles, summer 2025 single epoch: estimate ~3M chips.
- DINOv3 ViT-L/16 forward at batch 32 on A10G: ~6 ms/chip amortized → ~5 hours for 3M chips.
- On an A100 with batch 64+: well under 1 hour.
- Storage: 1024-dim float16 embeddings × 3M = ~6 GB if we cache. Worth doing — downstream re-classification (alternate heads, hard-negative mining) is then free.

## Deliverables

Files to add under `sites_us/phase2_classifier/`:

- `build_dataset.py` — assembles positive/negative chip set per spec, writes `data_us/stage1_dataset.parquet` with columns `(site_id, year, chip_uri, label, site_type, source)`.
- `embed.py` — DINOv3 ViT-L/16 SAT-493M inference over the chip set, writes `data_us/stage1_embeddings.npy` + index.
- `filter_negatives.py` — embedding-NN filter for candidate negatives, writes filtered set + `data_us/stage1_relabel_shortlist.json` (chips for user to confirm via labeling webapp).
- `train_industrial.py` — 5-fold CV linear probe training, writes `data_us/stage1_industrial_v1.pt` + per-fold metrics.
- `eval_industrial.py` — wild test set inference + by-category FP bucketing, writes `data_us/stage1_eval_report.json`.

## Sequencing

1. **User: finish labeling the 25 new negatives** in the manual webapp (current step). Until done, the negative-pool size is wrong.
2. `build_dataset.py` → assemble train pool.
3. `embed.py` → DINOv3 SAT-493M over all chips.
4. `filter_negatives.py` → produces relabel shortlist.
5. **User: label the relabel shortlist** in the webapp.
6. Re-run `filter_negatives.py` with the user's confirmations to lock the negative pool.
7. `train_industrial.py` → fit linear probe + report CV metrics.
8. **Pause for review.** If F1 ≥ 0.85, proceed; else descend the fallback ladder.
9. `eval_industrial.py` → wild test FP bucketing.
10. **Pause for review.** Decide whether v1 is good enough to slot into Phase 3 step 2, or whether to invest in an intermediate filter / Prithvi pivot.

## Out of scope for this plan

- Phase 3 step 2 wiring (CONUS inference orchestration). Separate plan once we trust the model.
- Lifecycle classifier — moved to Phase 3 step 4.
- Active learning loop. The labeling round + relabel-shortlist is enough hand-curation for v1.
