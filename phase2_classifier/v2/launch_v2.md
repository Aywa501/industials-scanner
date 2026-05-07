# Phase 3 v2 detector — EC2 launch

5-encoder embedding comparison (`dino_sat493m`, `dino_vitb`, `resnet50`,
`prithvi_300m`, `prithvi_600m`) over the v2 manifest, then 3-class probe
training. Runs single-process on one GPU box.

## 1. Local: build manifest + scenes index, push bundle

```
cd sites_us
python phase2_classifier/v2/v2_build_dataset.py        # data_us/v2_dataset_manifest.parquet (~103K rows / 5,659 groups)
python -m phase2_classifier.v2.v2_build_scenes_index   # data_us/v2_scenes_index.parquet (~33K scenes / 4,253 groups)
chmod +x phase2_classifier/v2/push_v2_bundle.sh phase2_classifier/v2/bootstrap_v2.sh
./phase2_classifier/v2/push_v2_bundle.sh
```

`v2_build_scenes_index.py` runs the STAC query for every (mgrs_tile, year)
once, with retry-with-backoff. **Do not** put STAC search back in the embed
loop — Element84 rate-limits aggressively (and often harder from a residential
IP than from EC2). If you must run the index build locally and see >5%
failures, re-run it from the EC2 box.

Pushes to `s3://industrials-scanner-us-west-2/v2-bundle/`:
- `v2_dataset_manifest.parquet`
- `v2_scenes_index.parquet`
- `code/sites_us/phase2_classifier/v2/{v2_train.py,v2_build_scenes_index.py,bootstrap_v2.sh,run_v2.sh}`
- `requirements_v2.txt`
- `.env`

## 2. Launch g6.2xlarge spot in us-west-2

g6.2xlarge (8 vCPU / 32 GB / L4 24 GB) is the right size for a single-process
run. Bandwidth (S2 reads from S3) is the bottleneck, not GPU — going larger
doesn't speed it up.

```
aws ec2 run-instances \
  --region us-west-2 \
  --instance-type g6.2xlarge \
  --instance-market-options 'MarketType=spot' \
  --image-id <DLAMI_AMI_ID> \
  --key-name "$AWS_EC2_KEY_NAME" \
  --security-group-ids <YOUR_SG> \
  --iam-instance-profile Name=<YOUR_INSTANCE_PROFILE> \
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=200,VolumeType=gp3}' \
  --instance-initiated-shutdown-behavior terminate \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=v2-probe-training}]'
```

Fallback: `g5.2xlarge` if g6 spot is tight.

**Do not shard.** A previous attempt with 4 shards on g6.2xlarge OOM'd
(per-worker RSS 5.5–9.2 GB) and split the GPU. The single-process design
deliberately uses `prep_pool=8` GIL-releasing threads to parallelize across
cores; the budget is sized for one process per box.

## 3. SSH and run

```
ssh -i "$AWS_EC2_KEY_PATH" ubuntu@<INSTANCE_PUBLIC_DNS>
curl -O https://industrials-scanner-us-west-2.s3.us-west-2.amazonaws.com/v2-bundle/bootstrap_v2.sh
BUCKET=industrials-scanner-us-west-2 nohup bash bootstrap_v2.sh > v2_train.out 2>&1 &
tail -f v2_train.out
```

`bootstrap_v2.sh` pulls the bundle, sets up the env, and launches `run_v2.sh`,
which `nohup`s `v2_train.py` and `tee`s to `v2_train.log`.

## 4. Architecture (what's in `v2_train.py`)

Mirrors v1's `infer_shard.py` pattern:
- `IO_WORKERS=32` (rasterio.open/read on S3)
- `PREP_WORKERS=8` (composite + percentile + resize + tensor build —
  numpy/PIL release the GIL, so 8 threads parallelize across cores)
- `PREP_CHUNK=256` bounds the pending-futures pile
- `BATCH_SIZE=32` per encoder
- `MEMORY_BUDGET_BYTES=20 GB` total (one process per box)
- `CLUSTER_EPS_M=5000` DBSCAN for chunk bbox
- Skip-if-output-exists at startup for crash recovery
- Per-encoder load_fn / forward_fn callables in `MODEL_REGISTRY` (no `kind` dispatch)

Per-batch log line:
```
[v2-train]   group 200/5659  filled=98  skipped=1179 (no_scn=899 no_rdr=0 no_dat=0 no_cmp=280) (0.4 tiles/s, ~78 hr left)
```

The script's `tiles/s` rate is `tiles_filled / elapsed`, so sparse zones with
many `no_scn` skips depress it artificially; trust wallclock-vs-progress over
the script's ETA until the run reaches CONUS-density mgrs tiles
(~group 500+).

## 5. Monitor

Watch for:
1. CPU% on the python PID climbing well above 100% in dense zones
   (proves `prep_pool` is parallelizing across cores).
2. RAM staying under 20 GB RSS (memory budget honored).
3. GPU util pulsing 30–70% during batch flushes.
4. `embed pass done` and probe training output at the end.

## 6. Pull artifacts

```
aws s3 sync s3://industrials-scanner-us-west-2/v2-artifacts/v2/ data_us/v2/
```

Artifacts (one set per encoder):
- `data_us/v2/emb_<MODEL>.npy`
- `data_us/v2/v2_embeddings_index.parquet`
- `data_us/v2/stage1_industrial_v2_<MODEL>.pt` (3-class probe)
- `data_us/v2/stage1_v2_train_report_<MODEL>.json`, `..._eval_report_<MODEL>.json`

## 7. Terminate

`bootstrap_v2.sh` ends with `sudo shutdown -h +1`; the instance was launched
with `InstanceInitiatedShutdownBehavior=terminate`. Verify in EC2 console.

## 8. Re-train iterations (cheap, no EC2)

Once embeddings are on S3, retraining probes is local:

```
aws s3 cp s3://industrials-scanner-us-west-2/v2-artifacts/v2/ data_us/v2/ --recursive
python sites_us/phase2_classifier/v2/v2_train.py --skip-embed
```
