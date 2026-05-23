#!/usr/bin/env bash
# Bootstrap a g6.8xlarge (primary) or g6.4xlarge (fallback) in us-west-2
# (DLAMI: Deep Learning OSS PyTorch 2.x) to run the Phase 3 v2 CONUS scan
# with dino_vitb + Overture pre-filter using the streaming pipeline.
#
# Usage on the EC2 host as ubuntu (bucket is private — use instance-role
# auth via `aws s3 cp`, not anonymous `curl`):
#   aws s3 cp s3://${BUCKET}/scan-v2-bundle/bootstrap_v2_scan.sh .
#   BUCKET=industrials-scanner-us-west-2 nohup bash bootstrap_v2_scan.sh > scan_v2.out 2>&1 &
#
# The script pulls the rest of the bundle from s3://$BUCKET/scan-v2-bundle/.

set -euo pipefail

BUCKET="${BUCKET:?set BUCKET=industrials-scanner-us-west-2 (or your bucket name)}"
WORK="${HOME}/scan-v2"
mkdir -p "$WORK"
cd "$WORK"

# Bootstrap creds: an SCP step from the launching machine put .env and
# bootstrap_v2_scan.sh in $HOME. Source .env first so the s3 sync below has AWS keys.
if [ -f "$HOME/.env" ]; then
  set -a; . "$HOME/.env"; set +a
fi

echo "[bootstrap-v2] pulling bundle from s3://${BUCKET}/scan-v2-bundle"
aws s3 sync "s3://${BUCKET}/scan-v2-bundle/" ./bundle/ --only-show-errors

mkdir -p data_us sites_us data_us/v2/probes
cp bundle/phase3_grid.parquet       data_us/
cp bundle/phase3_scenes.parquet     data_us/
cp bundle/overture_industrial_conus_2025_aligned.parquet data_us/
cp bundle/probe_dino_vitb.pt        data_us/v2/probes/
cp -r bundle/code/sites_us/*        sites_us/
cp bundle/.env                      sites_us/.env

# Export env so HF + AWS creds are visible to the worker
set -a
. ./sites_us/.env
set +a

# DLAMI Nov-2025+ ships PyTorch as conda env at /opt/conda/envs/pytorch.
# Older DLAMIs use the /opt/pytorch venv. Support both.
if [ -f /opt/pytorch/bin/activate ]; then
  # venv activate references $LD_LIBRARY_PATH unguarded — trips set -u.
  set +u
  source /opt/pytorch/bin/activate
  set -u
elif [ -f /opt/conda/etc/profile.d/conda.sh ]; then
  # conda's activate.d scripts reference unbound vars (MKL_INTERFACE_LAYER etc.)
  # which trips `set -u`; relax around activation.
  set +u
  source /opt/conda/etc/profile.d/conda.sh
  conda activate pytorch
  set -u
else
  echo "[bootstrap-v2] ERROR: no PyTorch env found at /opt/pytorch or /opt/conda/envs/pytorch" >&2
  exit 1
fi

# transformers pinned to the version that worked during v2 training (loaded
# the same dino_vitb model on the same DLAMI). Newer transformers releases
# register an MoE custom_op that PT 2.4.1's torch.library.infer_schema can't
# parse, crashing AutoModel.from_pretrained at import time.
pip install --quiet --upgrade \
  "transformers==4.56.0" \
  "rasterio>=1.3.9" \
  "boto3>=1.34.0" \
  "mgrs>=1.4.6" \
  "pyproj>=3.6.0" \
  "pandas>=2.1.0" \
  "pyarrow>=14.0.0" \
  "Pillow>=10.0.0" \
  "python-dotenv>=1.0.0" \
  "huggingface_hub>=0.19.0" \
  "pystac-client>=0.7.0" \
  "py-spy>=0.3.14"

# HF login for gated dino_vitb model
if [ -n "${HF_TOKEN:-}" ]; then
  echo "[bootstrap-v2] hf login"
  python - <<PY
from huggingface_hub import login
import os
login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
PY
fi

# Sanity check: GPU + S3 read
python - <<'PY'
import torch, rasterio
print("CUDA:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("rasterio:", rasterio.__version__)
PY

# IMDSv2 for instance metadata (needed for terminate-instances at end)
echo "[bootstrap-v2] querying instance metadata via IMDSv2..."
IMDS_TOKEN=$(curl -sS -X PUT "http://169.254.169.254/latest/api/token" \
  -H "X-aws-ec2-metadata-token-ttl-seconds: 60" \
  --connect-timeout 3 --max-time 5)
INSTANCE_ID=$(curl -sS \
  -H "X-aws-ec2-metadata-token: $IMDS_TOKEN" \
  http://169.254.169.254/latest/meta-data/instance-id \
  --connect-timeout 3 --max-time 5)
REGION=$(curl -sS \
  -H "X-aws-ec2-metadata-token: $IMDS_TOKEN" \
  http://169.254.169.254/latest/meta-data/placement/region \
  --connect-timeout 3 --max-time 5)
INSTANCE_TYPE=$(curl -sS \
  -H "X-aws-ec2-metadata-token: $IMDS_TOKEN" \
  http://169.254.169.254/latest/meta-data/instance-type \
  --connect-timeout 3 --max-time 5)
echo "[bootstrap-v2] instance: $INSTANCE_ID, type: $INSTANCE_TYPE, region: $REGION"

# Worker tunables. The streaming pipeline uses MEMORY_BUDGET as a backpressure
# cap on in-flight loaded scene-data — set well below RAM to leave headroom for
# prep_pool composites and model state. SUB_BBOX_KM controls bulk-load
# granularity; ~12 km keeps each load under ~150 MB so libcurl HTTP/2 doesn't
# saturate even on the post-Nov-2025 DLAMI stack.
case "$INSTANCE_TYPE" in
  g4dn.4xlarge|g6.4xlarge)
    # 16 vCPU, 64 GB RAM. Budget = in-flight scene_data only (model ~0.5 GB,
    # batch tensors + composites ~1 GB, libs ~3 GB); 50 GB leaves comfortable
    # headroom while letting the pipeline run as wide as PIPELINE_DEPTH allows.
    export INFER_IO_WORKERS=32
    export INFER_PREP_WORKERS=14
    export INFER_MEMORY_BUDGET_GB=50
    export INFER_SUB_BBOX_KM=12
    export INFER_PIPELINE_DEPTH=8
    export INFER_PRODUCER_THREADS=1
    export INFER_PREP_PIPELINE=4
    ;;
  g6.8xlarge)
    # 32 vCPU, 128 GB RAM. Single producer: rasterio/GDAL is unstable under
    # multi-thread reads against shared Datasets; we get the throughput from
    # GPU offload + prep-pipelining instead. Lock-based serialization wasn't
    # enough to stop hangs even with HTTP/1.1.
    export INFER_IO_WORKERS=32
    export INFER_PREP_WORKERS=30
    export INFER_MEMORY_BUDGET_GB=100
    export INFER_SUB_BBOX_KM=12
    export INFER_PIPELINE_DEPTH=16
    export INFER_PRODUCER_THREADS=1
    export INFER_PREP_PIPELINE=6
    ;;
  *)
    echo "[bootstrap-v2] WARN: unknown instance type '$INSTANCE_TYPE', using worker defaults"
    ;;
esac
echo "[bootstrap-v2] worker tunables: IO=${INFER_IO_WORKERS:-default} PREP=${INFER_PREP_WORKERS:-default} MEM=${INFER_MEMORY_BUDGET_GB:-default}GB SUB_BBOX=${INFER_SUB_BBOX_KM:-default}km DEPTH=${INFER_PIPELINE_DEPTH:-default} PRODUCERS=${INFER_PRODUCER_THREADS:-default} PREP_PIPELINE=${INFER_PREP_PIPELINE:-default}"

# GDAL HTTP settings. rasterio.Env() is thread-local — settings made there
# don't reach io_pool worker threads. Set at process level so all threads
# (including worker threads) inherit them. KEEP HTTP/2 multiplex enabled:
# under HTTP/1.1 libcurl caps at ~6-8 connections per host, which deadlocks
# 32 io_pool workers waiting for the pool. HTTP/2 lets all 32 share a handful
# of multiplexed connections. The original 10SDJ multiplex deadlock was the
# giant single bulk-read pattern, already fixed by SUB_BBOX_KM=12.
export GDAL_HTTP_TIMEOUT=120
export GDAL_HTTP_CONNECTTIMEOUT=30
echo "[bootstrap-v2] gdal: TIMEOUT=$GDAL_HTTP_TIMEOUT (HTTP/2 multiplex default)"

# Pull existing results from S3 so already-done shards are skipped on this
# fresh instance. Without this, the worker re-processes every shard from
# scratch even when 94 are already on S3 from earlier runs.
mkdir -p data_us/phase3_results_v2
echo "[bootstrap-v2] pulling existing results from s3://${BUCKET}/phase3_results_v2/"
aws s3 sync "s3://${BUCKET}/phase3_results_v2/" data_us/phase3_results_v2/ --only-show-errors

# Ensure the Sentinel-2 scenes index covers the full grid before scanning.
# The bundle ships a complete index built locally; find_s2_scenes is resumable
# and self-short-circuits ("index already complete") when nothing is missing,
# so this is a ~2s no-op in the normal case and a self-heal if the index is
# partial. EC2's datacenter IP is not rate-limited by Element84 the way a
# residential IP is, so a rebuild here is reliable.
echo "[bootstrap-v2] verifying scenes index (find_s2_scenes, resumable)"
( cd sites_us && SCENES_WORKERS=40 python -u -m phase3_scan.find_s2_scenes )

# Build work list = MGRS tiles with scenes but no result yet.
python - <<'PY'
from pathlib import Path
import pandas as pd
scenes = pd.read_parquet("data_us/phase3_scenes.parquet")
done = {p.stem for p in Path("data_us/phase3_results_v2").glob("*.parquet")
        if not p.stem.endswith("_emb")}
todo = sorted(set(scenes.mgrs_tile) - done)
Path("mgrs_todo.txt").write_text("\n".join(todo))
print(f"[bootstrap-v2] {len(todo)} MGRS shards to process ({len(done)} already done)")
PY

# Periodic background sync — spot-interrupt safety. Loses ≤5 min of work if
# AWS reclaims the instance. Also syncs the bootstrap+worker log so we can
# diagnose failures even when the instance is gone.
mkdir -p data_us/phase3_results_v2
LOG_S3="s3://${BUCKET}/scan-v2-logs/${INSTANCE_ID}.out"
(
  while true; do
    sleep 300
    aws s3 sync data_us/phase3_results_v2/ "s3://${BUCKET}/phase3_results_v2/" \
        --only-show-errors || true
    aws s3 cp "$HOME/scan_v2.out" "$LOG_S3" --only-show-errors || true
  done
) &
SYNC_PID=$!

# Auto-terminate on exit. Sync log + results FIRST so the final state (including
# whatever killed us) is captured before the instance disappears.
on_exit() {
  echo "[bootstrap-v2] EXIT: final log + results sync"
  aws s3 cp "$HOME/scan_v2.out" "$LOG_S3" --only-show-errors || true
  aws s3 sync data_us/phase3_results_v2/ "s3://${BUCKET}/phase3_results_v2/" \
      --only-show-errors || true
  kill $SYNC_PID 2>/dev/null || true
  echo "[bootstrap-v2] terminating $INSTANCE_ID"
  aws ec2 terminate-instances --region "$REGION" \
    --instance-ids "$INSTANCE_ID" 2>&1 || true
}
trap on_exit EXIT

# Run the scan, with retry + stall watchdog. rasterio's libcurl multi-handle
# deadlocks under sustained concurrent reads on this DLAMI (both HTTP/1.1 pool
# exhaustion and HTTP/2 multiplex stall variants observed). The watchdog kills
# python if no new shard completes in STALL_SEC seconds; the retry loop
# relaunches up to MAX_ATTEMPTS times. Worker checkpoints per-MGRS via
# out_path.exists() so already-done shards are skipped on resume — mgrs_todo.txt
# is built once and reused.
cd sites_us
MAX_ATTEMPTS=30
STALL_SEC=600
for attempt in $(seq 1 $MAX_ATTEMPTS); do
  echo "[bootstrap-v2] === python attempt $attempt of $MAX_ATTEMPTS ==="
  # Start watchdog in background — kills python if no new ' done:' line in STALL_SEC.
  # `grep -c` returns exit 1 with zero matches AND prints "0", so a naive
  # `|| echo 0` would produce "0\n0". Suffix `; true` instead.
  (
    last=$(grep -c " done:" "$HOME/scan_v2.out" 2>/dev/null; true)
    last=${last:-0}
    stall=0
    while sleep 60; do
      cur=$(grep -c " done:" "$HOME/scan_v2.out" 2>/dev/null; true)
      cur=${cur:-0}
      if [ "$cur" = "$last" ]; then
        stall=$((stall + 60))
        echo "[watchdog] no new shard for ${stall}s (limit ${STALL_SEC}s, done so far=$cur)" \
          | tee -a "$HOME/scan_v2.out"
        if [ "$stall" -ge "$STALL_SEC" ]; then
          echo "[watchdog] STALL: no new shard in ${STALL_SEC}s, killing python" \
            | tee -a "$HOME/scan_v2.out"
          pkill -9 -f infer_shard_v2 2>/dev/null
          exit 0
        fi
      else
        echo "[watchdog] progress: $((cur - last)) new shard(s), total=$cur" \
          | tee -a "$HOME/scan_v2.out"
        stall=0
        last=$cur
      fi
    done
  ) &
  WATCHDOG_PID=$!
  python -u -m phase3_scan.v2.infer_shard_v2 --mgrs-list ../mgrs_todo.txt || true
  kill $WATCHDOG_PID 2>/dev/null || true
  wait $WATCHDOG_PID 2>/dev/null || true
  # Done if no shards remain
  REMAINING=$(python - <<'PY'
from pathlib import Path
import pandas as pd
scenes = pd.read_parquet("../data_us/phase3_scenes.parquet")
done = {p.stem for p in Path("../data_us/phase3_results_v2").glob("*.parquet")
        if not p.stem.endswith("_emb")}
print(len(set(scenes.mgrs_tile) - done))
PY
)
  echo "[bootstrap-v2] attempt $attempt finished, $REMAINING MGRS shards remaining"
  if [ "$REMAINING" = "0" ]; then
    echo "[bootstrap-v2] all shards complete"
    break
  fi
  sleep 5
done
cd ..

# Final sync to catch anything the bg loop hasn't picked up.
aws s3 sync data_us/phase3_results_v2/ "s3://${BUCKET}/phase3_results_v2/" --only-show-errors
echo "[bootstrap-v2] uploaded results to s3://${BUCKET}/phase3_results_v2/"

echo "[bootstrap-v2] scan complete, terminating instance via EXIT trap"
# trap on_exit fires automatically on exit
