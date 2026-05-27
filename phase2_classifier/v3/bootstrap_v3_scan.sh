#!/usr/bin/env bash
# Bootstrap a g4dn.8xlarge (or g6.8xlarge) for the CONUS scan inference pass.
#
# Reuses v3 training tuning (open-once-per-tile fetch, V3_IO_WORKERS=512,
# 5-min periodic sync, EXIT-trap terminate via EC2 API).
#
# Env knobs (set in launch_v3_scan.sh user-data):
#   V3_SCAN_SAMPLE_N=10000   — validation runs; omit for full scan
set -euo pipefail

BUCKET="${BUCKET:?set BUCKET=industrials-scanner-us-west-2}"
WORK="${HOME}/v3scan"
mkdir -p "$WORK"
cd "$WORK"

IMDS_TOKEN=$(curl -sS -X PUT "http://169.254.169.254/latest/api/token" \
  -H "X-aws-ec2-metadata-token-ttl-seconds: 60" --connect-timeout 3 --max-time 5)
INSTANCE_ID=$(curl -sS -H "X-aws-ec2-metadata-token: $IMDS_TOKEN" \
  http://169.254.169.254/latest/meta-data/instance-id --connect-timeout 3 --max-time 5)
REGION=$(curl -sS -H "X-aws-ec2-metadata-token: $IMDS_TOKEN" \
  http://169.254.169.254/latest/meta-data/placement/region --connect-timeout 3 --max-time 5)
INSTANCE_TYPE=$(curl -sS -H "X-aws-ec2-metadata-token: $IMDS_TOKEN" \
  http://169.254.169.254/latest/meta-data/instance-type --connect-timeout 3 --max-time 5)
echo "[bootstrap-v3-scan] instance: $INSTANCE_ID, type: $INSTANCE_TYPE, region: $REGION"

case "$INSTANCE_TYPE" in
  *.xlarge)
    echo "ERROR: '$INSTANCE_TYPE' has 16 GB RAM, insufficient for scan."; exit 1 ;;
esac

[ -f "$HOME/.env" ] && set -a && . "$HOME/.env" && set +a

echo "[bootstrap-v3-scan] pulling bundle"
aws s3 sync "s3://${BUCKET}/v3-scan-bundle/" ./bundle/ --only-show-errors

mkdir -p data_us/phase2/v3/probes sites_us
cp bundle/v3_scan_manifest.parquet       data_us/phase2/
cp bundle/v3_scan_scenes_index.parquet   data_us/phase2/
cp -r bundle/code/sites_us/*             sites_us/
cp bundle/.env                           sites_us/.env || true

echo "[bootstrap-v3-scan] pulling saved probes"
aws s3 sync "s3://${BUCKET}/v3-artifacts/v3/probes/" data_us/phase2/v3/probes/ --only-show-errors

# Pull existing per-chunk scores so we resume after spot reclaim instead of redoing.
echo "[bootstrap-v3-scan] pulling existing chunk scores (resume)"
mkdir -p data_us/phase2/v3/scan_chunks/_scores
aws s3 sync "s3://${BUCKET}/v3-artifacts/v3/scan_chunks/_scores/" \
  data_us/phase2/v3/scan_chunks/_scores/ --only-show-errors
echo "[bootstrap-v3-scan] resumed chunk files: $(ls data_us/phase2/v3/scan_chunks/_scores/ 2>/dev/null | wc -l)"

set -a && . ./sites_us/.env && set +a

# Same GDAL knobs as training bootstrap.
export GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR
export GDAL_HTTP_MULTIPLEX=YES
export GDAL_HTTP_VERSION=2
export GDAL_HTTP_MAX_RETRY=5
export GDAL_HTTP_RETRY_DELAY=0.5
export GDAL_HTTP_TIMEOUT=20
export CPL_VSIL_CURL_USE_HEAD=NO
export CPL_VSIL_CURL_ALLOWED_EXTENSIONS=.tif
export GDAL_INGESTED_BYTES_AT_OPEN=524288
export VSI_CACHE=TRUE
export VSI_CACHE_SIZE=2147483648
export CPL_VSIL_CURL_CHUNK_SIZE=1048576
export AWS_REQUEST_PAYER=requester

if [ -f /opt/pytorch/bin/activate ]; then
  set +u; source /opt/pytorch/bin/activate; set -u
elif [ -f /opt/conda/etc/profile.d/conda.sh ]; then
  set +u; source /opt/conda/etc/profile.d/conda.sh; conda activate pytorch; set -u
else
  echo "[bootstrap-v3-scan] ERROR: no PyTorch env" >&2; exit 1
fi

# pip install fixed (2026-05-25): the prior `pip install --upgrade <many>`
# hung in dependency resolver on multiple full-scan launches (smoke5, full2,
# full3) — pid alive, 0% CPU, 0 network for 30+ min. `--upgrade` forces pip
# to re-evaluate every dep in the DLAMI's already-populated env, causing
# infinite backtracking. Fix: install ONLY transformers with an exact pin
# (the one version-sensitive package), then everything else without --upgrade
# so pip skips already-satisfied. Wrap each in `timeout` so a hang aborts
# instead of wedging the instance silently.
timeout 600 pip install --quiet "transformers==4.56.0" || {
  echo "[bootstrap-v3-scan] ERROR: transformers pin install timed out/failed" >&2
  exit 1
}
timeout 600 pip install --quiet \
  "rasterio>=1.3.9" "boto3>=1.34.0" "pyproj>=3.6.0" \
  "pandas>=2.1.0" "pyarrow>=14.0.0" "Pillow>=10.0.0" \
  "scikit-learn>=1.3.0" "python-dotenv>=1.0.0" \
  "huggingface_hub>=0.19.0" "psutil>=5.9.0" || {
  echo "[bootstrap-v3-scan] ERROR: secondary pip install timed out/failed" >&2
  exit 1
}
echo "[bootstrap-v3-scan] pip install complete"

if [ -n "${HF_TOKEN:-}" ]; then
  python - <<PY
from huggingface_hub import login; import os
login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
PY
fi

ulimit -n 16384

LOG_S3="s3://${BUCKET}/v3-artifacts/${INSTANCE_ID}.scan.out"

# Eager early log-sync: upload current log every 30s for the first 5 min so
# we can diagnose boot-time hangs (smoke5/full2 hung for 30+ min with no
# log ever uploaded by the 300s periodic sync). After the burst window we
# fall back to the 300s cadence to reduce S3 noise.
(
  for i in $(seq 1 10); do
    sleep 30
    aws s3 cp "$HOME/v3_scan.out" "$LOG_S3" --only-show-errors || true
    [ -f /tmp/v3_stacks.txt ] && \
      aws s3 cp /tmp/v3_stacks.txt "s3://${BUCKET}/v3-artifacts/${INSTANCE_ID}.stacks.txt" --only-show-errors || true
  done
  while true; do
    sleep 60   # tightened from 300s so stack dumps surface faster when chunks stall
    aws s3 sync data_us/phase2/v3/ "s3://${BUCKET}/v3-artifacts/v3/" \
        --exclude "embed_chunks/*" --only-show-errors || true
    aws s3 cp "$HOME/v3_scan.out" "$LOG_S3" --only-show-errors || true
    [ -f /tmp/v3_stacks.txt ] && \
      aws s3 cp /tmp/v3_stacks.txt "s3://${BUCKET}/v3-artifacts/${INSTANCE_ID}.stacks.txt" --only-show-errors || true
  done
) &
SYNC_PID=$!
echo "[bootstrap-v3-scan] eager log-sync started (30s × 10, then 300s)"

on_exit() {
  echo "[bootstrap-v3-scan] EXIT: final sync"
  aws s3 cp "$HOME/v3_scan.out" "$LOG_S3" --only-show-errors || true
  aws s3 sync data_us/phase2/v3/ "s3://${BUCKET}/v3-artifacts/v3/" --only-show-errors || true
  kill $SYNC_PID 2>/dev/null || true
  echo "[bootstrap-v3-scan] terminating $INSTANCE_ID"
  aws ec2 terminate-instances --region "$REGION" --instance-ids "$INSTANCE_ID" 2>&1 || true
}
trap on_exit EXIT

# Forward V3_SCAN_* env vars if the launch user-data set them.
export V3_SCAN_SAMPLE_N="${V3_SCAN_SAMPLE_N:-0}"
export V3_SCAN_SEED="${V3_SCAN_SEED:-7}"
export V3_SCAN_MAX_CHUNKS="${V3_SCAN_MAX_CHUNKS:-0}"
export V3_SCAN_ONLY_CHUNKS="${V3_SCAN_ONLY_CHUNKS:-}"
export V3_BATCH_SIZE="${V3_BATCH_SIZE:-64}"
export V3_PREFETCH_FACTOR="${V3_PREFETCH_FACTOR:-4}"

# DataLoader num_workers — each worker is a separate process with its own GDAL
# state, no GIL/thread-local race. Size matches vCPU budget (g4dn.4xlarge=16,
# g4dn.8xlarge=32). Per-worker memory ~500MB (rasterio + VSI cache 1GB).
case "$INSTANCE_TYPE" in
  g4dn.8xlarge|g6.8xlarge) export V3_NUM_WORKERS="${V3_NUM_WORKERS:-32}" ;;
  *)                       export V3_NUM_WORKERS="${V3_NUM_WORKERS:-16}" ;;
esac
echo "[bootstrap-v3-scan] V3_NUM_WORKERS=$V3_NUM_WORKERS V3_BATCH_SIZE=$V3_BATCH_SIZE V3_PREFETCH_FACTOR=$V3_PREFETCH_FACTOR"

cd sites_us
python -u phase2_classifier/v3/v3_scan_infer.py 2>&1 | tee ../v3_scan.log
cd ..
