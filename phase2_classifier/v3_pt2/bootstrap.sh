#!/usr/bin/env bash
# Bootstrap a cheap CPU spot instance (Ubuntu 22.04) for Stage 2b change scoring.
# No GPU, no torch/transformers. Pure rasterio + scipy + multiprocessing.
#
# Telemetry:
#   - Per-chunk stdout line tailed to S3 every 30s via background loop.
#   - Per-chunk JSONL stats in change_chunks/_stats.jsonl synced periodically.
#   - Heartbeat JSON synced every 30s.
#   - Stuck-detection in change_scan.py dumps thread stacks to S3.
#
# Safety: EXIT-trap terminate via EC2 API (per memory ec2-shutdown-verify).

set -euo pipefail

BUCKET="${BUCKET:?set BUCKET=industrials-scanner-us-west-2}"
WORK="${HOME}/v3pt2"
mkdir -p "$WORK"
cd "$WORK"

# IMDSv2 identity ----------------------------------------------------------- #
IMDS_TOKEN=$(curl -sS -X PUT "http://169.254.169.254/latest/api/token" \
  -H "X-aws-ec2-metadata-token-ttl-seconds: 60" --connect-timeout 3 --max-time 5)
INSTANCE_ID=$(curl -sS -H "X-aws-ec2-metadata-token: $IMDS_TOKEN" \
  http://169.254.169.254/latest/meta-data/instance-id --connect-timeout 3 --max-time 5)
REGION=$(curl -sS -H "X-aws-ec2-metadata-token: $IMDS_TOKEN" \
  http://169.254.169.254/latest/meta-data/placement/region --connect-timeout 3 --max-time 5)
INSTANCE_TYPE=$(curl -sS -H "X-aws-ec2-metadata-token: $IMDS_TOKEN" \
  http://169.254.169.254/latest/meta-data/instance-type --connect-timeout 3 --max-time 5)
echo "[bootstrap-pt2] instance=$INSTANCE_ID type=$INSTANCE_TYPE region=$REGION"

LOG_S3="s3://${BUCKET}/v3-pt2-artifacts/logs/${INSTANCE_ID}.log"
STATS_S3="s3://${BUCKET}/v3-pt2-artifacts/logs/${INSTANCE_ID}.stats.jsonl"

# Install AWS CLI v2 from Amazon (apt's awscli v1 conflicts with pip's boto3
# botocore — silently breaks every `aws` call from sync loop). v2 is a single
# bundled binary at /usr/local/bin/aws and shadows /usr/bin/aws via PATH.
if [ ! -x /usr/local/bin/aws ] || ! /usr/local/bin/aws --version 2>&1 | grep -q 'aws-cli/2'; then
  echo "[bootstrap-pt2] installing AWS CLI v2"
  sudo apt-get install -y -qq unzip
  (cd /tmp && curl -sS "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o awscliv2.zip \
     && unzip -q -o awscliv2.zip && sudo ./aws/install --update)
fi
hash -r
echo "[bootstrap-pt2] aws -> $(which aws) $(aws --version 2>&1)"

# Pull bundle. -------------------------------------------------------------- #
echo "[bootstrap-pt2] pulling bundle"
aws s3 sync "s3://${BUCKET}/v3-pt2-bundle/" ./bundle/ --only-show-errors

mkdir -p data_us/phase2/v3 sites_us
cp bundle/stage2_candidates.parquet     data_us/phase2/v3/
cp bundle/landsat_scenes_index.parquet  data_us/phase2/v3/
[ -f bundle/stage2_candidate_polygons.parquet ] && \
  cp bundle/stage2_candidate_polygons.parquet  data_us/phase2/v3/ || true
cp -r bundle/code/sites_us/*            sites_us/
cp bundle/.env                          sites_us/.env || true

# Resume support: pull existing chunks. ------------------------------------- #
mkdir -p data_us/phase2/v3/stage2b_change_chunks_v3${STAGE2B_RUN_TAG:-}
echo "[bootstrap-pt2] pulling existing chunks (resume)"
aws s3 sync "s3://${BUCKET}/v3-pt2-artifacts/change_chunks_v3${STAGE2B_RUN_TAG:-}/" \
  data_us/phase2/v3/stage2b_change_chunks_v3${STAGE2B_RUN_TAG:-}/ --only-show-errors
RESUME_N=$(find data_us/phase2/v3/stage2b_change_chunks_v3${STAGE2B_RUN_TAG:-} -maxdepth 1 -name 'chunk_*.parquet' 2>/dev/null | wc -l)
echo "[bootstrap-pt2] resumed chunks: $RESUME_N"

# Env. ---------------------------------------------------------------------- #
[ -f sites_us/.env ] && set -a && . sites_us/.env && set +a

# Python deps. -------------------------------------------------------------- #
echo "[bootstrap-pt2] python deps"
if command -v conda >/dev/null && [ -f /opt/conda/etc/profile.d/conda.sh ]; then
  set +u; source /opt/conda/etc/profile.d/conda.sh; conda activate base; set -u
elif [ -f /opt/pytorch/bin/activate ]; then
  set +u; source /opt/pytorch/bin/activate; set -u
fi
pip install --quiet --upgrade pip 2>&1 | tail -5
pip install --quiet \
  "rasterio>=1.3.9" "boto3>=1.34.0" "pyproj>=3.6.0" \
  "pandas>=2.1.0" "pyarrow>=14.0.0" "shapely>=2.0.0" \
  "scipy>=1.11.0" "scikit-learn>=1.3.0" "numpy>=1.26.0" \
  "psutil>=5.9.0" "python-dotenv>=1.0.0" 2>&1 | tail -5
echo "[bootstrap-pt2] pip done"

ulimit -n 16384

# vCPU count -> worker count: heavily oversubscribe (pure I/O wait). -------- #
VCPU=$(nproc)
WORKERS_DEFAULT=$(( VCPU * 4 ))
[ "$WORKERS_DEFAULT" -gt 64 ] && WORKERS_DEFAULT=64
export STAGE2B_NUM_WORKERS="${STAGE2B_NUM_WORKERS:-$WORKERS_DEFAULT}"
export STAGE2B_CHUNK_SIZE="${STAGE2B_CHUNK_SIZE:-200}"
export STAGE2B_S3_BUCKET="$BUCKET"
export STAGE2B_INSTANCE_ID="$INSTANCE_ID"
echo "[bootstrap-pt2] vCPU=$VCPU  workers=$STAGE2B_NUM_WORKERS  chunk_size=$STAGE2B_CHUNK_SIZE"

# Telemetry sync loop. ------------------------------------------------------ #
(
  # First-5-min burst: sync every 30s for fast feedback during boot.
  for i in $(seq 1 10); do
    sleep 30
    aws s3 cp "$HOME/v3pt2.out" "$LOG_S3" --only-show-errors 2>/dev/null || true
    [ -f data_us/phase2/v3/stage2b_change_chunks_v3${STAGE2B_RUN_TAG:-}/_stats.jsonl ] && \
      aws s3 cp data_us/phase2/v3/stage2b_change_chunks_v3${STAGE2B_RUN_TAG:-}/_stats.jsonl "$STATS_S3" \
        --only-show-errors 2>/dev/null || true
  done
  # Steady state: 60s.
  while true; do
    sleep 60
    aws s3 cp "$HOME/v3pt2.out" "$LOG_S3" --only-show-errors 2>/dev/null || true
    [ -f data_us/phase2/v3/stage2b_change_chunks_v3${STAGE2B_RUN_TAG:-}/_stats.jsonl ] && \
      aws s3 cp data_us/phase2/v3/stage2b_change_chunks_v3${STAGE2B_RUN_TAG:-}/_stats.jsonl "$STATS_S3" \
        --only-show-errors 2>/dev/null || true
    aws s3 sync data_us/phase2/v3/stage2b_change_chunks_v3${STAGE2B_RUN_TAG:-}/ \
      "s3://${BUCKET}/v3-pt2-artifacts/change_chunks_v3${STAGE2B_RUN_TAG:-}/" \
      --exclude "*" --include "chunk_*.parquet" --only-show-errors 2>/dev/null || true
  done
) &
SYNC_PID=$!
echo "[bootstrap-pt2] sync loop pid=$SYNC_PID"

# CloudWatch heartbeat (per memory monitor-heartbeat).
(
  while true; do
    sleep 60
    aws cloudwatch put-metric-data \
      --region "$REGION" --namespace "stage2b" \
      --metric-name "alive" --value 1 \
      --dimensions "InstanceId=$INSTANCE_ID" \
      --only-show-errors 2>/dev/null || true
  done
) &
CW_PID=$!

# EXIT trap: final sync + terminate. --------------------------------------- #
on_exit() {
  local code=$?
  echo "[bootstrap-pt2] EXIT code=$code; final sync"
  aws s3 cp "$HOME/v3pt2.out" "$LOG_S3" --only-show-errors 2>/dev/null || true
  aws s3 sync data_us/phase2/v3/stage2b_change_chunks_v3${STAGE2B_RUN_TAG:-}/ \
    "s3://${BUCKET}/v3-pt2-artifacts/change_chunks_v3${STAGE2B_RUN_TAG:-}/" \
    --only-show-errors 2>/dev/null || true
  kill $SYNC_PID $CW_PID 2>/dev/null || true
  echo "[bootstrap-pt2] terminating $INSTANCE_ID"
  aws ec2 terminate-instances --region "$REGION" --instance-ids "$INSTANCE_ID" 2>&1 || true
}
trap on_exit EXIT

# Run. --------------------------------------------------------------------- #
cd sites_us
python3 -u phase2_classifier/v3_pt2/change_scan.py 2>&1 | tee -a "$HOME/v3pt2.out"
