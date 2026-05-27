#!/usr/bin/env bash
# Bootstrap a cheap CPU spot instance (Ubuntu 22.04) for Stage 2b v3 NLCD
# fractional-impervious scoring. No GPU, no torch. rasterio + scipy + mp.
#
# Reads s3://usgs-landcover/ (requester-pays) — instance role pays.
#
# Telemetry mirrors v3_pt3 — per-chunk stdout, JSONL stats, heartbeat,
# stall stack dump, CloudWatch alive metric.
#
# Safety: EXIT-trap terminate via EC2 API (per memory ec2-shutdown-verify).

set -euo pipefail

BUCKET="${BUCKET:?set BUCKET=industrials-scanner-us-west-2}"
WORK="${HOME}/v3pt4"
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
echo "[bootstrap-pt4] instance=$INSTANCE_ID type=$INSTANCE_TYPE region=$REGION"

LOG_S3="s3://${BUCKET}/v3-pt4-artifacts/logs/${INSTANCE_ID}.log"
STATS_S3="s3://${BUCKET}/v3-pt4-artifacts/logs/${INSTANCE_ID}.stats.jsonl"

if [ ! -x /usr/local/bin/aws ] || ! /usr/local/bin/aws --version 2>&1 | grep -q 'aws-cli/2'; then
  echo "[bootstrap-pt4] installing AWS CLI v2"
  sudo apt-get install -y -qq unzip
  (cd /tmp && curl -sS "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o awscliv2.zip \
     && unzip -q -o awscliv2.zip && sudo ./aws/install --update)
fi
hash -r
echo "[bootstrap-pt4] aws -> $(which aws) $(aws --version 2>&1)"

echo "[bootstrap-pt4] pulling bundle"
aws s3 sync "s3://${BUCKET}/v3-pt4-bundle/" ./bundle/ --only-show-errors

mkdir -p data_us/phase2/v3 sites_us
cp bundle/stage3_candidates_v3.parquet          data_us/phase2/v3/
cp bundle/stage2_candidate_polygons.parquet     data_us/phase2/v3/
cp -r bundle/code/sites_us/*                    sites_us/
cp bundle/.env                                  sites_us/.env || true

CHUNK_DIR="data_us/phase2/v3/stage2b_nlcd_chunks${STAGE2B_RUN_TAG:-}"
mkdir -p "$CHUNK_DIR"
echo "[bootstrap-pt4] pulling existing chunks (resume)"
aws s3 sync "s3://${BUCKET}/v3-pt4-artifacts/stage2b_nlcd_chunks${STAGE2B_RUN_TAG:-}/" \
  "$CHUNK_DIR/" --only-show-errors
RESUME_N=$(find "$CHUNK_DIR" -maxdepth 1 -name 'chunk_*.parquet' 2>/dev/null | wc -l)
echo "[bootstrap-pt4] resumed chunks: $RESUME_N"

[ -f sites_us/.env ] && set -a && . sites_us/.env && set +a

echo "[bootstrap-pt4] python deps"
if command -v conda >/dev/null && [ -f /opt/conda/etc/profile.d/conda.sh ]; then
  set +u; source /opt/conda/etc/profile.d/conda.sh; conda activate base; set -u
elif [ -f /opt/pytorch/bin/activate ]; then
  set +u; source /opt/pytorch/bin/activate; set -u
fi
pip install --quiet --upgrade pip 2>&1 | tail -5
pip install --quiet \
  "rasterio>=1.3.9" "boto3>=1.34.0" "pyproj>=3.6.0" \
  "pandas>=2.1.0" "pyarrow>=14.0.0" "shapely>=2.0.0" \
  "scipy>=1.11.0" "numpy>=1.26.0" \
  "psutil>=5.9.0" "python-dotenv>=1.0.0" 2>&1 | tail -5
echo "[bootstrap-pt4] pip done"

ulimit -n 16384

VCPU=$(nproc)
WORKERS_DEFAULT=$(( VCPU * 4 ))
[ "$WORKERS_DEFAULT" -gt 64 ] && WORKERS_DEFAULT=64
export STAGE2B_NUM_WORKERS="${STAGE2B_NUM_WORKERS:-$WORKERS_DEFAULT}"
export STAGE2B_CHUNK_SIZE="${STAGE2B_CHUNK_SIZE:-500}"
export STAGE2B_S3_BUCKET="$BUCKET"
export STAGE2B_INSTANCE_ID="$INSTANCE_ID"
echo "[bootstrap-pt4] vCPU=$VCPU  workers=$STAGE2B_NUM_WORKERS  chunk_size=$STAGE2B_CHUNK_SIZE"

(
  for i in $(seq 1 10); do
    sleep 30
    aws s3 cp "$HOME/v3pt4.out" "$LOG_S3" --only-show-errors 2>/dev/null || true
    [ -f "$CHUNK_DIR/_stats.jsonl" ] && \
      aws s3 cp "$CHUNK_DIR/_stats.jsonl" "$STATS_S3" --only-show-errors 2>/dev/null || true
  done
  while true; do
    sleep 60
    aws s3 cp "$HOME/v3pt4.out" "$LOG_S3" --only-show-errors 2>/dev/null || true
    [ -f "$CHUNK_DIR/_stats.jsonl" ] && \
      aws s3 cp "$CHUNK_DIR/_stats.jsonl" "$STATS_S3" --only-show-errors 2>/dev/null || true
    aws s3 sync "$CHUNK_DIR/" \
      "s3://${BUCKET}/v3-pt4-artifacts/stage2b_nlcd_chunks${STAGE2B_RUN_TAG:-}/" \
      --exclude "*" --include "chunk_*.parquet" --only-show-errors 2>/dev/null || true
  done
) &
SYNC_PID=$!
echo "[bootstrap-pt4] sync loop pid=$SYNC_PID"

(
  while true; do
    sleep 60
    aws cloudwatch put-metric-data \
      --region "$REGION" --namespace "stage2b_nlcd" \
      --metric-name "alive" --value 1 \
      --dimensions "InstanceId=$INSTANCE_ID" \
      --only-show-errors 2>/dev/null || true
  done
) &
CW_PID=$!

on_exit() {
  local code=$?
  echo "[bootstrap-pt4] EXIT code=$code; final sync"
  aws s3 cp "$HOME/v3pt4.out" "$LOG_S3" --only-show-errors 2>/dev/null || true
  aws s3 sync "$CHUNK_DIR/" \
    "s3://${BUCKET}/v3-pt4-artifacts/stage2b_nlcd_chunks${STAGE2B_RUN_TAG:-}/" \
    --only-show-errors 2>/dev/null || true
  kill $SYNC_PID $CW_PID 2>/dev/null || true
  echo "[bootstrap-pt4] terminating $INSTANCE_ID"
  aws ec2 terminate-instances --region "$REGION" --instance-ids "$INSTANCE_ID" 2>&1 || true
}
trap on_exit EXIT

cd sites_us
python3 -u phase2_classifier/v3_pt4/nlcd_impervious_scan.py 2>&1 | tee -a "$HOME/v3pt4.out"
