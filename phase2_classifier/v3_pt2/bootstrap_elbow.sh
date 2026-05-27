#!/usr/bin/env bash
# Bootstrap for elbow_scan.py — L7 2008 SR 6-band signature over full cohort.
# Sibling of bootstrap.sh; writes to elbow_chunks{RUN_TAG} not change_chunks_v3.

set -euo pipefail

BUCKET="${BUCKET:?set BUCKET=industrials-scanner-us-west-2}"
WORK="${HOME}/v3pt2"
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
echo "[bootstrap-elbow] instance=$INSTANCE_ID type=$INSTANCE_TYPE region=$REGION"

LOG_S3="s3://${BUCKET}/v3-pt2-artifacts/logs/${INSTANCE_ID}.log"
STATS_S3="s3://${BUCKET}/v3-pt2-artifacts/logs/${INSTANCE_ID}.stats.jsonl"

if [ ! -x /usr/local/bin/aws ] || ! /usr/local/bin/aws --version 2>&1 | grep -q 'aws-cli/2'; then
  echo "[bootstrap-elbow] installing AWS CLI v2"
  sudo apt-get install -y -qq unzip
  (cd /tmp && curl -sS "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o awscliv2.zip \
     && unzip -q -o awscliv2.zip && sudo ./aws/install --update)
fi
hash -r
echo "[bootstrap-elbow] aws -> $(which aws) $(aws --version 2>&1)"

echo "[bootstrap-elbow] pulling bundle"
aws s3 sync "s3://${BUCKET}/v3-pt2-bundle/" ./bundle/ --only-show-errors

mkdir -p data_us/phase2/v3 sites_us
cp bundle/stage2_candidates.parquet     data_us/phase2/v3/
cp bundle/landsat_scenes_index.parquet  data_us/phase2/v3/
[ -f bundle/stage2_candidate_polygons.parquet ] && \
  cp bundle/stage2_candidate_polygons.parquet  data_us/phase2/v3/ || true
cp -r bundle/code/sites_us/*            sites_us/
cp bundle/.env                          sites_us/.env || true

mkdir -p data_us/phase2/v3/elbow_chunks${STAGE2B_RUN_TAG:-}
echo "[bootstrap-elbow] pulling existing chunks (resume)"
aws s3 sync "s3://${BUCKET}/v3-pt2-artifacts/elbow_chunks${STAGE2B_RUN_TAG:-}/" \
  data_us/phase2/v3/elbow_chunks${STAGE2B_RUN_TAG:-}/ --only-show-errors
RESUME_N=$(find data_us/phase2/v3/elbow_chunks${STAGE2B_RUN_TAG:-} -maxdepth 1 -name 'chunk_*.parquet' 2>/dev/null | wc -l)
echo "[bootstrap-elbow] resumed chunks: $RESUME_N"

[ -f sites_us/.env ] && set -a && . sites_us/.env && set +a

echo "[bootstrap-elbow] python deps"
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
echo "[bootstrap-elbow] pip done"

ulimit -n 16384

VCPU=$(nproc)
WORKERS_DEFAULT=$(( VCPU * 4 ))
[ "$WORKERS_DEFAULT" -gt 64 ] && WORKERS_DEFAULT=64
export STAGE2B_NUM_WORKERS="${STAGE2B_NUM_WORKERS:-$WORKERS_DEFAULT}"
export STAGE2B_CHUNK_SIZE="${STAGE2B_CHUNK_SIZE:-200}"
export STAGE2B_S3_BUCKET="$BUCKET"
export STAGE2B_INSTANCE_ID="$INSTANCE_ID"
echo "[bootstrap-elbow] vCPU=$VCPU  workers=$STAGE2B_NUM_WORKERS  chunk_size=$STAGE2B_CHUNK_SIZE"

(
  for i in $(seq 1 10); do
    sleep 30
    aws s3 cp "$HOME/v3pt2.out" "$LOG_S3" --only-show-errors 2>/dev/null || true
    [ -f data_us/phase2/v3/elbow_chunks${STAGE2B_RUN_TAG:-}/_stats.jsonl ] && \
      aws s3 cp data_us/phase2/v3/elbow_chunks${STAGE2B_RUN_TAG:-}/_stats.jsonl "$STATS_S3" \
        --only-show-errors 2>/dev/null || true
  done
  while true; do
    sleep 60
    aws s3 cp "$HOME/v3pt2.out" "$LOG_S3" --only-show-errors 2>/dev/null || true
    [ -f data_us/phase2/v3/elbow_chunks${STAGE2B_RUN_TAG:-}/_stats.jsonl ] && \
      aws s3 cp data_us/phase2/v3/elbow_chunks${STAGE2B_RUN_TAG:-}/_stats.jsonl "$STATS_S3" \
        --only-show-errors 2>/dev/null || true
    aws s3 sync data_us/phase2/v3/elbow_chunks${STAGE2B_RUN_TAG:-}/ \
      "s3://${BUCKET}/v3-pt2-artifacts/elbow_chunks${STAGE2B_RUN_TAG:-}/" \
      --exclude "*" --include "chunk_*.parquet" --only-show-errors 2>/dev/null || true
  done
) &
SYNC_PID=$!
echo "[bootstrap-elbow] sync loop pid=$SYNC_PID"

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

on_exit() {
  local code=$?
  echo "[bootstrap-elbow] EXIT code=$code; final sync"
  aws s3 cp "$HOME/v3pt2.out" "$LOG_S3" --only-show-errors 2>/dev/null || true
  aws s3 sync data_us/phase2/v3/elbow_chunks${STAGE2B_RUN_TAG:-}/ \
    "s3://${BUCKET}/v3-pt2-artifacts/elbow_chunks${STAGE2B_RUN_TAG:-}/" \
    --only-show-errors 2>/dev/null || true
  kill $SYNC_PID $CW_PID 2>/dev/null || true
  echo "[bootstrap-elbow] terminating $INSTANCE_ID"
  aws ec2 terminate-instances --region "$REGION" --instance-ids "$INSTANCE_ID" 2>&1 || true
}
trap on_exit EXIT

cd sites_us
python3 -u phase2_classifier/v3_pt2/elbow_scan.py 2>&1 | tee -a "$HOME/v3pt2.out"
