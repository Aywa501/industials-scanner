#!/usr/bin/env bash
# Bootstrap a tiny CPU spot to pull Overture building polygons for v3 candidates.
# In-region (us-west-2): zero egress to overturemaps S3 and to results bucket.
#
# Safety: EXIT-trap terminate via EC2 API (per memory ec2-shutdown-verify).

set -euo pipefail

BUCKET="${BUCKET:?set BUCKET=industrials-scanner-us-west-2}"
WORK="${HOME}/poly"
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
echo "[bootstrap-poly] instance=$INSTANCE_ID type=$INSTANCE_TYPE region=$REGION"

LOG_S3="s3://${BUCKET}/v3-pt2-artifacts/logs/${INSTANCE_ID}.poly.log"

# AWS CLI v2 (apt v1 conflicts with pip's botocore). ------------------------ #
if [ ! -x /usr/local/bin/aws ] || ! /usr/local/bin/aws --version 2>&1 | grep -q 'aws-cli/2'; then
  echo "[bootstrap-poly] installing AWS CLI v2"
  sudo apt-get install -y -qq unzip
  (cd /tmp && curl -sS "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o awscliv2.zip \
     && unzip -q -o awscliv2.zip && sudo ./aws/install --update)
fi
hash -r
echo "[bootstrap-poly] aws -> $(which aws) $(aws --version 2>&1)"

# Pull the script + candidates from the existing v3-pt2 bundle. ------------- #
echo "[bootstrap-poly] pulling bundle"
aws s3 sync "s3://${BUCKET}/v3-pt2-bundle/" ./bundle/ --only-show-errors
mkdir -p data_us/phase2/v3 sites_us
cp bundle/stage2_candidates.parquet  data_us/phase2/v3/
cp -r bundle/code/sites_us/*         sites_us/

# Python deps. -------------------------------------------------------------- #
echo "[bootstrap-poly] python deps"
pip3 install --quiet --upgrade pip 2>&1 | tail -3
pip3 install --quiet "duckdb>=1.5.0" "pandas>=2.1.0" "pyarrow>=14.0.0" 2>&1 | tail -3
echo "[bootstrap-poly] pip done"

# Telemetry sync loop (30s log tail). --------------------------------------- #
(
  while true; do
    sleep 30
    aws s3 cp "$HOME/poly.out" "$LOG_S3" --only-show-errors 2>/dev/null || true
  done
) &
SYNC_PID=$!

# EXIT trap: final sync + terminate. ---------------------------------------- #
on_exit() {
  local code=$?
  echo "[bootstrap-poly] EXIT code=$code; final sync"
  aws s3 cp "$HOME/poly.out" "$LOG_S3" --only-show-errors 2>/dev/null || true
  kill $SYNC_PID 2>/dev/null || true
  echo "[bootstrap-poly] terminating $INSTANCE_ID"
  aws ec2 terminate-instances --region "$REGION" --instance-ids "$INSTANCE_ID" 2>&1 || true
}
trap on_exit EXIT

# Run. Writes per-batch parquet to S3 for resumability. -------------------- #
export OUT_S3="s3://${BUCKET}/v3-pt2-artifacts/stage2_candidate_polygons.parquet"
export BATCHES_S3="s3://${BUCKET}/v3-pt2-artifacts/poly_batches"
export CANDS_PATH="$WORK/data_us/phase2/v3/stage2_candidates.parquet"
export OUT_PATH="$WORK/data_us/phase2/v3/stage2_candidate_polygons.parquet"
export OUT_DIR="$WORK/data_us/phase2/v3/poly_batches"
export BATCH_SIZE="${BATCH_SIZE:-400000}"
export DUCKDB_MEM="${DUCKDB_MEM:-48GB}"

cd sites_us
python3 -u phase2_classifier/v3_pt2/fetch_candidate_polygons.py 2>&1 | tee -a "$HOME/poly.out"
