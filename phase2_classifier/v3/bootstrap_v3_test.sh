#!/usr/bin/env bash
# Bootstrap a small GPU box to score saved v3 probes against the per-building
# test set (~2K Overture buildings, disjoint from training).
#
# Instance: g4dn.2xlarge (8 vCPU, 32 GB, T4, 25 Gbps). Workload is brief —
# fetch ~2K NAIP crops, embed once with two encoders, score saved probes.
#
# Usage on EC2:
#   aws s3 cp s3://${BUCKET}/v3-test-bundle/bootstrap_v3_test.sh .
#   BUCKET=industrials-scanner-us-west-2 nohup bash bootstrap_v3_test.sh > v3_test.out 2>&1 &
set -euo pipefail

BUCKET="${BUCKET:?set BUCKET=industrials-scanner-us-west-2}"
WORK="${HOME}/v3test"
mkdir -p "$WORK"
cd "$WORK"

# IMDSv2 metadata
IMDS_TOKEN=$(curl -sS -X PUT "http://169.254.169.254/latest/api/token" \
  -H "X-aws-ec2-metadata-token-ttl-seconds: 60" --connect-timeout 3 --max-time 5)
INSTANCE_ID=$(curl -sS -H "X-aws-ec2-metadata-token: $IMDS_TOKEN" \
  http://169.254.169.254/latest/meta-data/instance-id --connect-timeout 3 --max-time 5)
REGION=$(curl -sS -H "X-aws-ec2-metadata-token: $IMDS_TOKEN" \
  http://169.254.169.254/latest/meta-data/placement/region --connect-timeout 3 --max-time 5)
INSTANCE_TYPE=$(curl -sS -H "X-aws-ec2-metadata-token: $IMDS_TOKEN" \
  http://169.254.169.254/latest/meta-data/instance-type --connect-timeout 3 --max-time 5)
echo "[bootstrap-v3-test] instance: $INSTANCE_ID, type: $INSTANCE_TYPE, region: $REGION"

# Bootstrap creds
[ -f "$HOME/.env" ] && set -a && . "$HOME/.env" && set +a

echo "[bootstrap-v3-test] pulling test bundle"
aws s3 sync "s3://${BUCKET}/v3-test-bundle/" ./bundle/ --only-show-errors

mkdir -p data_us/phase2/v3/probes sites_us
cp bundle/v3_test_set_manifest.parquet     data_us/phase2/
cp bundle/v3_test_set_scenes_index.parquet data_us/phase2/
cp -r bundle/code/sites_us/*               sites_us/
cp bundle/.env                             sites_us/.env || true

# Pull saved probes (small).
echo "[bootstrap-v3-test] pulling probes"
aws s3 sync "s3://${BUCKET}/v3-artifacts/v3/probes/" data_us/phase2/v3/probes/ --only-show-errors

set -a && . ./sites_us/.env && set +a

# Same GDAL knobs as training bootstrap.
export GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR
export GDAL_HTTP_MULTIPLEX=YES
export GDAL_HTTP_VERSION=2
export GDAL_HTTP_MAX_RETRY=5
export GDAL_HTTP_RETRY_DELAY=0.5
export GDAL_HTTP_TIMEOUT=30
export CPL_VSIL_CURL_USE_HEAD=NO
export CPL_VSIL_CURL_ALLOWED_EXTENSIONS=.tif
export GDAL_INGESTED_BYTES_AT_OPEN=32768
export VSI_CACHE=TRUE
export VSI_CACHE_SIZE=536870912
export CPL_VSIL_CURL_CHUNK_SIZE=1048576
export AWS_REQUEST_PAYER=requester

# Activate PyTorch env (DLAMI Nov-2025+ uses conda).
if [ -f /opt/pytorch/bin/activate ]; then
  set +u; source /opt/pytorch/bin/activate; set -u
elif [ -f /opt/conda/etc/profile.d/conda.sh ]; then
  set +u; source /opt/conda/etc/profile.d/conda.sh; conda activate pytorch; set -u
else
  echo "[bootstrap-v3-test] ERROR: no PyTorch env" >&2; exit 1
fi

# Same transformers pin as training.
pip install --quiet --upgrade \
  "transformers==4.56.0" "rasterio>=1.3.9" "boto3>=1.34.0" \
  "pyproj>=3.6.0" "pandas>=2.1.0" "pyarrow>=14.0.0" "Pillow>=10.0.0" \
  "scikit-learn>=1.3.0" "python-dotenv>=1.0.0" "huggingface_hub>=0.19.0"

if [ -n "${HF_TOKEN:-}" ]; then
  python - <<PY
from huggingface_hub import login; import os
login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
PY
fi

ulimit -n 8192

LOG_S3="s3://${BUCKET}/v3-artifacts/${INSTANCE_ID}.test.out"
on_exit() {
  echo "[bootstrap-v3-test] EXIT: sync + terminate"
  aws s3 cp "$HOME/v3_test.out" "$LOG_S3" --only-show-errors || true
  aws s3 sync data_us/phase2/v3/ "s3://${BUCKET}/v3-artifacts/v3/" --only-show-errors || true
  aws ec2 terminate-instances --region "$REGION" --instance-ids "$INSTANCE_ID" 2>&1 || true
}
trap on_exit EXIT

# Tune for in-region EC2 (faster than the local-Mac defaults baked into the script).
export V3_TEST_DEVICE=cuda
export V3_TEST_IO_WORKERS=128
export V3_TEST_PREP_WORKERS=8
export V3_TEST_BATCH=64

cd sites_us
python -u phase2_classifier/v3/v3_score_test.py 2>&1 | tee ../v3_test.log
cd ..
