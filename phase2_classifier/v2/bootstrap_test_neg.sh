#!/usr/bin/env bash
# Bootstrap a GPU box to grade the v2 5-encoder probes against the
# in-distribution OSM-building negatives test set (test_neg_v2_manifest.parquet).
# Mirrors bootstrap_v2.sh's structure but for the test-neg embedding +
# grading flow only — no training.
#
# Usage on EC2:
#   curl -O https://${BUCKET}.s3.us-west-2.amazonaws.com/test-neg-bundle/bootstrap_test_neg.sh
#   BUCKET=industrials-scanner-us-west-2 bash bootstrap_test_neg.sh
#
# Required instance: g4dn.2xlarge (T4, 32 GB RAM). Same constraint as v2 train
# because v2_train.MEMORY_BUDGET_BYTES caps at 16 GB and Prithvi-600m on T4
# needs comfortable headroom. Override (g5.2xlarge / g6.2xlarge) via
# ALLOW_INSTANCE_TYPE.
#
# At completion this script self-terminates the instance via the EC2 API,
# not just `shutdown -h` — that handles the case where instance-initiated
# shutdown behavior is "stop" (not "terminate"), which would otherwise leave
# the box charging EBS until manual termination.

set -euo pipefail

BUCKET="${BUCKET:?set BUCKET=industrials-scanner-us-west-2}"
WORK="${HOME}/test-neg"
mkdir -p "$WORK"
cd "$WORK"

# ---------- IMDSv2 instance-type guard --------------------------------------
echo "[bootstrap-test-neg] checking instance type via IMDSv2..."
IMDS_TOKEN=$(curl -sS -X PUT "http://169.254.169.254/latest/api/token" \
  -H "X-aws-ec2-metadata-token-ttl-seconds: 60" \
  --connect-timeout 3 --max-time 5)
INSTANCE_TYPE=$(curl -sS \
  -H "X-aws-ec2-metadata-token: $IMDS_TOKEN" \
  http://169.254.169.254/latest/meta-data/instance-type \
  --connect-timeout 3 --max-time 5)
INSTANCE_ID=$(curl -sS \
  -H "X-aws-ec2-metadata-token: $IMDS_TOKEN" \
  http://169.254.169.254/latest/meta-data/instance-id \
  --connect-timeout 3 --max-time 5)
REGION=$(curl -sS \
  -H "X-aws-ec2-metadata-token: $IMDS_TOKEN" \
  http://169.254.169.254/latest/meta-data/placement/region \
  --connect-timeout 3 --max-time 5)
echo "[bootstrap-test-neg] running on: $INSTANCE_TYPE ($INSTANCE_ID, $REGION)"

case "$INSTANCE_TYPE" in
  *.xlarge)
    echo "ERROR: '$INSTANCE_TYPE' has only 16 GB RAM and will OOM."
    exit 1
    ;;
esac

EXPECTED_TYPE="g4dn.2xlarge"
ALLOWED_FALLBACKS="g5.2xlarge g6.2xlarge"
if [ "$INSTANCE_TYPE" != "$EXPECTED_TYPE" ]; then
  override_ok=0
  if [ "${ALLOW_INSTANCE_TYPE:-}" = "$INSTANCE_TYPE" ]; then
    for ok_type in $ALLOWED_FALLBACKS; do
      [ "$INSTANCE_TYPE" = "$ok_type" ] && override_ok=1
    done
  fi
  if [ $override_ok -eq 0 ]; then
    echo "ERROR: '$INSTANCE_TYPE' not allowed. Need $EXPECTED_TYPE (or override with ALLOW_INSTANCE_TYPE=$INSTANCE_TYPE for $ALLOWED_FALLBACKS)."
    exit 1
  fi
fi
echo "[bootstrap-test-neg] instance type OK"

# Always-on terminate-on-exit safety net (covers script crashes).
on_exit() {
  echo "[bootstrap-test-neg] EXIT trap: terminating $INSTANCE_ID"
  # DLAMI ships AWS CLI v1; --no-cli-pager is v2-only, so omit it.
  aws ec2 terminate-instances --region "$REGION" \
    --instance-ids "$INSTANCE_ID" 2>&1 || true
}
trap on_exit EXIT

# ---------- Pull bundles ----------------------------------------------------
echo "[bootstrap-test-neg] pulling test-neg bundle from s3://${BUCKET}/test-neg-bundle"
aws s3 sync "s3://${BUCKET}/test-neg-bundle/" ./bundle/ --only-show-errors

mkdir -p data_us sites_us data_us/v2

# Manifest + code from test-neg bundle
cp bundle/test_neg_v2_manifest.parquet data_us/
cp -r bundle/code/sites_us/*           sites_us/
cp bundle/.env                         sites_us/.env || true

# Pull the v2 dataset manifest (load_strict_positives joins is_inferred from it)
echo "[bootstrap-test-neg] pulling v2_dataset_manifest.parquet from v2-bundle"
aws s3 cp "s3://${BUCKET}/v2-bundle/v2_dataset_manifest.parquet" \
  data_us/v2_dataset_manifest.parquet --only-show-errors

# Pull v2-artifacts: probes + prior-run embeddings (for strict-positives lookup)
echo "[bootstrap-test-neg] pulling v2-artifacts (probes + emb_idx + emb_*.npy)"
aws s3 sync "s3://${BUCKET}/v2-artifacts/v2/probes/"  data_us/v2/probes/  --only-show-errors
aws s3 cp   "s3://${BUCKET}/v2-artifacts/v2/v2_embeddings_index.parquet" \
  data_us/v2/v2_embeddings_index.parquet --only-show-errors
# emb_*.npy live at v2-artifacts/v2/ root, not under embeds/ — pull only those
for m in dino_sat493m dino_vitb resnet50 prithvi_300m prithvi_600m; do
  aws s3 cp "s3://${BUCKET}/v2-artifacts/v2/emb_${m}.npy" \
    "data_us/v2/embeds/emb_${m}.npy" --only-show-errors || echo "  skip emb_${m}.npy"
done

set -a
. ./sites_us/.env || true
set +a

source /opt/pytorch/bin/activate

pip install --quiet --upgrade \
  "transformers>=4.45.0" \
  "rasterio>=1.3.9" \
  "boto3>=1.34.0" \
  "mgrs>=1.4.6" \
  "pyproj>=3.6.0" \
  "pandas>=2.1.0" \
  "pyarrow>=14.0.0" \
  "Pillow>=10.0.0" \
  "pystac-client>=0.7.0" \
  "scikit-learn>=1.3.0" \
  "python-dotenv>=1.0.0" \
  "terratorch>=1.0.0"

# HF login for gated DINOv3 vit-b
if [ -n "${HF_TOKEN:-}" ]; then
  echo "[bootstrap-test-neg] hf login"
  python - <<PY
from huggingface_hub import login
import os
login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
PY
fi

# GPU sanity
python - <<'PY'
import torch
print("CUDA:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
PY

# ---------- Run the embed + grade pipeline ----------------------------------
cd sites_us

# Env vars to make embed_test_negatives.py run in EC2 mode:
#   - point at the data_us/v2/ paths we just populated
#   - use cuda
#   - skip local-Mac throttling overrides (use v2_train EC2 defaults)
#   - leave AWS auth to v2_train's default rasterio env (instance role works)
export TEST_NEG_PROBES_DIR="${WORK}/data_us/v2/probes"
export TEST_NEG_EXISTING_EMB_INDEX="${WORK}/data_us/v2/v2_embeddings_index.parquet"
export TEST_NEG_EXISTING_EMB_DIR="${WORK}/data_us/v2/embeds"
export TEST_NEG_DEVICE="cuda"
export TEST_NEG_AWS_ANON="NO"
export TEST_NEG_USE_V2_DEFAULTS="1"
export TEST_NEG_STAC_WORKERS="40"

python -u -m phase2_classifier.v2.embed_test_negatives 2>&1 \
  | tee ../test_neg_embed.log

cd ..

# ---------- Sync results + log back to S3 -----------------------------------
echo "[bootstrap-test-neg] syncing results to s3://${BUCKET}/test-neg-artifacts/"
aws s3 sync data_us/test_neg_v2/ "s3://${BUCKET}/test-neg-artifacts/test_neg_v2/" --only-show-errors
aws s3 cp data_us/test_neg_v2_scenes_index.parquet \
  "s3://${BUCKET}/test-neg-artifacts/test_neg_v2_scenes_index.parquet" --only-show-errors || true
aws s3 cp test_neg_embed.log \
  "s3://${BUCKET}/test-neg-artifacts/test_neg_embed.log" --only-show-errors

echo "[bootstrap-test-neg] done — terminating self"
# trap on_exit runs aws ec2 terminate-instances
