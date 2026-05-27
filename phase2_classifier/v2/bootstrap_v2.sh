#!/usr/bin/env bash
# Bootstrap a GPU box in us-west-2 to run the v2 binary detector
# (5-encoder embed pass + per-encoder linear probe training).
# Required instance: g4dn.2xlarge (T4, 32 GB RAM). Script aborts otherwise.
#
# Why g4dn.2xlarge specifically:
#   - Bandwidth-bound, not GPU-bound — confirmed in prior round on g6.2xlarge.
#     T4 matches L4 wall-clock at ~30% lower spot cost.
#   - 32 GB RAM matches v2_train.py memory budget (peak ~20 GB).
#   - g4dn.xlarge (16 GB RAM) is too small and would OOM.
# Override (g5.2xlarge / g6.2xlarge fallback): set ALLOW_INSTANCE_TYPE=<type>.
#
# Usage on EC2:
#   curl -O https://${BUCKET}.s3.us-west-2.amazonaws.com/v2-bundle/bootstrap_v2.sh
#   BUCKET=industrials-scanner-us-west-2 bash bootstrap_v2.sh

set -euo pipefail

BUCKET="${BUCKET:?set BUCKET=industrials-scanner-us-west-2}"
WORK="${HOME}/v2train"
mkdir -p "$WORK"
cd "$WORK"

# ---------- Instance-type guard (IMDSv2) -------------------------------------
echo "[bootstrap-v2] checking instance type via IMDSv2..."
IMDS_TOKEN=$(curl -sS -X PUT "http://169.254.169.254/latest/api/token" \
  -H "X-aws-ec2-metadata-token-ttl-seconds: 60" \
  --connect-timeout 3 --max-time 5)
INSTANCE_TYPE=$(curl -sS \
  -H "X-aws-ec2-metadata-token: $IMDS_TOKEN" \
  http://169.254.169.254/latest/meta-data/instance-type \
  --connect-timeout 3 --max-time 5)
echo "[bootstrap-v2] running on: $INSTANCE_TYPE"

# Hard block: any *.xlarge variant has only 16 GB RAM and will OOM.
# This guard is unconditional — ALLOW_INSTANCE_TYPE cannot override it.
case "$INSTANCE_TYPE" in
  *.xlarge)
    echo "ERROR: '$INSTANCE_TYPE' has only 16 GB RAM."
    echo "  v2_train.py's MEMORY_BUDGET_BYTES=16 GB will OOM at peak."
    echo "  Required: g4dn.2xlarge (32 GB RAM)."
    echo "  This block cannot be overridden via ALLOW_INSTANCE_TYPE."
    exit 1
    ;;
esac

# Soft block: must be g4dn.2xlarge unless ALLOW_INSTANCE_TYPE matches.
EXPECTED_TYPE="g4dn.2xlarge"
ALLOWED_FALLBACKS="g5.2xlarge g6.2xlarge"
if [ "$INSTANCE_TYPE" != "$EXPECTED_TYPE" ]; then
  override_ok=0
  if [ "${ALLOW_INSTANCE_TYPE:-}" = "$INSTANCE_TYPE" ]; then
    for ok_type in $ALLOWED_FALLBACKS; do
      if [ "$INSTANCE_TYPE" = "$ok_type" ]; then
        override_ok=1
        break
      fi
    done
  fi
  if [ $override_ok -eq 0 ]; then
    echo "ERROR: instance type '$INSTANCE_TYPE' is not allowed."
    echo "  Required: '$EXPECTED_TYPE'."
    echo "  Permitted fallbacks (with ALLOW_INSTANCE_TYPE): $ALLOWED_FALLBACKS"
    echo "  Re-launch with: ALLOW_INSTANCE_TYPE=$INSTANCE_TYPE bash bootstrap_v2.sh"
    exit 1
  fi
  echo "[bootstrap-v2] using fallback instance type (override applied)"
fi
echo "[bootstrap-v2] instance type OK"

if [ -f "$HOME/.env" ]; then
  set -a; . "$HOME/.env"; set +a
fi

echo "[bootstrap-v2] pulling bundle from s3://${BUCKET}/v2-bundle"
aws s3 sync "s3://${BUCKET}/v2-bundle/" ./bundle/ --only-show-errors

mkdir -p data_us sites_us
cp bundle/v2_dataset_manifest.parquet  data_us/
cp bundle/v2_scenes_index.parquet      data_us/
cp -r bundle/code/sites_us/*           sites_us/
cp bundle/.env                         sites_us/.env || true

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

# GPU sanity
python - <<'PY'
import torch
print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
PY

# Background sync of v2/ artifacts during run (spot-interrupt safety).
# Syncs everything under data_us/phase2/v2/ except the per-group embed_chunks resume
# cache (large + only useful for in-place resume on the same box).
(
  while true; do
    sleep 300
    aws s3 sync data_us/phase2/v2/ "s3://${BUCKET}/v2-artifacts/v2/" \
        --exclude "embed_chunks/*" \
        --only-show-errors || true
  done
) &
SYNC_PID=$!
trap 'kill $SYNC_PID 2>/dev/null || true' EXIT

cd sites_us

# Fill scenes-index gaps using EC2's IP (Element84 throttles local IPs hard;
# EC2 sees <1% failure per prior measurements). Pre-existing entries in
# v2_scenes_index.parquet are kept; this only retries groups missing from it.
echo "[bootstrap-v2] retrying missing scenes from EC2 IP"
python -u -m phase2_classifier.v2.retry_missing_scenes 2>&1 | tee ../v2_scenes_retry.log

python -u phase2_classifier/v2/v2_train.py 2>&1 | tee ../v2_train.log
cd ..

# Final sync (include embed_chunks this time so a resumed run can pick up where we left off)
aws s3 sync data_us/phase2/v2/ "s3://${BUCKET}/v2-artifacts/v2/" --only-show-errors
aws s3 cp v2_train.log "s3://${BUCKET}/v2-artifacts/v2_train.log" --only-show-errors

echo "[bootstrap-v2] done, shutting down in 60s"
sudo shutdown -h +1 "v2 training complete"
