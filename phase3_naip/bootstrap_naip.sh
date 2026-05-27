#!/usr/bin/env bash
# Bootstrap a g6.8xlarge (primary) or g6.4xlarge (fallback) in us-west-2
# (DLAMI: Deep Learning OSS PyTorch 2.6+) to run the Phase 3 NAIP worker:
# SAM 3 text-prompted segmentation of NAIP imagery, per cluster.
#
# Usage on the EC2 host as ubuntu:
#   aws s3 cp s3://${BUCKET}/naip-bundle/bootstrap_naip.sh .
#   BUCKET=industrials-scanner-us-west-2 SHARD=0/1 \
#     nohup bash bootstrap_naip.sh > naip.out 2>&1 &
#
# Environment knobs:
#   BUCKET         — S3 bucket holding the bundle + outputs (required)
#   SHARD          — N/M, this instance processes the N-th of M slices (default 0/1)
#   LIMIT          — cap on clusters this run (default: no cap)
#   SAM3_PROMPTS   — override default prompt list (default: 9 prompts in naip_sam.py)

set -euo pipefail

BUCKET="${BUCKET:?set BUCKET=industrials-scanner-us-west-2 (or your bucket name)}"
SHARD="${SHARD:-0/1}"
LIMIT_FLAG=""
if [ -n "${LIMIT:-}" ]; then
  LIMIT_FLAG="--limit ${LIMIT}"
fi

WORK="${HOME}/naip-stage"
mkdir -p "$WORK"
cd "$WORK"

# Source pre-staged .env (HF_TOKEN, etc.) if present from user-data.
if [ -f "$HOME/.env" ]; then
  set -a; . "$HOME/.env"; set +a
fi

echo "[bootstrap-naip] pulling bundle from s3://${BUCKET}/naip-bundle"
aws s3 sync "s3://${BUCKET}/naip-bundle/" ./bundle/ --only-show-errors

mkdir -p data_us/phase3_naip sites_us
cp bundle/naip_manifest.parquet data_us/phase3_naip/
cp -r bundle/code/sites_us/*    sites_us/
cp bundle/.env                  sites_us/.env

# Export env so HF + AWS creds are visible to the worker
set -a
. ./sites_us/.env
set +a

# DLAMI Nov-2025+ ships PyTorch as conda env at /opt/conda/envs/pytorch.
# Older DLAMIs use the /opt/pytorch venv. Support both — both activate scripts
# reference unbound vars (LD_LIBRARY_PATH, MKL_INTERFACE_LAYER) that trip set -u.
if [ -f /opt/pytorch/bin/activate ]; then
  set +u
  source /opt/pytorch/bin/activate
  set -u
elif [ -f /opt/conda/etc/profile.d/conda.sh ]; then
  set +u
  source /opt/conda/etc/profile.d/conda.sh
  conda activate pytorch
  set -u
else
  echo "[bootstrap-naip] ERROR: no PyTorch env found at /opt/pytorch or /opt/conda/envs/pytorch" >&2
  exit 1
fi

# Install pinned worker deps. transformers 5.x is required for SAM 3 (Sam3Model /
# Sam3Processor); 4.x does not ship the sam3 module. See [[pin_transformers]].
pip install --quiet -r bundle/requirements_naip.txt

# HF login (SAM 3 weights public, but login is cheap and keeps HF rate-limits
# from biting under sustained 800k-cluster fetches)
if [ -n "${HF_TOKEN:-}" ]; then
  echo "[bootstrap-naip] hf login"
  python - <<PY
from huggingface_hub import login
import os
login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
PY
fi

# Sanity check: CUDA, rasterio, transformers version pin, SAM 3 import.
# Failing here is preferred to failing 30 min into a run.
python - <<'PY'
import sys, torch, rasterio, transformers
print("CUDA:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("rasterio:", rasterio.__version__)
print("transformers:", transformers.__version__)
from transformers import Sam3Model, Sam3Processor
print("SAM 3 classes resolvable: Sam3Model, Sam3Processor")
# Real-GPU kernel check. torch.cuda.is_available() is True even when the device's
# compute capability isn't supported by the installed PyTorch (e.g. sm_120 Blackwell
# on PT 2.6 which tops out at sm_90). A trivial matmul forces kernel dispatch and
# fails fast if the GPU is unusable.
try:
    a = torch.randn(64, 64, device="cuda")
    b = (a @ a).sum().item()
    print(f"GPU matmul OK ({b:.2f})")
except Exception as e:
    print(f"GPU matmul FAILED: {type(e).__name__}: {e}", file=sys.stderr)
    sys.exit(1)
PY

# IMDSv2 for instance metadata (needed for terminate-instances at end + log key)
echo "[bootstrap-naip] querying instance metadata via IMDSv2..."
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
echo "[bootstrap-naip] instance: $INSTANCE_ID, type: $INSTANCE_TYPE, region: $REGION, shard: $SHARD"

# Telemetry — local writes, periodic S3 sync below. Keys land under
# s3://$BUCKET/phase3-naip-telemetry/$INSTANCE_ID/ so we can analyze even if the
# instance is reclaimed.
TELEM_LOCAL="data_us/phase3_naip/telemetry"
mkdir -p "$TELEM_LOCAL"
TELEM_S3="s3://${BUCKET}/phase3-naip-telemetry/${INSTANCE_ID}"
LOG_S3="s3://${BUCKET}/phase3-naip-logs/${INSTANCE_ID}.out"

# GDAL HTTP tuning for NAIP COG reads. NAIP COGs are smaller (~50-200 MB) than
# Sentinel-2 but more numerous; HTTP/2 multiplex amortizes connection setup.
export GDAL_HTTP_TIMEOUT=120
export GDAL_HTTP_CONNECTTIMEOUT=30
export GDAL_HTTP_MULTIPLEX=YES
export GDAL_HTTP_VERSION=2

# Periodic background sync — spot-interrupt safety. Loses ≤5 min of telemetry
# if AWS reclaims the instance. Mask outputs are written direct to S3 by
# naip_sam.py so they're already durable; telemetry + bootstrap log are not.
(
  while true; do
    sleep 300
    aws s3 sync "$TELEM_LOCAL/" "$TELEM_S3/" --only-show-errors || true
    aws s3 cp "$HOME/naip.out" "$LOG_S3" --only-show-errors || true
  done
) &
SYNC_PID=$!

# Auto-terminate on exit. Sync log + telemetry FIRST so the final state
# (including whatever killed us) is captured before the instance disappears.
on_exit() {
  echo "[bootstrap-naip] EXIT: final telemetry + log sync"
  aws s3 sync "$TELEM_LOCAL/" "$TELEM_S3/" --only-show-errors || true
  aws s3 cp "$HOME/naip.out" "$LOG_S3" --only-show-errors || true
  kill $SYNC_PID 2>/dev/null || true
  echo "[bootstrap-naip] terminating $INSTANCE_ID"
  aws ec2 terminate-instances --region "$REGION" \
    --instance-ids "$INSTANCE_ID" 2>&1 || true
}
trap on_exit EXIT

# Run the worker. naip_sam.py skips clusters whose masks.parquet already exists
# on S3 (output_exists check), so a re-launch resumes cleanly. The structured
# stats.jsonl + system.jsonl land in $TELEM_LOCAL and sync to $TELEM_S3.
cd sites_us
echo "[bootstrap-naip] === starting naip_sam (shard=$SHARD) ==="
python -u -m phase3_naip.naip_sam --shard "$SHARD" $LIMIT_FLAG
cd ..

# Final sync to catch trailing telemetry the bg loop hasn't picked up.
aws s3 sync "$TELEM_LOCAL/" "$TELEM_S3/" --only-show-errors
echo "[bootstrap-naip] telemetry uploaded to $TELEM_S3"
echo "[bootstrap-naip] worker complete, terminating instance via EXIT trap"
# trap on_exit fires automatically on exit
