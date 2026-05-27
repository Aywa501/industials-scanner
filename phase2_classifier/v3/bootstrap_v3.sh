#!/usr/bin/env bash
# Bootstrap a GPU box in us-west-2 (DLAMI: Deep Learning OSS PyTorch 2.x) to
# run the v3 detector training (NAIP per-building crops, dino_sat493m + dino_vitb).
#
# Required instance: g4dn.8xlarge (T4, 128 GB, 50 Gbps up-to, NVMe) primary.
# This workload is I/O-bound (S3 NAIP-COG reads), not GPU-bound — we want the
# fattest network pipe within the 32-vCPU spot quota. Only fallback:
# g6.8xlarge (L4, 32 vCPU, 25 Gbps). No other sizes/families (user rule).
#
# Usage on EC2:
#   aws s3 cp s3://${BUCKET}/v3-bundle/bootstrap_v3.sh .
#   BUCKET=industrials-scanner-us-west-2 nohup bash bootstrap_v3.sh > v3_train.out 2>&1 &
#   tail -f v3_train.out

set -euo pipefail

BUCKET="${BUCKET:?set BUCKET=industrials-scanner-us-west-2}"
WORK="${HOME}/v3train"
mkdir -p "$WORK"
cd "$WORK"

# ---------- Instance-type guard (IMDSv2) -------------------------------------
echo "[bootstrap-v3] checking instance metadata via IMDSv2..."
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
echo "[bootstrap-v3] instance: $INSTANCE_ID, type: $INSTANCE_TYPE, region: $REGION"

# Hard block: any *.xlarge has 16 GB RAM and will OOM during prep_pool.
case "$INSTANCE_TYPE" in
  *.xlarge)
    echo "ERROR: '$INSTANCE_TYPE' has only 16 GB RAM, insufficient for v3 prep_pool."
    echo "  Required: g4dn.8xlarge (128 GB RAM)."
    exit 1
    ;;
esac

# Soft block: must be g4dn.8xlarge or g6.8xlarge (user rule). No .2xlarge or
# g6e variants. Bootstrap aborts on anything else.
EXPECTED_TYPE="g4dn.8xlarge"
ALLOWED_FALLBACKS="g6.8xlarge"
if [ "$INSTANCE_TYPE" != "$EXPECTED_TYPE" ]; then
  override_ok=0
  if [ "${ALLOW_INSTANCE_TYPE:-}" = "$INSTANCE_TYPE" ]; then
    for ok_type in $ALLOWED_FALLBACKS; do
      [ "$INSTANCE_TYPE" = "$ok_type" ] && override_ok=1 && break
    done
  fi
  if [ $override_ok -eq 0 ]; then
    echo "ERROR: '$INSTANCE_TYPE' not allowed. Required: $EXPECTED_TYPE."
    echo "  Fallbacks (via ALLOW_INSTANCE_TYPE): $ALLOWED_FALLBACKS"
    exit 1
  fi
  echo "[bootstrap-v3] using fallback instance type (override applied)"
fi
echo "[bootstrap-v3] instance type OK"

# Bootstrap creds: caller SCPs .env to $HOME for the initial S3 sync.
if [ -f "$HOME/.env" ]; then
  set -a; . "$HOME/.env"; set +a
fi

echo "[bootstrap-v3] pulling bundle from s3://${BUCKET}/v3-bundle"
aws s3 sync "s3://${BUCKET}/v3-bundle/" ./bundle/ --only-show-errors

mkdir -p data_us/phase2 sites_us
cp bundle/v3_dataset_manifest.parquet data_us/phase2/
cp bundle/v3_scenes_index.parquet     data_us/phase2/
cp -r bundle/code/sites_us/*          sites_us/
cp bundle/.env                        sites_us/.env || true

set -a; . ./sites_us/.env; set +a

# GDAL/curl tuning for the COG-from-S3 fetch loop. v3_train.py also sets these
# via rasterio.Env(), but exporting them here means any subprocess (e.g. aws s3
# sync, sanity checks) inherits the tuned defaults. Per memory
# `sentinel-cogs auth — IAM user works, instance role 403s`, the .env IAM user
# is what reads naip-analytic; AWS_REQUEST_PAYER is required for that bucket.
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

# g4dn.8xlarge ships a 900 GB NVMe instance store at /dev/nvme1n1, usually
# mounted by the DLAMI at /opt/dlami/nvme. Log its state for post-run audit;
# embed_chunks currently live on the EBS root (gp3 defaults handle the write
# load fine — chunks are small fp16 .npy + parquet, < 1 GB total).
if [ -d /opt/dlami/nvme ]; then
  echo "[bootstrap-v3] NVMe instance store available at /opt/dlami/nvme"
  df -h /opt/dlami/nvme || true
fi

# DLAMI Nov-2025+ ships PyTorch via conda. Older DLAMIs use the /opt/pytorch venv.
# Both activations reference unbound vars under `set -u`; relax around activation.
if [ -f /opt/pytorch/bin/activate ]; then
  set +u; source /opt/pytorch/bin/activate; set -u
elif [ -f /opt/conda/etc/profile.d/conda.sh ]; then
  set +u
  source /opt/conda/etc/profile.d/conda.sh
  conda activate pytorch
  set -u
else
  echo "[bootstrap-v3] ERROR: no PyTorch env at /opt/pytorch or /opt/conda/envs/pytorch" >&2
  exit 1
fi

# transformers pinned to 4.56.0 (matches v2 training stack — newer versions trip
# torch.library.infer_schema on the MoE custom_op at AutoModel import).
pip install --quiet --upgrade \
  "transformers==4.56.0" \
  "rasterio>=1.3.9" \
  "boto3>=1.34.0" \
  "pyproj>=3.6.0" \
  "pandas>=2.1.0" \
  "pyarrow>=14.0.0" \
  "Pillow>=10.0.0" \
  "scikit-learn>=1.3.0" \
  "python-dotenv>=1.0.0" \
  "huggingface_hub>=0.19.0"

# HF login for gated dino_vitb model (dino_sat493m is open, vitb is gated)
if [ -n "${HF_TOKEN:-}" ]; then
  echo "[bootstrap-v3] hf login"
  python - <<PY
from huggingface_hub import login; import os
login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
PY
fi

# GPU sanity
python - <<'PY'
import torch, rasterio
print("CUDA:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("rasterio:", rasterio.__version__)
PY

# Background sync of v3/ artifacts during run (spot-interrupt safety).
# Excludes per-chunk resume cache (large; only useful for in-place resume).
LOG_S3="s3://${BUCKET}/v3-artifacts/${INSTANCE_ID}.out"
(
  while true; do
    sleep 300
    aws s3 sync data_us/phase2/v3/ "s3://${BUCKET}/v3-artifacts/v3/" \
        --exclude "embed_chunks/*" \
        --only-show-errors || true
    aws s3 cp "$HOME/v3_train.out" "$LOG_S3" --only-show-errors || true
  done
) &
SYNC_PID=$!

# Auto-terminate on exit. Final sync first so spot-interrupt state is preserved.
# Per memory `feedback_ec2_shutdown_verify`: terminate via EC2 API, don't rely on
# `shutdown -h`, to avoid EBS orphaning if instance-initiated-behavior is "stop".
on_exit() {
  echo "[bootstrap-v3] EXIT: final sync"
  aws s3 cp "$HOME/v3_train.out" "$LOG_S3" --only-show-errors || true
  aws s3 sync data_us/phase2/v3/ "s3://${BUCKET}/v3-artifacts/v3/" \
      --only-show-errors || true
  kill $SYNC_PID 2>/dev/null || true
  echo "[bootstrap-v3] terminating $INSTANCE_ID"
  aws ec2 terminate-instances --region "$REGION" \
    --instance-ids "$INSTANCE_ID" 2>&1 || true
}
trap on_exit EXIT

# Raise FD limit: with V3_IO_WORKERS=1024 the multi-tile mosaic path can hold
# ~3000 concurrent rasterio dataset FDs. Default 1024 would hit EMFILE.
ulimit -n 16384

cd sites_us
if [ -n "${V3_PREFLIGHT_ONLY:-}" ]; then
  echo "[bootstrap-v3] preflight-only mode: N=$V3_PREFLIGHT_ONLY"
  python -u phase2_classifier/v3/v3_train.py --preflight-only "$V3_PREFLIGHT_ONLY" 2>&1 | tee ../v3_train.log
else
  python -u phase2_classifier/v3/v3_train.py 2>&1 | tee ../v3_train.log
fi
cd ..
