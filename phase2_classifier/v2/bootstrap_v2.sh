#!/usr/bin/env bash
# Bootstrap a g6.4xlarge in us-west-2 to train the v2 3-class probe.
#
# Usage on EC2:
#   curl -O https://${BUCKET}.s3.us-west-2.amazonaws.com/v2-bundle/bootstrap_v2.sh
#   BUCKET=industrials-scanner-us-west-2 bash bootstrap_v2.sh

set -euo pipefail

BUCKET="${BUCKET:?set BUCKET=industrials-scanner-us-west-2}"
WORK="${HOME}/v2train"
mkdir -p "$WORK"
cd "$WORK"

if [ -f "$HOME/.env" ]; then
  set -a; . "$HOME/.env"; set +a
fi

echo "[bootstrap-v2] pulling bundle from s3://${BUCKET}/v2-bundle"
aws s3 sync "s3://${BUCKET}/v2-bundle/" ./bundle/ --only-show-errors

mkdir -p data_us sites_us
cp bundle/v2_dataset_manifest.parquet     data_us/
cp bundle/stage1_embeddings.npy           data_us/
cp bundle/stage1_embeddings_index.parquet data_us/
cp -r bundle/code/sites_us/*              sites_us/
cp bundle/.env                            sites_us/.env || true

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
  "python-dotenv>=1.0.0"

# GPU sanity
python - <<'PY'
import torch
print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
PY

# Background sync of artifacts during run (spot-interrupt safety)
(
  while true; do
    sleep 300
    aws s3 sync data_us/ "s3://${BUCKET}/v2-artifacts/" \
        --exclude "*" \
        --include "v2_embeddings.npy" \
        --include "v2_embeddings_index.parquet" \
        --include "stage1_industrial_v2.pt" \
        --include "stage1_v2_train_report.json" \
        --include "stage1_v2_eval_report.json" \
        --only-show-errors || true
  done
) &
SYNC_PID=$!
trap 'kill $SYNC_PID 2>/dev/null || true' EXIT

cd sites_us
python -u phase2_classifier/v2/v2_train.py 2>&1 | tee ../v2_train.log
cd ..

# Final sync
aws s3 sync data_us/ "s3://${BUCKET}/v2-artifacts/" \
    --exclude "*" \
    --include "v2_embeddings.npy" \
    --include "v2_embeddings_index.parquet" \
    --include "stage1_industrial_v2.pt" \
    --include "stage1_v2_train_report.json" \
    --include "stage1_v2_eval_report.json" \
    --only-show-errors
aws s3 cp v2_train.log "s3://${BUCKET}/v2-artifacts/v2_train.log" --only-show-errors

echo "[bootstrap-v2] done, shutting down in 60s"
sudo shutdown -h +1 "v2 training complete"
