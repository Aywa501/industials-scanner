#!/usr/bin/env bash
# Bootstrap a g6e.xlarge in us-west-2 (DLAMI: Deep Learning OSS PyTorch 2.x)
# to run the Phase 3 CONUS scan.
#
# Usage on the EC2 host as ubuntu:
#   curl -O https://${BUCKET}.s3.us-west-2.amazonaws.com/scan-bundle/bootstrap.sh
#   BUCKET=industrials-scanner-us-west-2 bash bootstrap.sh
#
# The script pulls the rest of the bundle from s3://$BUCKET/scan-bundle/.

set -euo pipefail

BUCKET="${BUCKET:?set BUCKET=industrials-scanner-us-west-2 (or your bucket name)}"
WORK="${HOME}/scan"
mkdir -p "$WORK"
cd "$WORK"

# Bootstrap creds: an SCP step from the launching machine put .env and
# bootstrap.sh in $HOME. Source .env first so the s3 sync below has AWS keys.
if [ -f "$HOME/.env" ]; then
  set -a; . "$HOME/.env"; set +a
fi

echo "[bootstrap] pulling bundle from s3://${BUCKET}/scan-bundle"
aws s3 sync "s3://${BUCKET}/scan-bundle/" ./bundle/ --only-show-errors

mkdir -p data_us sites_us
cp bundle/phase3_grid.parquet      data_us/
cp bundle/phase3_scenes.parquet    data_us/
cp bundle/stage1_industrial_v1.pt  data_us/
cp -r bundle/code/sites_us/*       sites_us/
cp bundle/.env                     sites_us/.env

# Export env so HF + AWS creds are visible to the worker
set -a
. ./sites_us/.env
set +a

# DLAMI ships PyTorch as a venv at /opt/pytorch.
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
  "python-dotenv>=1.0.0"

# Sanity check: GPU + S3 read
python - <<'PY'
import torch, rasterio
print("CUDA:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("rasterio:", rasterio.__version__)
PY

# Build work list = MGRS tiles with scenes but no result yet.
mkdir -p data_us/phase3_results
python - <<'PY'
from pathlib import Path
import pandas as pd
scenes = pd.read_parquet("data_us/phase3_scenes.parquet")
done = {p.stem for p in Path("data_us/phase3_results").glob("*.parquet")
        if not p.stem.endswith("_emb")}
todo = sorted(set(scenes.mgrs_tile) - done)
Path("mgrs_todo.txt").write_text("\n".join(todo))
print(f"[bootstrap] {len(todo)} MGRS shards to process ({len(done)} already done)")
PY

# Periodic background sync — spot-interrupt safety. Loses ≤5 min of work if
# AWS reclaims the instance.
mkdir -p data_us/phase3_results
(
  while true; do
    sleep 300
    aws s3 sync data_us/phase3_results/ "s3://${BUCKET}/phase3_results/" \
        --only-show-errors || true
  done
) &
SYNC_PID=$!
trap 'kill $SYNC_PID 2>/dev/null || true' EXIT

# Run the scan. The worker checkpoints per-MGRS, so safe to interrupt + resume.
cd sites_us
python -m phase3_scan.infer_shard --mgrs-list ../mgrs_todo.txt 2>&1 | tee ../scan.log
cd ..

# Final sync to catch anything the bg loop hasn't picked up.
aws s3 sync data_us/phase3_results/ "s3://${BUCKET}/phase3_results/" --only-show-errors
echo "[bootstrap] uploaded results to s3://${BUCKET}/phase3_results/"

# Auto-terminate. The instance was launched with InstanceInitiatedShutdownBehavior=terminate
# so this halts the EC2 billing meter automatically.
echo "[bootstrap] scan complete, shutting down in 60s"
sudo shutdown -h +1 "phase3 scan complete"
