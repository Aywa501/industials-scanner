#!/usr/bin/env bash
# Push the v2 detector training bundle to S3 so an EC2 box can pull it.
# Mirrors push_bundle.sh but for v2 dataset + train code.
#
# Usage:
#   cd sites_us
#   ./phase2_classifier/v2/push_v2_bundle.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")"/../../.. && pwd)"
[ -f "$ROOT/sites_us/.env" ] && set -a && . "$ROOT/sites_us/.env" && set +a

BUCKET="${BUCKET:-${AWS_S3_RESULTS_BUCKET:?AWS_S3_RESULTS_BUCKET not set in sites_us/.env}}"
DEST="s3://${BUCKET}/v2-bundle"

DATA_US="$ROOT/data_us"
CODE_DIR="$ROOT/sites_us"

echo "[push-v2] uploading manifest + scenes index + code to $DEST"

aws s3 cp "$DATA_US/v2_dataset_manifest.parquet"  "$DEST/v2_dataset_manifest.parquet"  --only-show-errors
aws s3 cp "$DATA_US/v2_scenes_index.parquet"      "$DEST/v2_scenes_index.parquet"      --only-show-errors
aws s3 cp "$ROOT/sites_us/.env"                   "$DEST/.env"                         --only-show-errors

# Code: copy train script + scenes-index builders + bootstrap into bundle/code/sites_us/
# Scene-index gap retry runs on EC2 (its IP isn't rate-limited by Element84).
TMP="$(mktemp -d)"
mkdir -p "$TMP/code/sites_us/phase2_classifier/v2"
cp "$CODE_DIR/phase2_classifier/v2/v2_train.py"             "$TMP/code/sites_us/phase2_classifier/v2/"
cp "$CODE_DIR/phase2_classifier/v2/v2_build_scenes_index.py" "$TMP/code/sites_us/phase2_classifier/v2/"
cp "$CODE_DIR/phase2_classifier/v2/retry_missing_scenes.py" "$TMP/code/sites_us/phase2_classifier/v2/"
cp "$CODE_DIR/phase2_classifier/v2/bootstrap_v2.sh"         "$TMP/code/sites_us/phase2_classifier/v2/"
touch "$TMP/code/sites_us/__init__.py"
touch "$TMP/code/sites_us/phase2_classifier/__init__.py"
touch "$TMP/code/sites_us/phase2_classifier/v2/__init__.py"

# Sync the whole code tree (preserves __init__.py files needed for python -m).
aws s3 sync "$TMP/code/" "$DEST/code/" --only-show-errors

aws s3 cp "$TMP/code/sites_us/phase2_classifier/v2/bootstrap_v2.sh" "$DEST/bootstrap_v2.sh" --only-show-errors

rm -rf "$TMP"

echo "[push-v2] done. Now launch g4dn.2xlarge spot and run:"
echo "  curl -O https://${BUCKET}.s3.us-west-2.amazonaws.com/v2-bundle/bootstrap_v2.sh"
echo "  BUCKET=${BUCKET} nohup bash bootstrap_v2.sh > v2_train.out 2>&1 &"
echo "  tail -f v2_train.out"
