#!/usr/bin/env bash
# Push the v3 detector training bundle to S3 so an EC2 box can pull it.
#
# Usage:
#   cd sites_us
#   ./phase2_classifier/v3/push_v3_bundle.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")"/../../.. && pwd)"
[ -f "$ROOT/sites_us/.env" ] && set -a && . "$ROOT/sites_us/.env" && set +a

BUCKET="${BUCKET:-${AWS_S3_RESULTS_BUCKET:?AWS_S3_RESULTS_BUCKET not set in sites_us/.env}}"
DEST="s3://${BUCKET}/v3-bundle"

DATA_US="$ROOT/data_us"
CODE_DIR="$ROOT/sites_us"

echo "[push-v3] uploading manifest + scenes index + code to $DEST"

aws s3 cp "$DATA_US/phase2/v3_dataset_manifest.parquet" "$DEST/v3_dataset_manifest.parquet" --only-show-errors
aws s3 cp "$DATA_US/phase2/v3_scenes_index.parquet"     "$DEST/v3_scenes_index.parquet"     --only-show-errors
aws s3 cp "$ROOT/sites_us/.env"                         "$DEST/.env"                        --only-show-errors

# Code: v3 scripts + __init__ files so python -m works
TMP="$(mktemp -d)"
mkdir -p "$TMP/code/sites_us/phase2_classifier/v3"
cp "$CODE_DIR/phase2_classifier/v3/v3_train.py"             "$TMP/code/sites_us/phase2_classifier/v3/"
cp "$CODE_DIR/phase2_classifier/v3/v3_build_dataset.py"     "$TMP/code/sites_us/phase2_classifier/v3/"
cp "$CODE_DIR/phase2_classifier/v3/v3_build_scenes_index.py" "$TMP/code/sites_us/phase2_classifier/v3/"
cp "$CODE_DIR/phase2_classifier/v3/bootstrap_v3.sh"         "$TMP/code/sites_us/phase2_classifier/v3/"
touch "$TMP/code/sites_us/__init__.py"
touch "$TMP/code/sites_us/phase2_classifier/__init__.py"
touch "$TMP/code/sites_us/phase2_classifier/v3/__init__.py"

aws s3 sync "$TMP/code/" "$DEST/code/" --only-show-errors
aws s3 cp "$TMP/code/sites_us/phase2_classifier/v3/bootstrap_v3.sh" "$DEST/bootstrap_v3.sh" --only-show-errors

rm -rf "$TMP"

echo "[push-v3] done. Launch instance (g4dn.8xlarge primary; g6.8xlarge / g6e.2xlarge / g4dn.2xlarge fallback) and run:"
echo "  aws s3 cp s3://${BUCKET}/v3-bundle/bootstrap_v3.sh ."
echo "  BUCKET=${BUCKET} nohup bash bootstrap_v3.sh > v3_train.out 2>&1 &"
echo "  tail -f v3_train.out"
