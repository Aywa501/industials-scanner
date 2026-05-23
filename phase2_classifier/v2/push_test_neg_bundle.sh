#!/usr/bin/env bash
# Push the test-neg grading bundle to S3 so an EC2 box can pull it.
# Mirrors push_v2_bundle.sh but for test_neg_v2_manifest + the
# embed_test_negatives.py pipeline.
#
# Usage:
#   cd sites_us
#   ./phase2_classifier/v2/push_test_neg_bundle.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")"/../../.. && pwd)"
[ -f "$ROOT/sites_us/.env" ] && set -a && . "$ROOT/sites_us/.env" && set +a

BUCKET="${BUCKET:-${AWS_S3_RESULTS_BUCKET:?AWS_S3_RESULTS_BUCKET not set in sites_us/.env}}"
DEST="s3://${BUCKET}/test-neg-bundle"

DATA_US="$ROOT/data_us"
CODE_DIR="$ROOT/sites_us"

echo "[push-test-neg] uploading manifest + code + .env to $DEST"

# Manifest
aws s3 cp "$DATA_US/test_neg_v2_manifest.parquet" \
  "$DEST/test_neg_v2_manifest.parquet" --only-show-errors

# .env carries HF_TOKEN (gated dino_vitb) + AWS_S3_RESULTS_BUCKET
aws s3 cp "$ROOT/sites_us/.env" "$DEST/.env" --only-show-errors

# Code: embed pipeline + the v2 helpers it imports
TMP="$(mktemp -d)"
mkdir -p "$TMP/code/sites_us/phase2_classifier/v2"
cp "$CODE_DIR/phase2_classifier/v2/embed_test_negatives.py"   "$TMP/code/sites_us/phase2_classifier/v2/"
cp "$CODE_DIR/phase2_classifier/v2/build_test_negatives.py"   "$TMP/code/sites_us/phase2_classifier/v2/"
cp "$CODE_DIR/phase2_classifier/v2/v2_train.py"               "$TMP/code/sites_us/phase2_classifier/v2/"
cp "$CODE_DIR/phase2_classifier/v2/v2_build_dataset.py"       "$TMP/code/sites_us/phase2_classifier/v2/"
cp "$CODE_DIR/phase2_classifier/v2/v2_build_scenes_index.py"  "$TMP/code/sites_us/phase2_classifier/v2/"
cp "$CODE_DIR/phase2_classifier/v2/bootstrap_test_neg.sh"     "$TMP/code/sites_us/phase2_classifier/v2/"
touch "$TMP/code/sites_us/__init__.py"
touch "$TMP/code/sites_us/phase2_classifier/__init__.py"
touch "$TMP/code/sites_us/phase2_classifier/v2/__init__.py"

aws s3 sync "$TMP/code/" "$DEST/code/" --only-show-errors

# Top-level bootstrap (pulled by curl on EC2)
aws s3 cp "$TMP/code/sites_us/phase2_classifier/v2/bootstrap_test_neg.sh" \
  "$DEST/bootstrap_test_neg.sh" --only-show-errors

rm -rf "$TMP"

echo "[push-test-neg] done. Now launch g4dn.2xlarge spot and run:"
echo "  curl -O https://${BUCKET}.s3.us-west-2.amazonaws.com/test-neg-bundle/bootstrap_test_neg.sh"
echo "  BUCKET=${BUCKET} nohup bash bootstrap_test_neg.sh > test_neg.out 2>&1 &"
echo "  tail -f test_neg.out"
