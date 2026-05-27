#!/usr/bin/env bash
# Push the v3 test-set scoring bundle to S3.
# Separate from v3-bundle (training): smaller payload, no training manifest.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")"/../../.. && pwd)"
[ -f "$ROOT/sites_us/.env" ] && set -a && . "$ROOT/sites_us/.env" && set +a

BUCKET="${BUCKET:-${AWS_S3_RESULTS_BUCKET:?AWS_S3_RESULTS_BUCKET not set in sites_us/.env}}"
DEST="s3://${BUCKET}/v3-test-bundle"

DATA_US="$ROOT/data_us"
CODE_DIR="$ROOT/sites_us"

echo "[push-v3-test] uploading to $DEST"

aws s3 cp "$DATA_US/phase2/v3_test_set_manifest.parquet"     "$DEST/v3_test_set_manifest.parquet"     --only-show-errors
aws s3 cp "$DATA_US/phase2/v3_test_set_scenes_index.parquet" "$DEST/v3_test_set_scenes_index.parquet" --only-show-errors
aws s3 cp "$ROOT/sites_us/.env"                              "$DEST/.env"                             --only-show-errors

TMP="$(mktemp -d)"
mkdir -p "$TMP/code/sites_us/phase2_classifier/v3"
cp "$CODE_DIR/phase2_classifier/v3/v3_train.py"      "$TMP/code/sites_us/phase2_classifier/v3/"
cp "$CODE_DIR/phase2_classifier/v3/v3_score_test.py" "$TMP/code/sites_us/phase2_classifier/v3/"
cp "$CODE_DIR/phase2_classifier/v3/bootstrap_v3_test.sh" "$TMP/code/sites_us/phase2_classifier/v3/"
touch "$TMP/code/sites_us/__init__.py"
touch "$TMP/code/sites_us/phase2_classifier/__init__.py"
touch "$TMP/code/sites_us/phase2_classifier/v3/__init__.py"

aws s3 sync "$TMP/code/" "$DEST/code/" --only-show-errors
aws s3 cp "$TMP/code/sites_us/phase2_classifier/v3/bootstrap_v3_test.sh" "$DEST/bootstrap_v3_test.sh" --only-show-errors

rm -rf "$TMP"
echo "[push-v3-test] done."
