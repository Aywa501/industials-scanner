#!/usr/bin/env bash
# Push the v3 scan bundle to S3.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")"/../../.. && pwd)"
[ -f "$ROOT/sites_us/.env" ] && set -a && . "$ROOT/sites_us/.env" && set +a

BUCKET="${BUCKET:-${AWS_S3_RESULTS_BUCKET:?AWS_S3_RESULTS_BUCKET not set in sites_us/.env}}"
DEST="s3://${BUCKET}/v3-scan-bundle"

DATA_US="$ROOT/data_us"
CODE_DIR="$ROOT/sites_us"

echo "[push-scan] uploading to $DEST"

aws s3 cp "$DATA_US/phase2/v3_scan_manifest.parquet"      "$DEST/v3_scan_manifest.parquet"      --only-show-errors
aws s3 cp "$DATA_US/phase2/v3_scan_scenes_index.parquet"  "$DEST/v3_scan_scenes_index.parquet"  --only-show-errors
aws s3 cp "$ROOT/sites_us/.env"                           "$DEST/.env"                          --only-show-errors

TMP="$(mktemp -d)"
mkdir -p "$TMP/code/sites_us/phase2_classifier/v3"
cp "$CODE_DIR/phase2_classifier/v3/v3_train.py"        "$TMP/code/sites_us/phase2_classifier/v3/"
cp "$CODE_DIR/phase2_classifier/v3/v3_scan_infer.py"   "$TMP/code/sites_us/phase2_classifier/v3/"
cp "$CODE_DIR/phase2_classifier/v3/bootstrap_v3_scan.sh" "$TMP/code/sites_us/phase2_classifier/v3/"
touch "$TMP/code/sites_us/__init__.py"
touch "$TMP/code/sites_us/phase2_classifier/__init__.py"
touch "$TMP/code/sites_us/phase2_classifier/v3/__init__.py"

aws s3 sync "$TMP/code/" "$DEST/code/" --only-show-errors
aws s3 cp "$TMP/code/sites_us/phase2_classifier/v3/bootstrap_v3_scan.sh" "$DEST/bootstrap_v3_scan.sh" --only-show-errors

rm -rf "$TMP"
echo "[push-scan] done."
