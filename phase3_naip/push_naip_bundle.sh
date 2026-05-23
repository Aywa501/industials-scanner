#!/usr/bin/env bash
# Push the Phase 3 NAIP worker bundle to S3 so the EC2 box can pull it down.
# Mirrors phase3_scan/v2/push_bundle_v2.sh.
set -euo pipefail

cd "$(dirname "$0")/../.."   # repo root
[ -f sites_us/.env ] && set -a && . ./sites_us/.env && set +a

BUCKET="${AWS_S3_RESULTS_BUCKET:?AWS_S3_RESULTS_BUCKET not set}"
PREFIX="s3://${BUCKET}/naip-bundle"

echo "[push-naip] target: $PREFIX"

# Data the worker reads (only the manifest — NAIP COGs stream from naip-analytic)
aws s3 cp data_us/phase3_naip/naip_manifest.parquet "$PREFIX/naip_manifest.parquet"

# Worker code + telemetry helpers live in naip_sam.py.
aws s3 cp sites_us/phase3_naip/naip_sam.py            "$PREFIX/code/sites_us/phase3_naip/naip_sam.py"
aws s3 cp sites_us/phase3_naip/__init__.py            "$PREFIX/code/sites_us/phase3_naip/__init__.py"
aws s3 cp sites_us/phase3_naip/requirements_naip.txt  "$PREFIX/requirements_naip.txt"
aws s3 cp sites_us/phase3_naip/bootstrap_naip.sh      "$PREFIX/bootstrap_naip.sh"

# .env carries HF_TOKEN + IAM keys for the output bucket (NAIP itself uses
# requester-pays via the instance role)
aws s3 cp sites_us/.env "$PREFIX/.env"

echo "[push-naip] done. Launch with: ./sites_us/phase3_naip/launch_naip.sh"
