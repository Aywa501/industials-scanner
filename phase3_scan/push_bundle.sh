#!/usr/bin/env bash
# Push the Phase 3 worker bundle to S3 so the EC2 box can pull it down.
# Run from sites_us/.

set -euo pipefail

cd "$(dirname "$0")/../.."   # repo root
[ -f sites_us/.env ] && set -a && . ./sites_us/.env && set +a

BUCKET="${AWS_S3_RESULTS_BUCKET:?AWS_S3_RESULTS_BUCKET not set}"
PREFIX="s3://${BUCKET}/scan-bundle"

echo "[push] target: $PREFIX"

# Data the worker reads
aws s3 cp data_us/phase3_grid.parquet      "$PREFIX/phase3_grid.parquet"
aws s3 cp data_us/phase3_scenes.parquet    "$PREFIX/phase3_scenes.parquet"
aws s3 cp data_us/stage1_industrial_v1.pt  "$PREFIX/stage1_industrial_v1.pt"

# Worker code
aws s3 cp sites_us/phase3_scan/infer_shard.py "$PREFIX/code/sites_us/phase3_scan/infer_shard.py"
aws s3 cp sites_us/phase3_scan/__init__.py    "$PREFIX/code/sites_us/phase3_scan/__init__.py"
aws s3 cp sites_us/phase3_scan/bootstrap.sh   "$PREFIX/bootstrap.sh"

# Env (HF_TOKEN + AWS keys for the worker process)
aws s3 cp sites_us/.env "$PREFIX/.env"

echo "[push] done. To launch:"
echo "  aws ec2 run-instances ...   # see phase3_scan/launch_ec2.md"
