#!/usr/bin/env bash
# Push the Phase 3 v1 worker bundle to S3 so the EC2 box can pull it down.
# Run from anywhere — script cd's to repo root.

set -euo pipefail

cd "$(dirname "$0")/../../.."   # repo root
[ -f sites_us/.env ] && set -a && . ./sites_us/.env && set +a

BUCKET="${AWS_S3_RESULTS_BUCKET:?AWS_S3_RESULTS_BUCKET not set}"
PREFIX="s3://${BUCKET}/scan-bundle"

echo "[push] target: $PREFIX"

# Data the worker reads
aws s3 cp data_us/phase3_scan/phase3_grid.parquet      "$PREFIX/phase3_grid.parquet"
aws s3 cp data_us/phase3_scan/phase3_scenes.parquet    "$PREFIX/phase3_scenes.parquet"
aws s3 cp data_us/phase1/stage1_industrial_v1.pt  "$PREFIX/stage1_industrial_v1.pt"

# Worker code
aws s3 cp sites_us/phase3_scan/v1/infer_shard.py "$PREFIX/code/sites_us/phase3_scan/v1/infer_shard.py"
aws s3 cp sites_us/phase3_scan/v1/__init__.py    "$PREFIX/code/sites_us/phase3_scan/v1/__init__.py"
aws s3 cp sites_us/phase3_scan/__init__.py       "$PREFIX/code/sites_us/phase3_scan/__init__.py"
aws s3 cp sites_us/phase3_scan/v1/bootstrap.sh   "$PREFIX/bootstrap.sh"

# Env (HF_TOKEN + AWS keys for the worker process)
aws s3 cp sites_us/.env "$PREFIX/.env"

echo "[push] done. To launch:"
echo "  aws ec2 run-instances ...   # see phase3_scan/v1/launch_ec2.md"
