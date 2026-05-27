#!/usr/bin/env bash
# Push the Phase 3 v2 worker bundle to S3 so the EC2 box can pull it down.
# Run from anywhere — script cd's to repo root.

set -euo pipefail

cd "$(dirname "$0")/../../.."   # repo root
[ -f sites_us/.env ] && set -a && . ./sites_us/.env && set +a

BUCKET="${AWS_S3_RESULTS_BUCKET:?AWS_S3_RESULTS_BUCKET not set}"
PREFIX="s3://${BUCKET}/scan-v2-bundle"

echo "[push-v2] target: $PREFIX"

# Data the worker reads
aws s3 cp data_us/phase3_scan/phase3_grid.parquet      "$PREFIX/phase3_grid.parquet"
aws s3 cp data_us/phase3_scan/phase3_scenes.parquet    "$PREFIX/phase3_scenes.parquet"
aws s3 cp data_us/external/overture_industrial_conus_2025_aligned.parquet "$PREFIX/overture_industrial_conus_2025_aligned.parquet"
aws s3 cp data_us/phase2/v2/probes/probe_dino_vitb.pt "$PREFIX/probe_dino_vitb.pt"

# Worker code
aws s3 cp sites_us/phase3_scan/v2/infer_shard_v2.py "$PREFIX/code/sites_us/phase3_scan/v2/infer_shard_v2.py"
aws s3 cp sites_us/phase3_scan/v2/__init__.py       "$PREFIX/code/sites_us/phase3_scan/v2/__init__.py"
aws s3 cp sites_us/phase3_scan/__init__.py          "$PREFIX/code/sites_us/phase3_scan/__init__.py"
aws s3 cp sites_us/phase3_scan/find_s2_scenes.py    "$PREFIX/code/sites_us/phase3_scan/find_s2_scenes.py"
aws s3 cp sites_us/phase3_scan/v2/bootstrap_v2_scan.sh "$PREFIX/bootstrap_v2_scan.sh"

# Env (HF_TOKEN + AWS keys for the worker process)
aws s3 cp sites_us/.env "$PREFIX/.env"

echo "[push-v2] done. To launch:"
echo "  aws ec2 run-instances ...   # see phase3_scan/v2/launch_ec2.md"
