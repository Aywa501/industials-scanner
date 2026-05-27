#!/usr/bin/env bash
# Push v3_pt4 bundle to S3 for EC2 to pull. CPU-only — no torch/transformers.
set -euo pipefail
cd "$(dirname "$0")/../../.."

[ -f sites_us/.env.agent-profile ] && set -a && . sites_us/.env.agent-profile && set +a
[ -f sites_us/.env ] && set -a && . sites_us/.env && set +a

BUCKET="${AWS_S3_RESULTS_BUCKET:?AWS_S3_RESULTS_BUCKET not set}"
REGION="${AWS_REGION:-us-west-2}"
PREFIX="s3://${BUCKET}/v3-pt4-bundle"

STAGE=$(mktemp -d)
trap "rm -rf $STAGE" EXIT

mkdir -p "$STAGE/code/sites_us/phase2_classifier/v3_pt4"
cp sites_us/phase2_classifier/v3_pt4/nlcd_impervious_scan.py        "$STAGE/code/sites_us/phase2_classifier/v3_pt4/"
cp sites_us/phase2_classifier/v3_pt4/bootstrap.sh                   "$STAGE/"
cp data_us/phase2/v3/stage3_candidates_v3.parquet                   "$STAGE/"
cp data_us/phase2/v3/stage2_candidate_polygons.parquet              "$STAGE/"
cp sites_us/.env                                                    "$STAGE/" || true

echo "[push] uploading bundle -> ${PREFIX}/"
aws s3 sync "$STAGE/" "${PREFIX}/" --region "$REGION" --only-show-errors --delete
echo "[push] done"
aws s3 ls "${PREFIX}/" --region "$REGION" --recursive --human-readable --summarize
