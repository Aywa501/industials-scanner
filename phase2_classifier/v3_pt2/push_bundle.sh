#!/usr/bin/env bash
# Push v3_pt2 bundle to S3 for EC2 to pull. Tiny — no torch/transformers needed.
set -euo pipefail
cd "$(dirname "$0")/../../.."

[ -f sites_us/.env.agent-profile ] && set -a && . sites_us/.env.agent-profile && set +a
[ -f sites_us/.env ] && set -a && . sites_us/.env && set +a

BUCKET="${AWS_S3_RESULTS_BUCKET:?AWS_S3_RESULTS_BUCKET not set}"
REGION="${AWS_REGION:-us-west-2}"
PREFIX="s3://${BUCKET}/v3-pt2-bundle"

STAGE=$(mktemp -d)
trap "rm -rf $STAGE" EXIT

mkdir -p "$STAGE/code/sites_us/phase2_classifier/v3_pt2"
cp sites_us/phase2_classifier/v3_pt2/build_candidates.py    "$STAGE/code/sites_us/phase2_classifier/v3_pt2/"
cp sites_us/phase2_classifier/v3_pt2/build_scenes_index.py  "$STAGE/code/sites_us/phase2_classifier/v3_pt2/"
cp sites_us/phase2_classifier/v3_pt2/change_scan.py         "$STAGE/code/sites_us/phase2_classifier/v3_pt2/"
cp sites_us/phase2_classifier/v3_pt2/elbow_scan.py          "$STAGE/code/sites_us/phase2_classifier/v3_pt2/"
cp sites_us/phase2_classifier/v3_pt2/proximity_rescue.py    "$STAGE/code/sites_us/phase2_classifier/v3_pt2/"
cp sites_us/phase2_classifier/v3_pt2/bootstrap.sh           "$STAGE/"
cp sites_us/phase2_classifier/v3_pt2/bootstrap_elbow.sh     "$STAGE/"
cp data_us/phase2/v3/stage2_candidates.parquet              "$STAGE/"
cp data_us/phase2/v3/landsat_scenes_index.parquet           "$STAGE/"
[ -f data_us/phase2/v3/stage2_candidate_polygons.parquet ] && \
  cp data_us/phase2/v3/stage2_candidate_polygons.parquet    "$STAGE/"
cp sites_us/.env                                            "$STAGE/" || true

echo "[push] uploading bundle -> ${PREFIX}/"
aws s3 sync "$STAGE/" "${PREFIX}/" --region "$REGION" --only-show-errors --delete
echo "[push] done"
aws s3 ls "${PREFIX}/" --region "$REGION" --recursive --human-readable --summarize
