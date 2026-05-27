#!/usr/bin/env bash
# Tail the running Stage 2b instance's log + heartbeat.
# Usage: ./monitor.sh [<instance_id>]
set -euo pipefail
cd "$(dirname "$0")/../../.."

[ -f sites_us/.env.agent-profile ] && set -a && . sites_us/.env.agent-profile && set +a
[ -f sites_us/.env ] && set -a && . sites_us/.env && set +a

BUCKET="${AWS_S3_RESULTS_BUCKET:?AWS_S3_RESULTS_BUCKET not set}"
REGION="${AWS_REGION:-us-west-2}"

INSTANCE_ID="${1:-$(cat /tmp/stage2b_instance_id.txt 2>/dev/null || true)}"
[ -z "$INSTANCE_ID" ] && { echo "usage: $0 <instance_id>"; exit 1; }

echo "=== heartbeat ==="
aws s3 cp "s3://${BUCKET}/v3-pt2-artifacts/heartbeat/${INSTANCE_ID}.json" - --region "$REGION" 2>/dev/null || echo "(no heartbeat yet)"
echo
echo "=== last 60 log lines ==="
aws s3 cp "s3://${BUCKET}/v3-pt2-artifacts/logs/${INSTANCE_ID}.log" - --region "$REGION" 2>/dev/null | tail -60 || echo "(no log yet)"
echo
echo "=== chunks done ==="
aws s3 ls "s3://${BUCKET}/v3-pt2-artifacts/change_chunks/" --region "$REGION" 2>/dev/null | wc -l
