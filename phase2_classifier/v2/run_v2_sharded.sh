#!/usr/bin/env bash
# Sharded launcher: runs N parallel python workers each handling one shard
# of the manifest, then merges and trains probes. N defaults to nproc but
# can be overridden by NUM_SHARDS env var.
set -euo pipefail

WORK=/home/ubuntu/v2train
cd "$WORK"

set -a
[ -f "$HOME/.env" ] && . "$HOME/.env"
[ -f "$WORK/sites_us/.env" ] && . "$WORK/sites_us/.env"
set +a

source /opt/pytorch/bin/activate

NUM_SHARDS="${NUM_SHARDS:-$(nproc)}"
BUCKET="${BUCKET:-industrials-scanner-us-west-2}"

echo "[run-v2-sharded] launching $NUM_SHARDS parallel embed workers"

# Periodic S3 sync of artifacts as shards finish (spot-interrupt safety)
(
  while true; do
    sleep 300
    aws s3 sync "$WORK/data_us/phase2/v2/" "s3://${BUCKET}/v2-artifacts/v2/" --only-show-errors || true
  done
) &
SYNC_PID=$!
trap "kill $SYNC_PID 2>/dev/null || true" EXIT

cd sites_us
PIDS=()
for i in $(seq 0 $((NUM_SHARDS - 1))); do
  python -u phase2_classifier/v2/v2_train.py --num-shards "$NUM_SHARDS" --shard-index "$i" \
    > "$WORK/v2_train_shard${i}.log" 2>&1 &
  PIDS+=($!)
  echo "[run-v2-sharded]   shard $i: pid=${PIDS[$i]}, log=v2_train_shard${i}.log"
done
cd ..

echo "[run-v2-sharded] waiting on ${#PIDS[@]} shards..."
FAIL=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    echo "[run-v2-sharded] shard pid=$pid FAILED"
    FAIL=1
  fi
done

if [ "$FAIL" = "1" ]; then
  echo "[run-v2-sharded] one or more shards failed; aborting before merge"
  exit 1
fi

echo "[run-v2-sharded] all shards done; merging"
python -u -m sites_us.phase2_classifier.v2.v2_merge_shards

echo "[run-v2-sharded] training probes on merged embeddings"
cd sites_us
python -u phase2_classifier/v2/v2_train.py --skip-embed
cd ..

aws s3 sync "$WORK/data_us/phase2/v2/" "s3://${BUCKET}/v2-artifacts/v2/" --only-show-errors || true
echo "[run-v2-sharded] done"
