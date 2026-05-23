#!/usr/bin/env bash
# Drive find_s2_scenes.py to completion. The script is resumable (skips tiles
# already on disk), so each attempt only re-queries the still-missing tiles.
# On a rate-limit (403) attempt N leaves progress on disk; we cool down and
# re-run until all MGRS tiles are covered.
set -u
cd "$(dirname "$0")/.."

GRID="../data_us/phase3_grid.parquet"
SCENES="../data_us/phase3_scenes.parquet"
COOLDOWN="${SCENES_COOLDOWN:-360}"
MAX_ATTEMPTS="${SCENES_MAX_ATTEMPTS:-40}"

total=$(python3 -c "import pandas as pd;print(pd.read_parquet('$GRID',columns=['mgrs_tile']).mgrs_tile.nunique())")
echo "[driver] target: $total MGRS tiles"

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  echo "[driver] === attempt $attempt/$MAX_ATTEMPTS ($(date '+%H:%M:%S')) ==="
  SCENES_WORKERS="${SCENES_WORKERS:-6}" python3 -u -m phase3_scan.find_s2_scenes

  covered=0
  if [ -f "$SCENES" ]; then
    covered=$(python3 -c "import pandas as pd;print(pd.read_parquet('$SCENES',columns=['mgrs_tile']).mgrs_tile.nunique())")
  fi
  echo "[driver] after attempt $attempt: $covered/$total tiles covered"

  if [ "$covered" -ge "$total" ]; then
    echo "[driver] COMPLETE — $covered/$total tiles. done."
    exit 0
  fi

  echo "[driver] $((total - covered)) tiles still missing; cooling down ${COOLDOWN}s before retry"
  sleep "$COOLDOWN"
done

echo "[driver] gave up after $MAX_ATTEMPTS attempts — $covered/$total covered. progress is saved; re-run to resume."
exit 1
