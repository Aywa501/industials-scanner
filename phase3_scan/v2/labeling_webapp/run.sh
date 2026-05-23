#!/usr/bin/env bash
# Launch the v2 candidate labeling webapp. Run from anywhere.
set -euo pipefail
cd "$(dirname "$0")/../../.."   # cd to sites_us/
exec python3 -m uvicorn phase3_scan.v2.labeling_webapp.server:app \
    --host 127.0.0.1 --port 8765 "$@"
