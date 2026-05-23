"""Heuristic cloud/quality flag for each chip — looks at pixel std after stretch.

Adds `chip_quality` to each queue entry: "ok", "cloudy" (very low std → washed
out), or "empty" (mostly zero pixels → over-water / edge of scene). Also adds
`chip_std` for the labeler to inspect.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = ROOT / ".artifacts" / "labeling_v2"
QUEUE_PATH = ARTIFACTS / "queue.json"
CHIPS_DIR = ARTIFACTS / "chips"

LOW_STD = 8.0          # uint8 std below this → washed out (cloudy)
EMPTY_FRAC = 0.30      # >30% black pixels → empty/edge of scene


def score(tile_id: str):
    p = CHIPS_DIR / f"{tile_id}.png"
    if not p.exists():
        return tile_id, None, None
    arr = np.asarray(Image.open(p).convert("RGB"))
    std = float(arr.std())
    zero_frac = float(np.mean(arr.sum(axis=-1) < 10))
    if zero_frac > EMPTY_FRAC:
        q = "empty"
    elif std < LOW_STD:
        q = "cloudy"
    else:
        q = "ok"
    return tile_id, q, std


def main() -> int:
    queue = json.loads(QUEUE_PATH.read_text())
    print(f"checking {len(queue)} chips...")
    results = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        for tid, q, s in ex.map(score, [c["tile_id"] for c in queue]):
            results[tid] = (q, s)

    counts = {"ok": 0, "cloudy": 0, "empty": 0, "missing": 0}
    for c in queue:
        q, s = results.get(c["tile_id"], (None, None))
        if q is None:
            counts["missing"] += 1
            c["chip_quality"] = "missing"
        else:
            counts[q] += 1
            c["chip_quality"] = q
            c["chip_std"] = round(s, 1)

    QUEUE_PATH.write_text(json.dumps(queue, indent=2))
    print(f"  ok: {counts['ok']}  cloudy: {counts['cloudy']}  empty: {counts['empty']}  missing: {counts['missing']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
