"""Score saved v3 probes against the per-building test set.

Reads:
  data_us/phase2/v3_test_set_manifest.parquet
  data_us/phase2/v3_test_set_scenes_index.parquet
  data_us/phase2/v3/probes/probe_<model>.pt

Writes:
  data_us/phase2/v3/test_leaderboard.json
  data_us/phase2/v3/test_predictions.parquet   (per-row probs for both models)

Runs on local MPS (Apple GPU). Set V3_TEST_DEVICE=cpu / cuda to override.
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# AWS_NO_SIGN_REQUEST OFF: naip-analytic is requester-pays; .env IAM user is the
# principal we use locally (per memory `sentinel-cogs auth — IAM user works…`).
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
ENV_PATH = ROOT / "sites_us" / ".env"
if ENV_PATH.exists():
    for line in ENV_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score

from sites_us.phase2_classifier.v3.v3_train import (
    _fetch_tile_group, _fetch_crop, prep_crop, setup_rasterio_env,
    load_models, build_norm_tensors, MODEL_REGISTRY, IMG_INPUT,
)

DATA_US = ROOT / "data_us"
TEST_MANIFEST = DATA_US / "phase2" / "v3_test_set_manifest.parquet"
TEST_SCENES = DATA_US / "phase2" / "v3_test_set_scenes_index.parquet"
PROBES_DIR = DATA_US / "phase2" / "v3" / "probes"
OUT_DIR = DATA_US / "phase2" / "v3"
LEADERBOARD = OUT_DIR / "test_leaderboard.json"
PREDS = OUT_DIR / "test_predictions.parquet"

DEVICE = os.environ.get("V3_TEST_DEVICE", "mps")
IO_WORKERS = int(os.environ.get("V3_TEST_IO_WORKERS", 32))    # residential bandwidth — modest
PREP_WORKERS = int(os.environ.get("V3_TEST_PREP_WORKERS", 8))
BATCH_SIZE = int(os.environ.get("V3_TEST_BATCH", 32))
N_CLASSES = 2


def fetch_and_embed(manifest: pd.DataFrame, scenes: pd.DataFrame,
                    device) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Join manifest+scenes, group by primary tile, fetch crops, embed all models."""
    merged = manifest.merge(scenes[["building_id", "naip_uris",
                                    "fetch_xmin", "fetch_ymin",
                                    "fetch_xmax", "fetch_ymax", "n_tiles"]],
                            on="building_id", how="left")
    merged = merged[merged["n_tiles"].fillna(0) > 0].reset_index(drop=True)
    print(f"[v3-score] {len(merged):,} test buildings with NAIP coverage")

    # Sort by primary tile URI so open-once amortizes
    merged["primary_uri"] = merged["naip_uris"].map(
        lambda u: u[0] if isinstance(u, (list, np.ndarray)) and len(u) > 0 else ""
    )
    merged = merged.sort_values("primary_uri").reset_index(drop=True)

    # Partition: single-tile vs multi-tile
    single_groups: dict[str, list[tuple]] = {}
    multi_items: list[tuple] = []
    for i, r in enumerate(merged.itertuples(index=False)):
        uris = list(r.naip_uris) if r.naip_uris is not None else []
        coords = (i, float(r.fetch_xmin), float(r.fetch_ymin),
                  float(r.fetch_xmax), float(r.fetch_ymax))
        if len(uris) == 1:
            single_groups.setdefault(uris[0], []).append(coords)
        elif len(uris) > 1:
            multi_items.append((i, uris, coords[1], coords[2], coords[3], coords[4]))
    print(f"[v3-score] tile groups: {len(single_groups)} single + {len(multi_items)} multi")

    t0 = time.time()
    crops_arr: list[np.ndarray | None] = [None] * len(merged)
    with setup_rasterio_env():
        io_pool = ThreadPoolExecutor(max_workers=IO_WORKERS)
        group_futs = [io_pool.submit(_fetch_tile_group, 0, uri, items)
                      for uri, items in single_groups.items()]
        multi_futs = [io_pool.submit(_fetch_crop, 0, uris, x0, y0, x1, y1)
                      for (_, uris, x0, y0, x1, y1) in multi_items]
        done = 0
        for fu in group_futs:
            for ri, a in fu.result():
                crops_arr[ri] = a
            done += 1
            if done % 100 == 0:
                print(f"[v3-score] fetch groups: {done}/{len(group_futs)} "
                      f"({time.time() - t0:.0f}s)")
        for (ri, *_), fu in zip(multi_items, multi_futs):
            crops_arr[ri] = fu.result()
        io_pool.shutdown(wait=False)
    print(f"[v3-score] fetch done in {time.time() - t0:.0f}s")

    prep_pool = ThreadPoolExecutor(max_workers=PREP_WORKERS)
    prep_futs = [prep_pool.submit(prep_crop, 0, a) for a in crops_arr]
    crops = [pf.result() for pf in prep_futs]
    prep_pool.shutdown(wait=False)
    n_ok = sum(1 for c in crops if c is not None)
    print(f"[v3-score] prep: {n_ok}/{len(crops)} succeeded")

    keep_rows = [i for i, c in enumerate(crops) if c is not None]
    if not keep_rows:
        raise RuntimeError("no successful prep")
    merged = merged.iloc[keep_rows].reset_index(drop=True)
    crops = [crops[i] for i in keep_rows]

    requested = [m["name"] for m in MODEL_REGISTRY]
    print(f"[v3-score] loading models: {requested}")
    models = load_models(device, requested)
    norms = build_norm_tensors(models, device)

    embs: dict[str, list[np.ndarray]] = {n: [] for n in models}
    t_emb = time.time()
    for i in range(0, len(crops), BATCH_SIZE):
        batch = crops[i:i+BATCH_SIZE]
        x = torch.stack(batch).to(device)
        with torch.inference_mode():
            for name, (model, _) in models.items():
                mean, std = norms[name]
                emb = MODEL_REGISTRY_BY_NAME[name]["forward_fn"](model, (x - mean) / std)
                embs[name].append(emb.float().cpu().numpy().astype(np.float16))
        if (i // BATCH_SIZE + 1) % 5 == 0:
            print(f"[v3-score] embed batch {i//BATCH_SIZE + 1}/{(len(crops)+BATCH_SIZE-1)//BATCH_SIZE} "
                  f"({time.time() - t_emb:.0f}s)")
    print(f"[v3-score] embed done in {time.time() - t_emb:.0f}s")
    embs_arr = {n: np.concatenate(embs[n], axis=0) for n in embs}
    return merged, embs_arr


MODEL_REGISTRY_BY_NAME = {m["name"]: m for m in MODEL_REGISTRY}


def score(name: str, emb: np.ndarray, y_bin: np.ndarray, device,
          probe_suffix: str = "") -> tuple[dict, np.ndarray]:
    """Score one probe. probe_suffix='' loads probe_<name>.pt (binary);
    probe_suffix='_multiclass' loads probe_<name>_multiclass.pt and reads
    industrial_idx from the checkpoint to pick the right softmax column."""
    ckpt = torch.load(PROBES_DIR / f"probe_{name}{probe_suffix}.pt",
                      map_location=device, weights_only=False)
    head = nn.Linear(ckpt["emb_dim"], ckpt.get("n_classes", N_CLASSES)).to(device)
    head.load_state_dict(ckpt["state_dict"]); head.eval()
    with torch.inference_mode():
        probs = F.softmax(head(torch.from_numpy(emb).float().to(device)),
                          dim=1).cpu().numpy()
    industrial_idx = ckpt.get("industrial_idx", 1)
    p_ind = probs[:, industrial_idx]
    out = {
        "model": f"{name}{probe_suffix}",
        "test_n": int(len(y_bin)),
        "non_n": int((y_bin == 0).sum()),
        "industrial_n": int((y_bin == 1).sum()),
    }
    if y_bin.any() and (1 - y_bin).any():
        out["auroc"] = float(roc_auc_score(y_bin, p_ind))
        out["ap"] = float(average_precision_score(y_bin, p_ind))
    out["recall_p>=0.5"] = float((p_ind[y_bin == 1] >= 0.5).mean())
    out["recall_p>=0.7"] = float((p_ind[y_bin == 1] >= 0.7).mean())
    out["recall_p>=0.95"] = float((p_ind[y_bin == 1] >= 0.95).mean())
    out["fpr_p>=0.5"] = float((p_ind[y_bin == 0] >= 0.5).mean())
    out["fpr_p>=0.7"] = float((p_ind[y_bin == 0] >= 0.7).mean())
    return out, p_ind


def main() -> None:
    print(f"[v3-score] device: {DEVICE}")
    device = torch.device(DEVICE)

    manifest = pd.read_parquet(TEST_MANIFEST)
    scenes = pd.read_parquet(TEST_SCENES)
    print(f"[v3-score] manifest: {len(manifest):,}, scenes: {len(scenes):,}")

    merged, embs = fetch_and_embed(manifest, scenes, device)
    y_bin = (merged["class_id"].values == 2).astype(int)
    print(f"[v3-score] scored set: industrial={int(y_bin.sum())} non={int((1-y_bin).sum())}")

    results = {}
    pred_df = merged[["building_id", "ovt_id", "class_id", "source", "lat", "lon"]].copy()
    for name in embs:
        for suffix in ["", "_multiclass"]:
            probe_path = PROBES_DIR / f"probe_{name}{suffix}.pt"
            if not probe_path.exists():
                continue
            res, p_ind = score(name, embs[name], y_bin, device, probe_suffix=suffix)
            key = f"{name}{suffix}"
            results[key] = res
            pred_df[f"p_industrial_{key}"] = p_ind
            print(f"[v3-score] {key}: AUROC={res.get('auroc'):.4f} AP={res.get('ap'):.4f} "
                  f"r@0.5={res['recall_p>=0.5']:.3f} r@0.7={res['recall_p>=0.7']:.3f} "
                  f"fpr@0.5={res['fpr_p>=0.5']:.3f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LEADERBOARD.write_text(json.dumps(results, indent=2))
    pred_df.to_parquet(PREDS, index=False)
    print(f"[v3-score] wrote {LEADERBOARD} and {PREDS}")


if __name__ == "__main__":
    main()
