"""Embed the test_neg_v2 held-out negatives and re-grade existing probes.

Pipeline:
  1. Build a scenes index for test_neg_v2_manifest (mirrors v2_build_scenes_index.py).
  2. Run v2_train.fetch_and_embed on the new manifest with redirected output paths
     (data_us/test_neg_v2/ instead of data_us/v2/).
  3. Apply the existing 5 saved probes and compute thresholds capturing
     95 / 99 / 100 % of strict-complete hand-labeled positives, with FPR
     measured on these in-distribution negatives.

Local Mac CPU run: terratorch is unavailable so Prithvi encoders skip
silently in load_models. Probes for those encoders are simply not graded
(matches v2_train's existing tolerance pattern).
"""
from __future__ import annotations

# sentinel-cogs is fully public — on the local Mac, disable AWS signing
# globally before any rasterio import so worker threads (which don't inherit
# `rasterio.Env`) read COGs anonymously instead of trying to refresh an
# expired SSO token. On EC2 the instance role works fine, so the bootstrap
# sets TEST_NEG_AWS_ANON=NO to fall through to v2_train's default rasterio
# env (AWSSession + requester_pays). Must run BEFORE `import rasterio`
# (or transitive imports like v2_train) so GDAL picks it up.
import os as _os
if _os.environ.get("TEST_NEG_AWS_ANON", "YES").upper() != "NO":
    _os.environ["AWS_NO_SIGN_REQUEST"] = "YES"
    _os.environ.pop("AWS_PROFILE", None)
    _os.environ.pop("AWS_SESSION_TOKEN", None)
    _os.environ.pop("AWS_ACCESS_KEY_ID", None)
    _os.environ.pop("AWS_SECRET_ACCESS_KEY", None)

import json
import time
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
DATA_US = ROOT / "data_us"

TEST_NEG_MANIFEST = DATA_US / "test_neg_v2_manifest.parquet"
TEST_NEG_SCENES   = DATA_US / "test_neg_v2_scenes_index.parquet"
TEST_NEG_DIR      = DATA_US / "test_neg_v2"
TEST_NEG_DIR.mkdir(parents=True, exist_ok=True)

# Probes + prior-run embeddings (used to look up strict positives).
# Defaults match the local-Mac /tmp layout from the earlier 2-encoder run;
# the EC2 bootstrap overrides via env vars to point at data_us/v2/ where it
# pulls v2-artifacts.
LOCAL_PROBES_DIR   = Path(_os.environ.get("TEST_NEG_PROBES_DIR",
                                          "/tmp/v2-results/probes"))
EXISTING_EMB_INDEX = Path(_os.environ.get("TEST_NEG_EXISTING_EMB_INDEX",
                                          "/tmp/v2-results/v2_embeddings_index.parquet"))
EXISTING_EMB_DIR   = Path(_os.environ.get("TEST_NEG_EXISTING_EMB_DIR",
                                          "/tmp/v2-results/embeds"))


# ---------------------------------------------------------------------------
# Step 1: build scenes index for the new manifest. Reuses the helpers from
# v2_build_scenes_index.py exactly.
# ---------------------------------------------------------------------------

def build_scenes_index_for(manifest: pd.DataFrame) -> pd.DataFrame:
    from phase2_classifier.v2 import v2_build_scenes_index as bsi
    from pystac_client import Client

    print(f"[test-neg-embed] computing MGRS for {len(manifest):,} rows")
    mgrs_arr = bsi._compute_mgrs(manifest.lat.to_numpy(), manifest.lon.to_numpy())
    manifest = manifest.assign(mgrs_tile=mgrs_arr).dropna(subset=["mgrs_tile"]).reset_index(drop=True)

    groups = manifest[["mgrs_tile", "target_year"]].drop_duplicates().reset_index(drop=True)
    groups["target_year"] = groups["target_year"].astype(int)
    print(f"[test-neg-embed] {len(groups)} unique (mgrs, year) groups to query STAC")

    client = Client.open(bsi.STAC_URL)

    rows: list[dict] = []
    failed: list[tuple[str, int, str]] = []
    empty: list[tuple[str, int]] = []
    # Lower concurrency than EC2 default since residential IPs throttle harder.
    # On EC2 the bootstrap sets TEST_NEG_STAC_WORKERS to bsi.STAC_WORKERS.
    N_WORKERS = int(_os.environ.get("TEST_NEG_STAC_WORKERS", "4"))
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(bsi._query_one, client, r.mgrs_tile, int(r.target_year)):
                (r.mgrs_tile, int(r.target_year))
                for r in groups.itertuples(index=False)}
        for i, fut in enumerate(as_completed(futs), 1):
            key = futs[fut]
            try:
                got = fut.result()
            except Exception as e:
                failed.append((*key, repr(e)))
                continue
            if got:
                rows.extend(got)
            else:
                empty.append(key)
            if i % 100 == 0:
                print(f"[test-neg-embed]   STAC {i}/{len(groups)} done "
                      f"(scene_rows={len(rows)} empty={len(empty)} failed={len(failed)})")
    df = pd.DataFrame(rows)
    print(f"[test-neg-embed] STAC done: {len(df):,} scene rows for "
          f"{df.groupby(['mgrs_tile','year']).ngroups if len(df) else 0} groups; "
          f"empty={len(empty)} failed={len(failed)}")
    return df, manifest


# ---------------------------------------------------------------------------
# Step 2: run v2_train.fetch_and_embed with redirected outputs
# ---------------------------------------------------------------------------

def run_embed(manifest: pd.DataFrame, scenes_path: Path, out_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Monkey-patch v2_train's path constants to point at our test_neg_v2 dir,
    then drive fetch_and_embed. Returns (idx_df, embs_dict)."""
    from phase2_classifier.v2 import v2_train
    import rasterio

    # On the local Mac, override v2_train's AWSSession-based rasterio env to
    # use anonymous auth so we don't trip on an expired SSO token. On EC2,
    # TEST_NEG_AWS_ANON=NO leaves v2_train's default in place so the instance
    # role handles auth normally.
    if _os.environ.get("TEST_NEG_AWS_ANON", "YES").upper() != "NO":
        def _anon_env():
            return rasterio.Env(
                AWS_NO_SIGN_REQUEST="YES",
                GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                GDAL_HTTP_MULTIPLEX="YES",
                GDAL_HTTP_VERSION="2",
                VSI_CACHE="TRUE",
                VSI_CACHE_SIZE=200_000_000,
                CPL_VSIL_CURL_CHUNK_SIZE=1_048_576,
                CPL_VSIL_CURL_CACHE_SIZE=200_000_000,
            )
        v2_train.setup_rasterio_env = _anon_env

    # Override paths to land all artifacts under TEST_NEG_DIR
    v2_train.SCENES_INDEX     = scenes_path
    v2_train.V2_DIR           = out_dir
    v2_train.PROBES_DIR       = out_dir / "probes_unused"
    v2_train.EMB_IDX_OUT      = out_dir / "test_neg_v2_index.parquet"
    v2_train.EMBED_CHUNKS_DIR = out_dir / "embed_chunks"
    v2_train.STATS_LOG        = out_dir / "stats.jsonl"
    v2_train.LEADERBOARD      = out_dir / "leaderboard_unused.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    v2_train.EMBED_CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    # Fewer workers for a local Mac. On EC2 (TEST_NEG_USE_V2_DEFAULTS=1) skip
    # these overrides so v2_train's tuned defaults take effect.
    if _os.environ.get("TEST_NEG_USE_V2_DEFAULTS", "0") != "1":
        v2_train.IO_WORKERS    = 24
        v2_train.PREP_WORKERS  = 4
        v2_train.PREFETCH_GROUPS = 1
        v2_train.PREP_CHUNK    = 64
        v2_train.MEMORY_BUDGET_BYTES = 4 * 1024**3   # 4 GB sub-chunk budget

    # Compute MGRS for the manifest (fetch_and_embed groups by mgrs_tile)
    print(f"[test-neg-embed] computing MGRS on manifest")
    manifest = manifest.assign(mgrs_tile=v2_train.compute_mgrs(
        manifest.lat.to_numpy(), manifest.lon.to_numpy()
    )).dropna(subset=["mgrs_tile"]).reset_index(drop=True)
    print(f"[test-neg-embed] manifest rows after MGRS: {len(manifest):,}")

    # TEST_NEG_DEVICE: cpu (default, local Mac) | cuda (EC2 GPU). MPS is flaky
    # for DINOv3 and Prithvi is unavailable on Mac anyway, so cpu locally.
    device = torch.device(_os.environ.get("TEST_NEG_DEVICE", "cpu"))
    print(f"[test-neg-embed] device: {device}")

    requested = {s["name"] for s in v2_train.MODEL_REGISTRY}
    models = v2_train.load_models(device, requested)
    if not models:
        raise RuntimeError("no models loaded")

    print(f"[test-neg-embed] running fetch_and_embed on {len(manifest):,} rows")
    n, idx_df, embs = v2_train.fetch_and_embed(manifest, models, device)
    print(f"[test-neg-embed] embed pass complete: {n} tiles, "
          f"models with embeddings: {sorted(embs.keys())}")

    # Save outputs
    idx_df.to_parquet(out_dir / "test_neg_v2_index.parquet", index=False)
    for name, arr in embs.items():
        np.save(out_dir / f"emb_{name}.npy", arr)
    return idx_df, embs


# ---------------------------------------------------------------------------
# Step 3: apply existing probes and compute metrics
# ---------------------------------------------------------------------------

def load_strict_positives() -> dict[str, np.ndarray]:
    """Returns {model_name: p_industrial array of length n_strict_pos}."""
    DATA_US_LOCAL = ROOT / "data_us"
    manifest_full = pd.read_parquet(DATA_US_LOCAL / "v2_dataset_manifest.parquet")
    emb_idx = pd.read_parquet(EXISTING_EMB_INDEX)

    # Bring is_inferred from manifest into idx
    join = emb_idx.merge(manifest_full[["tile_id","is_inferred"]], on="tile_id", how="left")
    strict_pos = join[(join.split == "test")
                      & join.source.astype(str).str.startswith("hand_complete")
                      & (join.class_id == 2)
                      & (join.is_inferred == False)].reset_index(drop=True)
    print(f"[test-neg-embed] strict positives (directly-labeled, survived embed): {len(strict_pos)}")

    out = {}
    for probe_path in sorted(LOCAL_PROBES_DIR.glob("probe_*.pt")):
        if probe_path.name.endswith("_eval.json"): continue
        name = probe_path.stem.removeprefix("probe_")
        ckpt = torch.load(probe_path, map_location="cpu", weights_only=False)
        emb = np.load(EXISTING_EMB_DIR / f"emb_{name}.npy")
        head = nn.Linear(ckpt["emb_dim"], ckpt["n_classes"])
        head.load_state_dict(ckpt["state_dict"]); head.eval()
        with torch.inference_mode():
            logits = head(torch.from_numpy(emb).float())
            p_ind  = F.softmax(logits, dim=1).numpy()[:, 1]
        out[name] = p_ind[strict_pos.row_idx.values]
    return out


def score_negatives(idx_df: pd.DataFrame, embs: dict) -> dict[str, np.ndarray]:
    out = {}
    for name, emb in embs.items():
        probe_path = LOCAL_PROBES_DIR / f"probe_{name}.pt"
        if not probe_path.exists():
            print(f"  no probe for {name}, skipping")
            continue
        ckpt = torch.load(probe_path, map_location="cpu", weights_only=False)
        head = nn.Linear(ckpt["emb_dim"], ckpt["n_classes"])
        head.load_state_dict(ckpt["state_dict"]); head.eval()
        with torch.inference_mode():
            logits = head(torch.from_numpy(emb).float())
            p_ind  = F.softmax(logits, dim=1).numpy()[:, 1]
        # Only test rows survive in idx_df by construction (manifest had only test rows)
        out[name] = p_ind[idx_df.row_idx.values]
    return out


def grade(strict_pos_p: dict, neg_p: dict) -> None:
    print()
    print("=" * 92)
    print("=== Threshold required to capture X% of strict positives, FPR on FRESH OSM negatives ===")
    print("=" * 92)
    print(f"{'model':<14} {'pos':>5} {'neg':>5} | "
          f"{'thr@95':>8} {'FPR':>7} | {'thr@99':>8} {'FPR':>7} | {'thr@100':>9} {'FPR':>7}")
    common = sorted(set(strict_pos_p) & set(neg_p))
    for name in common:
        pos = np.sort(strict_pos_p[name])
        neg = neg_p[name]
        n_pos = len(pos)
        def thr_fpr(target_recall):
            k = max(0, min(int(np.floor((1 - target_recall) * n_pos)), n_pos - 1))
            t = pos[k]
            return float(t), float((neg >= t).mean())
        t95, f95   = thr_fpr(0.95)
        t99, f99   = thr_fpr(0.99)
        t100, f100 = thr_fpr(1.00)
        print(f"{name:<14} {n_pos:>5} {len(neg):>5} | "
              f"{t95:>8.4f} {f95:>7.1%} | {t99:>8.4f} {f99:>7.1%} | {t100:>9.4f} {f100:>7.1%}")


def main() -> None:
    if not TEST_NEG_MANIFEST.exists():
        sys.exit(f"missing {TEST_NEG_MANIFEST}; run build_test_negatives.py first")
    manifest = pd.read_parquet(TEST_NEG_MANIFEST)
    print(f"[test-neg-embed] manifest rows: {len(manifest):,}")

    if TEST_NEG_SCENES.exists():
        scenes_df = pd.read_parquet(TEST_NEG_SCENES)
        print(f"[test-neg-embed] reusing existing scenes index: {len(scenes_df):,} rows")
    else:
        scenes_df, manifest = build_scenes_index_for(manifest)
        scenes_df.to_parquet(TEST_NEG_SCENES, index=False)
        print(f"[test-neg-embed] wrote scenes index -> {TEST_NEG_SCENES}")

    idx_out = TEST_NEG_DIR / "test_neg_v2_index.parquet"
    emb_paths = {s["name"]: TEST_NEG_DIR / f"emb_{s['name']}.npy" for s in __import__("phase2_classifier.v2.v2_train", fromlist=["MODEL_REGISTRY"]).MODEL_REGISTRY}
    if idx_out.exists() and any(p.exists() for p in emb_paths.values()):
        print(f"[test-neg-embed] reusing existing embeddings at {TEST_NEG_DIR}")
        idx_df = pd.read_parquet(idx_out)
        embs = {n: np.load(p) for n, p in emb_paths.items() if p.exists()}
    else:
        idx_df, embs = run_embed(manifest, TEST_NEG_SCENES, TEST_NEG_DIR)

    print()
    print(f"[test-neg-embed] graded negatives: {len(idx_df)} of 1000 manifest rows survived embed")

    print(f"[test-neg-embed] loading strict positives from existing embeddings")
    strict_pos_p = load_strict_positives()
    neg_p = score_negatives(idx_df, embs)

    grade(strict_pos_p, neg_p)


if __name__ == "__main__":
    main()
