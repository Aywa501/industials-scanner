"""Generate per-chip industrial-probability heatmaps via patch-token probe.

For each S2 chip we want to visualize: run DINOv3 ViT-L/16 SAT-493M, take the
196 patch tokens (14×14 grid for a 224 input), apply the trained linear probe
head to each, softmax → per-patch probability of "industrial". Upsample to
chip pixel resolution and blend as a heatmap overlay.

Model + chip-PNG cache loaded lazily on first call. Heatmap PNGs cached to
disk so repeat views are instant.
"""

from __future__ import annotations

import io
import os
import threading
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dotenv import load_dotenv
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

DATA_US = ROOT.parent / "data_us"
MODEL_PATH = DATA_US / "phase1" / "stage1_industrial_v1.pt"
ARTIFACTS = ROOT / ".artifacts" / "labeling"
CHIPS_DIR = ARTIFACTS / "chips"
HEATMAP_CACHE_DIR = ARTIFACTS / "heatmaps"

MODEL_ID = "facebook/dinov3-vitl16-pretrain-sat493m"
IMG_SIZE = 224
PATCH_GRID = 14  # 224 / 16
NUM_REG_TOKENS = 4

MEAN = torch.tensor([0.430, 0.411, 0.296]).view(1, 3, 1, 1)
STD = torch.tensor([0.213, 0.156, 0.143]).view(1, 3, 1, 1)

_lock = threading.Lock()
_state: dict = {}


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _ensure_loaded() -> None:
    if "model" in _state:
        return
    with _lock:
        if "model" in _state:
            return
        from transformers import AutoModel
        device = _device()
        print(f"[heatmap] loading {MODEL_ID} on {device}")
        model = AutoModel.from_pretrained(MODEL_ID, dtype=torch.float32)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        model.to(device)

        ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        head = nn.Linear(1024, 2)
        head.load_state_dict(ckpt["state_dict"])
        head.eval()
        head.to(device)

        _state["model"] = model
        _state["head"] = head
        _state["device"] = device
        _state["mean"] = MEAN.to(device)
        _state["std"] = STD.to(device)


def _png_to_input_tensor(png_path: Path) -> torch.Tensor:
    img = Image.open(png_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def _heatmap_to_overlay(prob_grid: np.ndarray, chip_png: bytes) -> bytes:
    """Render a 14x14 prob grid as a translucent RGBA PNG.

    Hot color (red→yellow) where prob is high, transparent elsewhere. The CSS
    overlay (mix-blend-mode: screen) does the visual blending in the browser.
    """
    grid = torch.from_numpy(prob_grid).float().unsqueeze(0).unsqueeze(0)
    base = Image.open(io.BytesIO(chip_png)).convert("RGB")
    H, W = base.size[1], base.size[0]
    up = F.interpolate(grid, size=(H, W), mode="bilinear", align_corners=False)
    up = up.squeeze().numpy()  # (H, W) in [0, 1]

    # Color ramp: 0 → black/transparent, 0.5 → red, 1.0 → yellow.
    rgba = np.zeros((H, W, 4), dtype=np.float32)
    rgba[..., 0] = 255.0                    # red always max
    rgba[..., 1] = np.clip((up - 0.5) * 2, 0, 1) * 255.0  # green ramps in past 0.5
    rgba[..., 3] = np.clip(np.power(up, 0.7), 0, 1) * 220.0  # alpha
    out = Image.fromarray(rgba.clip(0, 255).astype(np.uint8), mode="RGBA")
    buf = io.BytesIO()
    out.save(buf, "PNG", optimize=False)
    return buf.getvalue()


def heatmap_path(site_id: str, year: int) -> Path:
    return HEATMAP_CACHE_DIR / site_id / f"{year}.png"


def generate(site_id: str, year: int) -> bytes:
    """Return overlay PNG bytes for (site_id, year). Disk-cached."""
    out = heatmap_path(site_id, year)
    if out.exists():
        return out.read_bytes()

    chip_png_path = CHIPS_DIR / site_id / f"{year}.png"
    if not chip_png_path.exists():
        raise FileNotFoundError(f"no chip at {chip_png_path}")
    chip_bytes = chip_png_path.read_bytes()

    _ensure_loaded()
    model = _state["model"]
    head = _state["head"]
    device = _state["device"]

    x = _png_to_input_tensor(chip_png_path).to(device)
    x = (x - _state["mean"]) / _state["std"]

    with torch.inference_mode():
        outputs = model(x).last_hidden_state  # (1, 1+R+P, 1024)
        patch_tokens = outputs[:, 1 + NUM_REG_TOKENS:, :]  # (1, 196, 1024)
        logits = head(patch_tokens.squeeze(0))  # (196, 2)
        probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
    grid = probs.reshape(PATCH_GRID, PATCH_GRID)

    overlay = _heatmap_to_overlay(grid, chip_bytes)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(overlay)
    return overlay
