"""FastAPI backend for the v2 phase3 candidate-labeling webapp.

Run: python -m uvicorn phase3_scan.v2.labeling_webapp.server:app --reload
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = ROOT / ".artifacts" / "labeling_v2"
CHIPS_DIR = ARTIFACTS / "chips"
QUEUE_PATH = ARTIFACTS / "queue.json"
STATIC_DIR = Path(__file__).parent / "static"

DATA_US = ROOT.parent / "data_us"
LABELS_PATH = DATA_US / "labels" / "candidate_labels_v2.parquet"
NOTES_PATH = DATA_US / "candidate_notes_v2.parquet"
FLAGS_PATH = DATA_US / "candidate_flags_v2.parquet"
WIDE_DIR = ARTIFACTS / "chips_wide"

VALID_LABELS = {"industrial", "not_industrial", "unsure"}
VALID_FLAGS = {"follow_up"}

_lock = threading.Lock()

app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class LabelIn(BaseModel):
    tile_id: str
    label: str


class DeleteIn(BaseModel):
    tile_id: str


class NoteIn(BaseModel):
    tile_id: str
    note: str


class FlagIn(BaseModel):
    tile_id: str
    flag: str


def _load(path: Path, columns: list[str]) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame({c: pd.Series(dtype="object") for c in columns})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@app.get("/")
def root() -> FileResponse:
    return FileResponse(
        STATIC_DIR / "index.html",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@app.get("/api/queue")
def get_queue() -> JSONResponse:
    if not QUEUE_PATH.exists():
        raise HTTPException(404, "queue.json not found — run prep_candidates.py")
    return JSONResponse(json.loads(QUEUE_PATH.read_text()))


@app.get("/api/labels")
def get_labels() -> JSONResponse:
    df = _load(LABELS_PATH, ["tile_id", "label", "labeled_at"])
    if df.empty:
        return JSONResponse([])
    return JSONResponse(df.to_dict("records"))


@app.post("/api/label")
def post_label(body: LabelIn) -> dict:
    if body.label not in VALID_LABELS:
        raise HTTPException(400, f"label must be one of {sorted(VALID_LABELS)}")
    with _lock:
        df = _load(LABELS_PATH, ["tile_id", "label", "labeled_at"])
        df = df[df.tile_id != body.tile_id]
        new = pd.DataFrame([{
            "tile_id": body.tile_id,
            "label": body.label,
            "labeled_at": _now(),
        }])
        df = pd.concat([df, new], ignore_index=True)
        df.to_parquet(LABELS_PATH, index=False)
    return {"ok": True, "n_labels": len(df)}


@app.delete("/api/label")
def delete_label(body: DeleteIn) -> dict:
    with _lock:
        df = _load(LABELS_PATH, ["tile_id", "label", "labeled_at"])
        before = len(df)
        df = df[df.tile_id != body.tile_id]
        df.to_parquet(LABELS_PATH, index=False)
    return {"ok": True, "removed": before - len(df), "n_labels": len(df)}


@app.get("/api/notes")
def get_notes() -> JSONResponse:
    df = _load(NOTES_PATH, ["tile_id", "note", "noted_at"])
    if df.empty:
        return JSONResponse([])
    return JSONResponse(df.to_dict("records"))


@app.post("/api/note")
def post_note(body: NoteIn) -> dict:
    with _lock:
        df = _load(NOTES_PATH, ["tile_id", "note", "noted_at"])
        df = df[df.tile_id != body.tile_id]
        text = body.note.strip()
        if text:
            new = pd.DataFrame([{
                "tile_id": body.tile_id,
                "note": text,
                "noted_at": _now(),
            }])
            df = pd.concat([df, new], ignore_index=True)
        df.to_parquet(NOTES_PATH, index=False)
    return {"ok": True, "n_notes": len(df)}


@app.get("/api/flags")
def get_flags() -> JSONResponse:
    df = _load(FLAGS_PATH, ["tile_id", "flag", "flagged_at"])
    if df.empty:
        return JSONResponse([])
    return JSONResponse(df.to_dict("records"))


@app.post("/api/flag")
def post_flag(body: FlagIn) -> dict:
    if body.flag not in VALID_FLAGS:
        raise HTTPException(400, f"flag must be one of {sorted(VALID_FLAGS)}")
    with _lock:
        df = _load(FLAGS_PATH, ["tile_id", "flag", "flagged_at"])
        mask = (df.tile_id == body.tile_id) & (df.flag == body.flag)
        if mask.any():
            return {"ok": True, "n_flags": len(df), "noop": True}
        new = pd.DataFrame([{"tile_id": body.tile_id, "flag": body.flag, "flagged_at": _now()}])
        df = pd.concat([df, new], ignore_index=True)
        df.to_parquet(FLAGS_PATH, index=False)
    return {"ok": True, "n_flags": len(df)}


@app.delete("/api/flag")
def delete_flag(body: FlagIn) -> dict:
    with _lock:
        df = _load(FLAGS_PATH, ["tile_id", "flag", "flagged_at"])
        before = len(df)
        df = df[~((df.tile_id == body.tile_id) & (df.flag == body.flag))]
        df.to_parquet(FLAGS_PATH, index=False)
    return {"ok": True, "removed": before - len(df), "n_flags": len(df)}


@app.get("/chips/{tile_id}.png")
def get_chip(tile_id: str) -> FileResponse:
    p = CHIPS_DIR / f"{tile_id}.png"
    if not p.exists():
        raise HTTPException(404, f"no chip at {p}")
    return FileResponse(p, media_type="image/png")


@app.get("/chips_wide/{tile_id}.png")
def get_wide_chip(tile_id: str) -> FileResponse:
    p = WIDE_DIR / f"{tile_id}.png"
    if not p.exists():
        raise HTTPException(404, f"no wide chip at {p}")
    return FileResponse(p, media_type="image/png")
