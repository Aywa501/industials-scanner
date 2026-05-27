"""FastAPI backend for the manual labeling webapp.

Run: python -m uvicorn phase2_classifier.labeling_webapp.server:app --reload
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).parent.parent.parent
ARTIFACTS = ROOT / ".artifacts" / "labeling"
CHIPS_DIR = ARTIFACTS / "chips"
# LABELING_QUEUE_FILE env var picks the queue file relative to ARTIFACTS.
# Defaults to queue.json. Use shortlist_queue.json for the relabel-shortlist pass.
QUEUE_FILE = os.getenv("LABELING_QUEUE_FILE", "queue.json")
QUEUE_PATH = ARTIFACTS / QUEUE_FILE
STATIC_DIR = Path(__file__).parent / "static"

DATA_US = ROOT.parent / "data_us"
LABELS_PATH = DATA_US / "labels" / "manual_labels.parquet"
FLAGS_PATH = DATA_US / "labels" / "manual_site_flags.parquet"
NOTES_PATH = DATA_US / "labels" / "manual_site_notes.parquet"
OUTLINES_PATH = DATA_US / "labels" / "manual_site_outlines.parquet"

VALID_LABELS = {"complete", "partial", "not_a_site", "unsure"}
VALID_FLAGS = {"bad_geocode"}

_lock = threading.Lock()

app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class LabelIn(BaseModel):
    site_id: str
    year: int
    label: str


class FlagIn(BaseModel):
    site_id: str
    flag: str


class NoteIn(BaseModel):
    site_id: str
    note: str


class DeleteIn(BaseModel):
    site_id: str
    year: int


class OutlineIn(BaseModel):
    site_id: str
    polygon: list[list[float]]


class OutlineDeleteIn(BaseModel):
    site_id: str


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
        raise HTTPException(404, "queue.json not found — run prep_data.py")
    return JSONResponse(json.loads(QUEUE_PATH.read_text()))


@app.get("/api/labels")
def get_labels() -> JSONResponse:
    df = _load(LABELS_PATH, ["site_id", "year", "label", "labeled_at"])
    if df.empty:
        return JSONResponse([])
    return JSONResponse(df.to_dict("records"))


@app.get("/api/flags")
def get_flags() -> JSONResponse:
    df = _load(FLAGS_PATH, ["site_id", "flag", "flagged_at"])
    if df.empty:
        return JSONResponse([])
    return JSONResponse(df.to_dict("records"))


@app.post("/api/label")
def post_label(body: LabelIn) -> dict:
    if body.label not in VALID_LABELS:
        raise HTTPException(400, f"label must be one of {sorted(VALID_LABELS)}")
    with _lock:
        df = _load(LABELS_PATH, ["site_id", "year", "label", "labeled_at"])
        mask = (df.site_id == body.site_id) & (df.year == body.year)
        df = df[~mask]
        new = pd.DataFrame([{
            "site_id": body.site_id,
            "year": int(body.year),
            "label": body.label,
            "labeled_at": _now(),
        }])
        df = pd.concat([df, new], ignore_index=True)
        df.to_parquet(LABELS_PATH, index=False)
    return {"ok": True, "n_labels": len(df)}


@app.delete("/api/label")
def delete_label(body: DeleteIn) -> dict:
    with _lock:
        df = _load(LABELS_PATH, ["site_id", "year", "label", "labeled_at"])
        before = len(df)
        df = df[~((df.site_id == body.site_id) & (df.year == body.year))]
        df.to_parquet(LABELS_PATH, index=False)
        return {"ok": True, "removed": before - len(df), "n_labels": len(df)}


@app.post("/api/flag")
def post_flag(body: FlagIn) -> dict:
    if body.flag not in VALID_FLAGS:
        raise HTTPException(400, f"flag must be one of {sorted(VALID_FLAGS)}")
    with _lock:
        df = _load(FLAGS_PATH, ["site_id", "flag", "flagged_at"])
        mask = (df.site_id == body.site_id) & (df.flag == body.flag)
        if mask.any():
            return {"ok": True, "n_flags": len(df), "noop": True}
        new = pd.DataFrame([{
            "site_id": body.site_id,
            "flag": body.flag,
            "flagged_at": _now(),
        }])
        df = pd.concat([df, new], ignore_index=True)
        df.to_parquet(FLAGS_PATH, index=False)
    return {"ok": True, "n_flags": len(df)}


@app.delete("/api/flag")
def delete_flag(body: FlagIn) -> dict:
    with _lock:
        df = _load(FLAGS_PATH, ["site_id", "flag", "flagged_at"])
        before = len(df)
        df = df[~((df.site_id == body.site_id) & (df.flag == body.flag))]
        df.to_parquet(FLAGS_PATH, index=False)
    return {"ok": True, "removed": before - len(df), "n_flags": len(df)}


@app.get("/api/notes")
def get_notes() -> JSONResponse:
    df = _load(NOTES_PATH, ["site_id", "note", "noted_at"])
    if df.empty:
        return JSONResponse([])
    return JSONResponse(df.to_dict("records"))


@app.post("/api/note")
def post_note(body: NoteIn) -> dict:
    with _lock:
        df = _load(NOTES_PATH, ["site_id", "note", "noted_at"])
        df = df[df.site_id != body.site_id]
        text = body.note.strip()
        if text:
            new = pd.DataFrame([{
                "site_id": body.site_id,
                "note": text,
                "noted_at": _now(),
            }])
            df = pd.concat([df, new], ignore_index=True)
        df.to_parquet(NOTES_PATH, index=False)
    return {"ok": True, "n_notes": len(df)}


@app.get("/api/outlines")
def get_outlines() -> JSONResponse:
    df = _load(OUTLINES_PATH, ["site_id", "polygon", "outlined_at"])
    if df.empty:
        return JSONResponse([])
    out = []
    for r in df.to_dict("records"):
        try:
            r["polygon"] = json.loads(r["polygon"])
        except Exception:
            r["polygon"] = []
        out.append(r)
    return JSONResponse(out)


@app.post("/api/outline")
def post_outline(body: OutlineIn) -> dict:
    if len(body.polygon) < 3:
        raise HTTPException(400, "polygon needs at least 3 vertices")
    for pt in body.polygon:
        if len(pt) != 2 or not all(0.0 <= v <= 1.0 for v in pt):
            raise HTTPException(400, "polygon vertices must be normalized [x,y] in [0,1]")
    with _lock:
        df = _load(OUTLINES_PATH, ["site_id", "polygon", "outlined_at"])
        df = df[df.site_id != body.site_id]
        new = pd.DataFrame([{
            "site_id": body.site_id,
            "polygon": json.dumps(body.polygon),
            "outlined_at": _now(),
        }])
        df = pd.concat([df, new], ignore_index=True)
        df.to_parquet(OUTLINES_PATH, index=False)
    return {"ok": True, "n_outlines": len(df)}


@app.delete("/api/outline")
def delete_outline(body: OutlineDeleteIn) -> dict:
    with _lock:
        df = _load(OUTLINES_PATH, ["site_id", "polygon", "outlined_at"])
        before = len(df)
        df = df[df.site_id != body.site_id]
        df.to_parquet(OUTLINES_PATH, index=False)
    return {"ok": True, "removed": before - len(df), "n_outlines": len(df)}


@app.get("/chips/{site_id}/{filename}")
def get_chip(site_id: str, filename: str) -> FileResponse:
    p = CHIPS_DIR / site_id / filename
    if not p.exists():
        raise HTTPException(404, f"no chip at {p}")
    return FileResponse(p, media_type="image/png")


@app.get("/heatmaps/{site_id}/{year}.png")
def get_heatmap(site_id: str, year: int):
    from fastapi import Response
    from phase2_classifier import heatmap_gen
    try:
        png = heatmap_gen.generate(site_id, year)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    return Response(content=png, media_type="image/png")
