"""Validation bench. Not the product.

Rendering happens server-side here on purpose: this exists to answer whether
the render looks convincing on a real photo, not to be fast. The spec's
client-side WebGL swap is a separate plan.
"""
import json
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response

from renderer.compose import render
from renderer.masking import build_mask
from renderer.tiles import build_tile_sheet

app = FastAPI(title="Render bench")
STATIC = Path(__file__).parent / "static"
VALID_AUTO = {"none", "colour-deviation"}


def _decode(raw: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(422, "could not decode image")
    return img


@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC / "index.html").read_text()


@app.post("/render")
async def do_render(payload: str = Form(...),
                    photo: UploadFile = File(...),
                    faces: list[UploadFile] = File(...)):
    try:
        body = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise HTTPException(422, f"payload is not JSON: {exc}")

    quad = body.get("quad") or []
    if len(quad) != 4:
        raise HTTPException(422, "quad needs exactly 4 points, ordered "
                                 "FAR-LEFT FAR-RIGHT NEAR-RIGHT NEAR-LEFT")
    auto = body.get("auto", "none")
    if auto not in VALID_AUTO:
        raise HTTPException(422, f"auto must be one of {sorted(VALID_AUTO)}")

    base = _decode(await photo.read())
    face_images = [_decode(await f.read()) for f in faces]
    if not face_images:
        raise HTTPException(422, "at least one texture face is required")

    sheet = build_tile_sheet(
        face_images,
        tuple(body.get("tile_mm", (600, 600))),
        tuple(body.get("floor_mm", (4000, 3000))),
        float(body.get("px_per_mm", 0.35)),
        layout=body.get("layout", "grid"),
        grout_mm=float(body.get("grout_mm", 2.0)),
        grout_bgr=tuple(body.get("grout_bgr", (205, 205, 205))))

    mask = build_mask(base, [tuple(p) for p in quad],
                      exclude=[[tuple(p) for p in poly]
                               for poly in body.get("exclude", [])],
                      auto=auto,
                      threshold=float(body.get("threshold", 13.0)))
    out = render(base, sheet, [tuple(p) for p in quad], mask=mask,
                 strength=float(body.get("strength", 1.0)))

    ok, buf = cv2.imencode(".png", out)
    if not ok:
        raise HTTPException(500, "failed to encode result")
    return Response(buf.tobytes(), media_type="image/png")
