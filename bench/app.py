"""Validation bench. Not the product.

Rendering happens server-side here on purpose: this exists to answer whether
the render looks convincing on a real photo, not to be fast. The spec's
client-side WebGL swap is a separate plan.
"""
import io
import json
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageCms, ImageOps
from pillow_heif import register_heif_opener
from starlette.concurrency import run_in_threadpool

from bench.quotes import router as quote_router
from bench.projects import router as project_router
from renderer.compose import render
from renderer.floor_detection import detect_floor
from renderer.floor_plane import estimate_floor_plane
from renderer.surface_selection import refine_floor_detection, select_surface
from renderer.masking import build_mask
from renderer.tiles import build_tile_sheet, tile_sheet_scale

app = FastAPI(title="Render bench")
app.include_router(quote_router)
app.include_router(project_router)
STATIC = Path(__file__).parent / "static"
VALID_AUTO = {"none", "colour-deviation"}
app.mount("/static", StaticFiles(directory=STATIC), name="static")
register_heif_opener()


def _decode(raw: bytes) -> np.ndarray:
    if not raw:
        raise HTTPException(422, "could not decode image")
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        try:
            with Image.open(io.BytesIO(raw)) as source:
                rgb = ImageOps.exif_transpose(source).convert("RGB")
                if profile := source.info.get("icc_profile"):
                    rgb = ImageCms.profileToProfile(
                        rgb, ImageCms.ImageCmsProfile(io.BytesIO(profile)),
                        ImageCms.createProfile("sRGB"), outputMode="RGB")
                img = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
        except (OSError, ValueError, ImageCms.PyCMSError) as exc:
            raise HTTPException(422, "could not decode image") from exc
    return img


@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC / "index.html").read_text()


@app.post("/image")
async def normalize_image(file: UploadFile = File(...)):
    image = _decode(await file.read())
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise HTTPException(500, "failed to encode image")
    return Response(buf.tobytes(), media_type="image/png")


def _render_surface(base: np.ndarray, face_images: list[np.ndarray], body: dict):
    quad = body.get("quad") or []
    if len(quad) != 4:
        raise HTTPException(422, "quad needs exactly 4 points, ordered "
                                 "FAR-LEFT FAR-RIGHT NEAR-RIGHT NEAR-LEFT")
    floor_boundary = body.get("floor_boundary") or None
    if floor_boundary is not None and len(floor_boundary) < 3:
        raise HTTPException(422, "floor boundary needs at least 3 points")
    auto = body.get("auto", "none")
    if auto not in VALID_AUTO:
        raise HTTPException(422, f"auto must be one of {sorted(VALID_AUTO)}")

    if not face_images:
        raise HTTPException(422, "at least one texture face is required")

    surface_finish = body.get('surface_finish', 'matte')
    if surface_finish not in ('matte', 'satin', 'polished'):
        raise HTTPException(422, 'surface finish must be matte, satin, or polished')

    tile_mm = tuple(body.get("tile_mm", (600, 600)))
    floor_mm = tuple(body.get("floor_mm", (4000, 3000)))
    rotation = body.get('tile_rotation', 0)
    if rotation not in (0, 90):
        raise HTTPException(422, 'tile rotation must be 0 or 90 degrees')
    if rotation == 90:
        face_images = [cv2.rotate(face, cv2.ROTATE_90_CLOCKWISE) for face in face_images]
        tile_mm = tile_mm[::-1]
    scale = (float(body["px_per_mm"]) if "px_per_mm" in body else
             tile_sheet_scale(face_images, tile_mm, floor_mm, quad))
    reflectance = None
    if surface_finish != 'matte':
        reflectance = np.empty((max(2, round(floor_mm[1] * scale)),
                                max(2, round(floor_mm[0] * scale))), np.uint8)
    sheet = build_tile_sheet(
        face_images, tile_mm, floor_mm, scale,
        layout=body.get("layout", "grid"),
        grout_mm=float(body.get("grout_mm", 2.0)),
        grout_bgr=tuple(body.get("grout_bgr", (205, 205, 205))), reflectance=reflectance)

    mask = build_mask(base, [tuple(p) for p in quad],
                      exclude=[[tuple(p) for p in poly]
                               for poly in body.get("exclude", [])],
                      auto=auto,
                      threshold=float(body.get("threshold", 13.0)),
                      floor_boundary=floor_boundary)
    out = render(base, sheet, [tuple(p) for p in quad], mask=mask,
                 strength=float(body.get("strength", 1.0)), surface_finish=surface_finish,
                 reflectance=reflectance)
    return out, mask


@app.post("/render")
async def do_render(payload: str = Form(...),
                    photo: UploadFile = File(...),
                    faces: list[UploadFile] = File(...)):
    try:
        body = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise HTTPException(422, f"payload is not JSON: {exc}") from exc
    if not isinstance(body, dict):
        raise HTTPException(422, 'payload must be an object')
    base = _decode(await photo.read())
    face_images = [_decode(await face.read()) for face in faces]
    layers = body.get('layers')
    if layers is None:
        out, _ = await run_in_threadpool(_render_surface, base, face_images, body)
    else:
        if not isinstance(layers, list) or not 1 <= len(layers) <= 12:
            raise HTTPException(422, 'select between 1 and 12 surfaces')
        counts = [layer.get('face_count') if isinstance(layer, dict) else None for layer in layers]
        if any(type(count) is not int or count < 1 for count in counts) or sum(counts) != len(face_images):
            raise HTTPException(422, 'surface texture counts do not match the uploaded textures')
        out, offset = base.copy(), 0
        for layer, count in zip(layers, counts):
            painted, mask = await run_in_threadpool(
                _render_surface, base, face_images[offset:offset + count], layer)
            out[mask > 0] = painted[mask > 0]
            offset += count

    ok, buf = cv2.imencode(".png", out)
    if not ok:
        raise HTTPException(500, "failed to encode result")
    return Response(buf.tobytes(), media_type="image/png")


@app.post('/select-surface')
async def find_surface(payload: str = Form(...), photo: UploadFile = File(...)):
    started = time.perf_counter()
    try:
        body = json.loads(payload)
        if not isinstance(body, dict) or not 1 <= len(body.get('points', [])) <= 64:
            raise ValueError('Select between 1 and 64 points, starting inside the surface.')
        if body.get('kind', 'floor') not in {'floor', 'wall'}:
            raise ValueError('Choose a floor or wall surface.')
        base = _decode(await photo.read())
        result = await run_in_threadpool(select_surface, base, body.get('points'), body.get('labels'))
    except (ValueError, TypeError) as exc:
        raise HTTPException(422, str(exc)) from exc
    if body.get('estimate_plane') and body.get('kind', 'floor') == 'floor':
        plane = await run_in_threadpool(estimate_floor_plane, base, result['floor_boundary'])
        result['floor_boundary'] = plane.pop('floor_boundary')
        result['plane'] = plane
    return {**result, 'seconds': round(time.perf_counter() - started, 3)}


@app.post("/detect-floor")
async def find_floor(photo: UploadFile = File(...)):
    started = time.perf_counter()
    base = _decode(await photo.read())
    try:
        result = await run_in_threadpool(detect_floor, base)
        result = await run_in_threadpool(refine_floor_detection, base, result)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    plane = await run_in_threadpool(
        estimate_floor_plane, base, result["floor_boundary"])
    result["floor_boundary"] = plane.pop("floor_boundary")
    if len(result["floor_boundary"]) < 3:
        raise HTTPException(422, "No floor was detected. Use Floor outline to select it manually.")
    result["plane"] = plane
    return {**result, "seconds": round(time.perf_counter() - started, 3)}
