import io
import json

import cv2
import numpy as np
from fastapi.testclient import TestClient

from bench.app import app
from tests.fixtures import QUAD, room_with_objects

client = TestClient(app)


def photo_bytes():
    ok, buf = cv2.imencode(".png", room_with_objects())
    assert ok
    return io.BytesIO(buf.tobytes())


def face_bytes(value=235):
    ok, buf = cv2.imencode(".png", np.full((600, 600, 3), value, np.uint8))
    assert ok
    return io.BytesIO(buf.tobytes())


def payload(**over):
    body = {"quad": QUAD, "tile_mm": [600, 600], "floor_mm": [4200, 3200],
            "grout_mm": 2.0, "grout_bgr": [205, 205, 205], "layout": "grid",
            "auto": "none", "exclude": []}
    body.update(over)
    return json.dumps(body)


def test_index_page_is_served():
    r = client.get("/")
    assert r.status_code == 200
    assert "canvas" in r.text.lower()


def test_render_returns_a_png_the_same_size_as_the_photo():
    r = client.post("/render",
                    files={"photo": ("room.png", photo_bytes(), "image/png"),
                           "faces": ("f1.png", face_bytes(), "image/png")},
                    data={"payload": payload()})
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    out = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    assert out.shape == room_with_objects().shape


def test_render_actually_changes_the_floor():
    r = client.post("/render",
                    files={"photo": ("room.png", photo_bytes(), "image/png"),
                           "faces": ("f1.png", face_bytes(), "image/png")},
                    data={"payload": payload()})
    out = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    base = room_with_objects()
    assert out[770, 600].mean() > base[770, 600].mean()


def test_quad_with_wrong_point_count_is_rejected():
    r = client.post("/render",
                    files={"photo": ("room.png", photo_bytes(), "image/png"),
                           "faces": ("f1.png", face_bytes(), "image/png")},
                    data={"payload": payload(quad=[[0, 0], [10, 0], [10, 10]])})
    assert r.status_code == 422


def test_unknown_auto_mode_is_rejected():
    r = client.post("/render",
                    files={"photo": ("room.png", photo_bytes(), "image/png"),
                           "faces": ("f1.png", face_bytes(), "image/png")},
                    data={"payload": payload(auto="magic")})
    assert r.status_code == 422


def test_missing_photo_is_rejected():
    r = client.post("/render",
                    files={"faces": ("f1.png", face_bytes(), "image/png")},
                    data={"payload": payload()})
    assert r.status_code == 422
