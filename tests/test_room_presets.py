import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from bench.app import app


@pytest.mark.parametrize("room_id, floor_point, protected_points", [
    ("living", (900, 800), [(300, 420), (1375, 570)]),
    ("bedroom", (900, 800), [(400, 300), (850, 230)]),
    ("kitchen", (800, 750), [(800, 300), (380, 310)]),
    ("bathroom", (800, 750), [(800, 280), (1250, 500)]),
])
def test_room_preset_replaces_floor_and_preserves_furniture(
        room_id, floor_point, protected_points):
    client = TestClient(app)
    presets = client.get("/static/rooms/presets.json")
    assert presets.status_code == 200
    room = next(room for room in presets.json() if room["id"] == room_id)
    photo = client.get(room["image"])
    assert photo.status_code == 200
    assert client.get(room["thumbnail"]).status_code == 200
    base = cv2.imdecode(np.frombuffer(photo.content, np.uint8), cv2.IMREAD_COLOR)
    _, face = cv2.imencode(".png", np.full((64, 64, 3), (40, 90, 180), np.uint8))
    payload = {"quad": room["quad"], "floor_boundary": room["floor_boundary"],
               "exclude": room["exclude"], "tile_mm": [600, 600],
               "floor_mm": [value * 1000 for value in room["floor_m"]],
               "auto": "none"}
    response = client.post("/render", data={"payload": json.dumps(payload)},
                           files={"photo": ("room.jpg", photo.content, "image/jpeg"),
                                  "faces": ("tile.png", face.tobytes(), "image/png")})
    assert response.status_code == 200
    rendered = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
    assert rendered.shape == base.shape
    x, y = floor_point
    assert np.max(np.abs(rendered[y, x].astype(int) - base[y, x])) > 20
    for x, y in protected_points:
        assert np.array_equal(rendered[y, x], base[y, x])
