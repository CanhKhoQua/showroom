import io
import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pillow_heif import register_heif_opener

from bench.app import app
from tests.fixtures import QUAD, room_with_objects

client = TestClient(app)
register_heif_opener()


def photo_bytes():
    ok, buf = cv2.imencode(".png", room_with_objects())
    assert ok
    return io.BytesIO(buf.tobytes())


def face_bytes(value=235):
    ok, buf = cv2.imencode(".png", np.full((600, 600, 3), value, np.uint8))
    assert ok
    return io.BytesIO(buf.tobytes())


def heic_bytes():
    image = Image.new("RGB", (64, 48), (15, 30, 220))
    image.paste((20, 200, 50), (32, 0, 64, 48))
    exif = Image.Exif()
    exif[274] = 6
    raw = io.BytesIO()
    image.save(raw, format="HEIF", quality=100, exif=exif.tobytes())
    return raw.getvalue()


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


def test_render_combines_independent_surface_materials_and_keeps_objects():
    base = np.full((100, 100, 3), 120, np.uint8)
    _, photo = cv2.imencode('.png', base)
    _, red = cv2.imencode('.png', np.full((30, 30, 3), (20, 20, 220), np.uint8))
    _, blue = cv2.imencode('.png', np.full((30, 30, 3), (220, 20, 20), np.uint8))
    layers = [
        {'quad': [[0, 0], [45, 0], [45, 99], [0, 99]], 'face_count': 1,
         'exclude': [[[10, 40], [30, 40], [30, 60], [10, 60]]]},
        {'quad': [[55, 0], [99, 0], [99, 99], [55, 99]], 'face_count': 1},
    ]
    response = client.post('/render', data={'payload': json.dumps({'layers': layers})},
        files=[('photo', ('room.png', photo.tobytes(), 'image/png')),
               ('faces', ('red.png', red.tobytes(), 'image/png')),
               ('faces', ('blue.png', blue.tobytes(), 'image/png'))])
    assert response.status_code == 200, response.text[:200]
    out = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
    assert out[75, 20, 2] > out[75, 20, 0] + 100
    assert out[75, 75, 0] > out[75, 75, 2] + 100
    assert np.array_equal(out[50, 20], base[50, 20])
    assert np.array_equal(out[:, 48:52], base[:, 48:52])


def test_render_rejects_a_layer_texture_count_mismatch():
    response = client.post('/render', data={'payload': json.dumps({'layers': [
        {'quad': QUAD, 'face_count': 2}]})},
        files={'photo': ('room.png', photo_bytes(), 'image/png'),
               'faces': ('face.png', face_bytes(), 'image/png')})
    assert response.status_code == 422
    assert 'texture' in response.json()['detail'].lower()


def test_unknown_surface_finish_is_rejected():
    response = client.post('/render', data={'payload': payload(surface_finish='mirror')},
        files={'photo': ('room.png', photo_bytes(), 'image/png'),
               'faces': ('face.png', face_bytes(), 'image/png')})

    assert response.status_code == 422
    assert 'finish' in response.json()['detail'].lower()


def test_render_api_applies_each_layers_own_surface_finish():
    y, x = np.mgrid[:160, :360]
    spot = np.exp(-((x % 180 - 90) / 18) ** 2 - ((y - 80) / 18) ** 2)
    base = np.repeat((100 + 90 * spot).astype(np.uint8)[..., None], 3, axis=2)
    _, photo = cv2.imencode('.png', base)
    _, texture = cv2.imencode('.png', np.full((40, 40, 3), 60, np.uint8))
    layers = [dict(quad=[[start, 0], [start + 179, 0], [start + 179, 159], [start, 159]],
                   face_count=1, grout_mm=0, surface_finish=finish)
              for start, finish in ((0, 'matte'), (180, 'polished'))]
    response = client.post('/render', data={'payload': json.dumps({'layers': layers})},
        files=[('photo', ('room.png', photo.tobytes(), 'image/png')),
               ('faces', ('matte.png', texture.tobytes(), 'image/png')),
               ('faces', ('polished.png', texture.tobytes(), 'image/png'))])

    assert response.status_code == 200
    out = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
    assert float(out[80, 270].mean()) > float(out[80, 90].mean()) + 10


def test_polished_tiles_do_not_turn_matte_grout_into_a_reflection():
    y, x = np.mgrid[:200, :320]
    spot = np.exp(-((x - 200) / 24) ** 2 - ((y - 100) / 24) ** 2)
    base = np.repeat((100 + 90 * spot).astype(np.uint8)[..., None], 3, axis=2)
    _, photo = cv2.imencode('.png', base)
    outputs = []
    for finish in ('matte', 'polished'):
        response = client.post('/render',
            data={'payload': payload(quad=[[0, 0], [319, 0], [319, 199], [0, 199]],
                tile_mm=[100, 100], floor_mm=[320, 200], px_per_mm=1,
                grout_mm=4, grout_bgr=[30, 30, 30], surface_finish=finish)},
            files={'photo': ('room.png', photo.tobytes(), 'image/png'),
                   'faces': ('face.png', face_bytes(120), 'image/png')})
        assert response.status_code == 200
        outputs.append(cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR))

    assert np.allclose(outputs[0][100, 200], outputs[1][100, 200], atol=1)
    assert outputs[1][90, 190].mean() > outputs[0][90, 190].mean() + 10


def test_tile_direction_turns_grain_and_preserves_rectangular_scale():
    _, photo = cv2.imencode('.png', np.full((100, 100, 3), 120, np.uint8))
    face = np.full((60, 20, 3), 220, np.uint8)
    face[:, :10] = 40
    _, texture = cv2.imencode('.png', face)
    outputs = []
    for rotation in (0, 90):
        body = {'quad': [[0, 0], [99, 0], [99, 99], [0, 99]], 'tile_mm': [20, 60],
                'floor_mm': [100, 100], 'tile_rotation': rotation, 'grout_mm': 0}
        response = client.post('/render', data={'payload': json.dumps(body)},
            files={'photo': ('room.png', photo.tobytes(), 'image/png'),
                   'faces': ('grain.png', texture.tobytes(), 'image/png')})
        assert response.status_code == 200
        outputs.append(cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR))
    assert abs(float(outputs[0][30, 5].mean()) - float(outputs[0][30, 15].mean())) > 100
    assert abs(float(outputs[1][5, 30].mean()) - float(outputs[1][15, 30].mean())) > 100
    assert abs(float(outputs[1][5, 5].mean()) - float(outputs[1][5, 15].mean())) < 2


def test_surface_selection_passes_original_photo_points_and_exclusions(monkeypatch):
    from bench import app as module

    def select(base, points, labels):
        assert base.shape == room_with_objects().shape
        assert points == [[300, 700], [900, 600]]
        assert labels == [1, 0]
        return {'floor_boundary': [[0, 500], [1300, 500], [1300, 899], [0, 899]], 'exclude': []}

    monkeypatch.setattr(module, 'select_surface', select, raising=False)
    response = client.post('/select-surface',
        files={'photo': ('room.png', photo_bytes(), 'image/png')},
        data={'payload': json.dumps({'points': [[300, 700], [900, 600]], 'labels': [1, 0], 'kind': 'wall'})})
    assert response.status_code == 200
    assert len(response.json()['floor_boundary']) == 4
    assert 'plane' not in response.json()


@pytest.mark.parametrize('body', [[], {'points': [], 'labels': []},
    {'points': [[300, 700]], 'labels': [0]},
    {'points': [[float('inf'), 700]], 'labels': [1]}])
def test_invalid_surface_prompts_are_rejected(body):
    response = client.post('/select-surface',
        files={'photo': ('room.png', photo_bytes(), 'image/png')},
        data={'payload': json.dumps(body)})
    assert response.status_code == 422


@pytest.mark.parametrize("filename, media_types", [
    ("styles.css", {"text/css"}),
    ("app.js", {"text/javascript", "application/javascript"}),
])
def test_browser_assets_are_served(filename, media_types):
    r = client.get(f"/static/{filename}")
    assert r.status_code == 200
    assert r.headers["content-type"].split(";")[0] in media_types
    assert r.content


def test_demo_room_is_served_as_a_useful_colour_photo():
    r = client.get("/static/demo-room.jpg")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/jpeg"
    image = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_UNCHANGED)
    assert image is not None
    assert image.ndim == 3 and image.shape[2] == 3
    assert image.shape[0] >= 600 and image.shape[1] >= 900


def test_missing_static_asset_returns_not_found():
    r = client.get("/static/missing-asset.png")
    assert r.status_code == 404


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


def test_render_accepts_a_six_point_floor_outline_without_painting_its_notch():
    base = np.full((300, 400, 3), 120, np.uint8)
    _, encoded = cv2.imencode(".png", base)
    boundary = [[20, 20], [180, 20], [180, 140], [380, 140],
                [380, 280], [20, 280]]
    r = client.post("/render",
                    files={"photo": ("room.png", encoded.tobytes(), "image/png"),
                           "faces": ("f1.png", face_bytes(), "image/png")},
                    data={"payload": payload(
                        quad=[[20, 20], [380, 20], [380, 280], [20, 280]],
                        floor_boundary=boundary)})

    assert r.status_code == 200
    out = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    assert np.array_equal(out[40:120, 220:360], base[40:120, 220:360])
    assert out[220, 300].mean() > 200
    assert out[80, 100].mean() > 200


def test_incomplete_floor_outline_is_rejected():
    r = client.post("/render",
                    files={"photo": ("room.png", photo_bytes(), "image/png"),
                           "faces": ("f1.png", face_bytes(), "image/png")},
                    data={"payload": payload(floor_boundary=[[0, 0], [10, 10]])})
    assert r.status_code == 422
    assert "3 points" in r.json()["detail"]


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


def test_heic_upload_becomes_an_upright_colour_png():
    r = client.post("/image", files={"file": ("phone.HEIC", heic_bytes(),
                                             "image/heic")})
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    out = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    assert out.shape == (64, 48, 3)
    assert np.abs(out[8, 24].astype(int) - [220, 30, 15]).max() < 10
    assert np.abs(out[56, 24].astype(int) - [50, 200, 20]).max() < 10


def test_image_upload_preserves_a_png():
    r = client.post("/image", files={"file": ("room.png", photo_bytes(),
                                             "image/png")})
    assert r.status_code == 200
    out = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    assert np.array_equal(out, room_with_objects())


def test_floor_detection_runs_the_real_model_on_the_demo_room():
    from bench.app import STATIC

    r = client.post("/detect-floor", files={"photo": (
        "room.jpg", (STATIC / "demo-room.jpg").read_bytes(), "image/jpeg")})

    assert r.status_code == 200
    result = r.json()
    assert len(result["floor_boundary"]) >= 3
    assert result["seconds"] > 0
    assert result["exclude"] == [] or all(len(poly) >= 3 for poly in result["exclude"])
    base = cv2.imread(str(STATIC / "demo-room.jpg"))
    from renderer.masking import build_mask
    h, w = base.shape[:2]
    mask = build_mask(base, [(0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)],
                      floor_boundary=result["floor_boundary"],
                      exclude=result["exclude"])
    assert mask[100, 400] == 0
    assert 0.05 < (mask > 0).mean() < 0.6


def test_floor_detection_suggests_a_perspective_plane_around_the_outline():
    from bench.app import STATIC

    r = client.post("/detect-floor", files={"photo": (
        "room.jpg", (STATIC / "demo-room.jpg").read_bytes(), "image/jpeg")})

    plane = r.json()["plane"]
    quad = np.float32(plane["quad"])
    assert quad.shape == (4, 2)
    assert plane["source"] in {"vertical-lines", "default"}
    assert all(0 < v <= 31000 for v in plane["floor_mm"])
    # Perspective: the far edge is narrower on screen than the near edge.
    assert np.linalg.norm(quad[1] - quad[0]) < np.linalg.norm(quad[2] - quad[3])
    for point in r.json()["floor_boundary"]:
        assert cv2.pointPolygonTest(quad, tuple(map(float, point)), True) >= -1


def test_floor_detection_rejects_an_invalid_photo():
    r = client.post("/detect-floor", files={"photo": (
        "broken.heic", b"not an image", "image/heic")})
    assert r.status_code == 422
    assert "decode" in r.json()["detail"]


@pytest.mark.parametrize("raw", [b"not an image", b""])
def test_invalid_heic_upload_returns_a_useful_error(raw):
    r = client.post("/image", files={"file": ("broken.heic", raw,
                                             "image/heic")})
    assert r.status_code == 422
    assert "decode" in r.json()["detail"]


def test_render_accepts_an_original_heic_photo():
    r = client.post("/render",
                    files={"photo": ("room.heic", heic_bytes(), "image/heic"),
                           "faces": ("f1.png", face_bytes(), "image/png")},
                    data={"payload": payload(
                        quad=[[0, 40], [47, 40], [47, 63], [0, 63]])})
    assert r.status_code == 200
    out = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    assert out.shape == (64, 48, 3)
