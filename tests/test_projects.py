import copy

import pytest
from fastapi.testclient import TestClient

from bench.app import app


PNG = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a6ioAAAAASUVORK5CYII="


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("SHOWROOM_DATA_DIR", str(tmp_path))
    return TestClient(app)


def project():
    workspace = {
        "name": "Uploaded room", "photo": PNG, "room_id": None,
        "quad": [[10, 10], [90, 10], [90, 90], [10, 90]],
        "boundary": [], "boundary_closed": False, "auto_plane": False,
        "exclusions": [[[20, 20], [30, 20], [25, 30]]],
        "ai_exclusions": [], "material_id": None, "custom_faces": [PNG],
        "settings": {"floorW": "3", "floorD": "3", "tileW": "600",
                     "tileD": "600", "layout": "brick", "groutMm": "2",
                     "groutColour": "#aaa99e", "auto": False},
    }
    return {"name": "Townhouse", "schema_version": 1, "workspace": workspace,
            "quote": {"rooms": [{"name": "Bedroom", "size_m": [3, 3],
                                  "sku_name": "WOOD", "workspace": workspace}],
                      "skus": {"WOOD": {"tile_mm": [600, 600],
                                        "tiles_per_box": 4,
                                        "price_per_box": 240000,
                                        "waste_pct": 8}}}}


def test_project_preserves_photos_geometry_custom_faces_and_quote(client):
    response = client.post("/projects", json=project())
    assert response.status_code == 201
    saved = response.json()
    assert len(saved["id"]) == 43
    assert saved["revision"] == 1
    loaded = client.get(f'/projects/{saved["id"]}')
    assert loaded.status_code == 200
    for field, expected in project().items():
        assert loaded.json()[field] == expected


def test_another_browser_can_open_the_same_project(client):
    saved = client.post("/projects", json=project()).json()
    other = TestClient(app)
    loaded = other.get(f'/projects/{saved["id"]}').json()
    assert loaded["name"] == "Townhouse"
    assert loaded["quote"]["rooms"][0]["name"] == "Bedroom"
    request = {**loaded["quote"], "dimensions_confirmed": True}
    assert other.post("/quote", json=request).json()["total"] == 1680000


def test_saved_project_is_not_held_in_process_memory(client):
    saved = client.post("/projects", json=project()).json()
    with TestClient(app) as restarted:
        loaded = restarted.get(f'/projects/{saved["id"]}').json()
    assert loaded["workspace"]["photo"] == PNG
    assert loaded["quote"] == saved["quote"]


def test_update_increments_revision_and_preserves_project_id(client):
    saved = client.post("/projects", json=project()).json()
    changed = project()
    changed["name"] = "Revised townhouse"
    response = client.put(f'/projects/{saved["id"]}',
                          json={**changed, "revision": saved["revision"]})
    assert response.status_code == 200
    assert response.json()["revision"] == 2
    loaded = client.get(f'/projects/{saved["id"]}').json()
    assert loaded["name"] == "Revised townhouse"
    assert loaded["id"] == saved["id"]


def test_stale_browser_cannot_overwrite_another_browsers_changes(client):
    saved = client.post("/projects", json=project()).json()
    path = f'/projects/{saved["id"]}'
    first = {**project(), "revision": 1, "name": "Device A"}
    second = {**project(), "revision": 1, "name": "Device B"}
    assert client.put(path, json=first).status_code == 200
    assert client.put(path, json=second).status_code == 409
    assert client.get(path).json()["name"] == "Device A"


def test_unknown_project_returns_not_found(client):
    path = "/projects/" + "a" * 43
    assert client.get(path).status_code == 404
    assert client.put(path, json={**project(), "revision": 1}).status_code == 404


def test_projects_have_distinct_unlisted_links(client):
    first = client.post("/projects", json=project()).json()
    second = client.post("/projects", json=project()).json()
    assert first["id"] != second["id"]
    assert client.get("/projects").status_code == 405


def test_empty_quote_and_unfinished_outline_can_be_saved(client):
    body = project()
    body["quote"] = {"rooms": [], "skus": {}}
    body["workspace"]["quad"] = [[10, 10]]
    assert client.post("/projects", json=body).status_code == 201


def test_missing_product_code_is_rejected_before_saving(client):
    body = project()
    body["quote"]["skus"] = {}
    assert client.post("/projects", json=body).status_code == 422


@pytest.mark.parametrize("image", ["https://example.com/photo.jpg",
                                    "data:text/html;base64,PGgxPkhlbGxvPC9oMT4=",
                                    "data:image/svg+xml;base64,PHN2Zz48L3N2Zz4="])
def test_project_images_must_be_embedded_raster_images(client, image):
    body = copy.deepcopy(project())
    body["workspace"]["photo"] = image
    assert client.post("/projects", json=body).status_code == 422


def test_non_finite_floor_coordinates_are_rejected(client):
    body = copy.deepcopy(project())
    body["workspace"]["quad"][0] = ["NaN", 10]
    assert client.post("/projects", json=body).status_code == 422
