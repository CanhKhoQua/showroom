import json

import cv2
import numpy as np
import pytest
from renderer.catalogue import Sku, load_faces, load_registry


def write_registry(tmp_path, entries):
    p = tmp_path / "registry.json"
    p.write_text(json.dumps(entries))
    return p


def test_loads_sku_with_physical_size(tmp_path):
    p = write_registry(tmp_path, {
        "marble-600": {"tile_mm": [600, 600], "faces": ["a.png", "b.png"],
                       "tiles_per_box": 4, "finish": "matte"}})
    reg = load_registry(p)
    assert reg["marble-600"].tile_mm == (600.0, 600.0)
    assert reg["marble-600"].tiles_per_box == 4
    assert reg["marble-600"].m2_per_tile == pytest.approx(0.36)


def test_missing_tile_mm_is_an_error_not_a_default(tmp_path):
    """Guessing physical size silently renders the floor at the wrong scale."""
    p = write_registry(tmp_path, {"bad": {"faces": ["a.png"], "tiles_per_box": 4,
                                     "finish": "matte"}})
    with pytest.raises(KeyError):
        load_registry(p)


def test_single_face_is_rejected_for_stone_look(tmp_path):
    """One face makes the rendered floor visibly repeat. Warn loudly."""
    p = write_registry(tmp_path, {"one": {"tile_mm": [600, 600], "faces": [],
                                     "tiles_per_box": 4, "finish": "matte"}})
    with pytest.raises(ValueError, match="at least one face"):
        load_registry(p)


def test_load_faces_reads_every_declared_file(tmp_path):
    tmp = tmp_path
    for name in ("a.png", "b.png"):
        cv2.imwrite(str(tmp / name), np.full((60, 60, 3), 200, np.uint8))
    sku = Sku("s", (600.0, 600.0), ["a.png", "b.png"], 4, "matte")
    faces = load_faces(sku, tmp)
    assert len(faces) == 2
    assert all(f.shape == (60, 60, 3) for f in faces)


def test_load_faces_fails_loudly_on_a_missing_file(tmp_path):
    tmp = tmp_path
    sku = Sku("s", (600.0, 600.0), ["nope.png"], 4, "matte")
    with pytest.raises(FileNotFoundError):
        load_faces(sku, tmp)
