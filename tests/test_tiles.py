import numpy as np
from renderer.tiles import build_tile_sheet


def face(colour):
    return np.full((600, 600, 3), colour, np.uint8)


def test_sheet_pixel_size_follows_physical_size():
    sheet = build_tile_sheet([face(200)], (600, 600), (4200, 3200), 0.5)
    assert sheet.shape[:2] == (1600, 2100)   # 3200mm*0.5, 4200mm*0.5


def test_tile_count_across_width_matches_floor_width():
    ppm = 0.5
    sheet = build_tile_sheet([face(200)], (600, 600), (3000, 3000), ppm,
                             grout_mm=0)
    tile_px = int(600 * ppm)
    assert sheet.shape[1] // tile_px == 5      # 3000mm / 600mm


def test_grout_colour_survives_between_tiles():
    grout = (10, 20, 30)
    sheet = build_tile_sheet([face(240)], (600, 600), (1800, 1800), 0.5,
                             grout_mm=10, grout_bgr=grout)
    assert (sheet == np.array(grout, np.uint8)).all(axis=2).any()


def test_same_seed_gives_identical_sheet():
    faces = [face(100), face(150), face(200), face(250)]
    a = build_tile_sheet(faces, (600, 600), (3000, 3000), 0.4, seed=7)
    b = build_tile_sheet(faces, (600, 600), (3000, 3000), 0.4, seed=7)
    assert np.array_equal(a, b)


def test_different_seed_changes_face_placement():
    faces = [face(100), face(150), face(200), face(250)]
    a = build_tile_sheet(faces, (600, 600), (3000, 3000), 0.4, seed=7)
    b = build_tile_sheet(faces, (600, 600), (3000, 3000), 0.4, seed=99)
    assert not np.array_equal(a, b)


def test_brick_layout_differs_from_grid():
    faces = [face(100), face(250)]
    grid = build_tile_sheet(faces, (600, 600), (3000, 3000), 0.4, layout="grid", seed=3)
    brick = build_tile_sheet(faces, (600, 600), (3000, 3000), 0.4, layout="brick", seed=3)
    assert not np.array_equal(grid, brick)
