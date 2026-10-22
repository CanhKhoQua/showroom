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


def test_tile_edges_have_subtle_depth_even_when_face_and_grout_colors_match():
    sheet = build_tile_sheet([face(180)], (100, 100), (200, 200), 1,
                             grout_mm=4, grout_bgr=(180, 180, 180))

    assert 160 < sheet[50, 2, 0] < sheet[50, 10, 0] == 180
    assert sheet[50, 0, 0] == 180
    assert sheet[50, 97, 0] == sheet[50, 2, 0]


def test_zero_grout_keeps_a_continuous_material_without_artificial_edge_shadows():
    sheet = build_tile_sheet([face(180)], (100, 100), (200, 200), 1, grout_mm=0)

    assert np.all(sheet == 180)


def test_reflection_mask_follows_staggered_vertical_plank_joints():
    reflectance = np.empty((240, 80), np.uint8)
    build_tile_sheet([face(200)], (200, 1200), (800, 2400), 0.1,
                     layout='brick', grout_mm=20, reflectance=reflectance)

    assert reflectance[120, 10] == 0
    assert reflectance[120, 30] == 255
    assert reflectance[180, 10] == 255
    assert reflectance[180, 30] == 0


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


def test_vertical_planks_stagger_end_joints_along_their_length():
    sheet = build_tile_sheet([face(200)], (200, 1200), (800, 2400), 0.1,
                             layout="brick", grout_mm=20, grout_bgr=(0, 0, 0))
    # At a 1200 mm end joint, the adjacent column must still be inside a plank.
    assert sheet[120, 10, 0] == 0
    assert sheet[120, 30, 0] == 200
    assert sheet[180, 10, 0] == 200
    assert sheet[180, 30, 0] == 0


def test_horizontal_planks_keep_their_half_length_row_offset():
    sheet = build_tile_sheet([face(200)], (1200, 200), (2400, 800), 0.1,
                             layout="brick", grout_mm=20, grout_bgr=(0, 0, 0))
    assert sheet[10, 120, 0] == 0
    assert sheet[30, 120, 0] == 200
    assert sheet[10, 180, 0] == 200
    assert sheet[30, 180, 0] == 0


def test_vertical_grid_keeps_aligned_end_joints():
    sheet = build_tile_sheet([face(200)], (200, 1200), (800, 2400), 0.1,
                             layout="grid", grout_mm=20, grout_bgr=(0, 0, 0))
    assert sheet[120, 10, 0] == 0
    assert sheet[120, 30, 0] == 0


def test_staggering_vertical_planks_preserves_the_face_orientation():
    plank = np.full((120, 20, 3), 200, np.uint8)
    plank[:, :10] = 50
    sheet = build_tile_sheet([plank], (200, 1200), (800, 2400), 0.1,
                             layout="brick", grout_mm=0)
    assert sheet.shape == (240, 80, 3)
    assert sheet[20, 5, 0] == 50
    assert sheet[20, 15, 0] == 200
    assert sheet[200, 5, 0] == 50
    assert sheet[200, 15, 0] == 200


def test_sheet_scale_retains_available_grain_detail_for_a_large_photo():
    from renderer.tiles import tile_sheet_scale

    plank = np.empty((960, 152, 3), np.uint8)
    scale = tile_sheet_scale([plank], (190, 1200), (2000, 2000),
                             [[0, 0], [1000, 0], [2000, 2000], [0, 2000]])
    assert scale == 0.8


def test_sheet_scale_follows_photo_resolution_instead_of_oversampling():
    from renderer.tiles import tile_sheet_scale

    scale = tile_sheet_scale([face(200)], (600, 600), (2000, 2000),
                             [[0, 0], [200, 0], [200, 200], [0, 200]])
    assert scale == 0.1


def test_sheet_scale_caps_large_floor_allocations():
    from renderer.tiles import tile_sheet_scale

    scale = tile_sheet_scale([face(200)], (600, 600), (10000, 10000),
                             [[0, 0], [10000, 0], [10000, 10000], [0, 10000]])
    assert scale == 0.4


def test_tiny_texture_keeps_the_physical_tile_count():
    from renderer.tiles import tile_sheet_scale

    tiny = np.full((1, 1, 3), 200, np.uint8)
    scale = tile_sheet_scale([tiny], (600, 600), (4200, 3000),
                             [[0, 0], [1000, 0], [1000, 800], [0, 800]])
    sheet = build_tile_sheet([tiny], (600, 600), (4200, 3000), scale,
                             grout_mm=0)
    assert sheet.shape[:2] == (10, 14)
