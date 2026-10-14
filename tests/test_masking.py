import numpy as np
from renderer.masking import build_mask, drop_small_components
from tests.fixtures import (BARE_FLOOR, CABINET_FRONT, POT_BODY, QUAD,
                            kept_pct, room_with_objects, tiled_pct)


def test_plain_quad_covers_the_whole_floor_and_nothing_above_it():
    base = room_with_objects()
    m = build_mask(base, QUAD)
    assert tiled_pct(m, BARE_FLOOR) == 100.0
    assert m[100, 650] == 0


def test_hand_drawn_exclusion_removes_exactly_that_polygon():
    base = room_with_objects()
    poly = [(820, 500), (1045, 500), (1045, 694), (820, 694)]
    m = build_mask(base, QUAD, exclude=[poly])
    assert kept_pct(m, CABINET_FRONT) == 100.0
    assert tiled_pct(m, BARE_FLOOR) == 100.0     # untouched elsewhere


def test_concave_floor_boundary_keeps_the_wall_inside_its_notch():
    base = np.full((300, 400, 3), 120, np.uint8)
    quad = [(20, 20), (380, 20), (380, 280), (20, 280)]
    boundary = [(20, 20), (180, 20), (180, 140), (380, 140),
                (380, 280), (20, 280)]

    mask = build_mask(base, quad, floor_boundary=boundary)

    assert mask[80, 300] == 0
    assert mask[80, 100] == 255
    assert mask[220, 300] == 255
    assert mask[10, 100] == 0


def test_floor_boundary_still_preserves_objects_and_clips_to_the_plane():
    base = np.full((300, 400, 3), 120, np.uint8)
    quad = [(20, 20), (380, 20), (380, 280), (20, 280)]
    boundary = [(0, 0), (400, 0), (400, 300), (0, 300)]
    object_outline = [(80, 170), (130, 170), (130, 230), (80, 230)]

    mask = build_mask(base, quad, exclude=[object_outline],
                      floor_boundary=boundary)

    assert mask[200, 100] == 0
    assert mask[200, 200] == 255
    assert mask[10, 100] == 0


def test_colour_deviation_catches_the_chromatic_cabinet():
    base = room_with_objects()
    m = build_mask(base, QUAD, auto="colour-deviation", threshold=13.0)
    assert kept_pct(m, CABINET_FRONT) > 90.0


def test_colour_deviation_leaves_shadowed_floor_alone():
    """Lightness is weighted down so a cast shadow is not read as an object."""
    base = room_with_objects()
    m = build_mask(base, QUAD, auto="colour-deviation", threshold=13.0)
    assert tiled_pct(m, BARE_FLOOR) > 99.0


def test_morphology_must_not_swallow_objects():
    """Regression. Closing the FLOOR mask let stray floor-looking pixels inside
    the cabinet dilate outward and merge, taking the cabinet from 88% correctly
    excluded to 9%. Morphology must act on the objects instead."""
    base = room_with_objects()
    m = build_mask(base, QUAD, auto="colour-deviation", threshold=13.0)
    assert kept_pct(m, CABINET_FRONT) > 90.0


def test_near_neutral_object_is_documented_as_out_of_reach():
    """Spec section 5: no threshold keeps a near-grey object AND the floor.
    This test pins that limitation so nobody 'fixes' it by lowering the
    threshold and silently shredding the floor."""
    base = room_with_objects()
    usable = build_mask(base, QUAD, auto="colour-deviation", threshold=13.0)
    aggressive = build_mask(base, QUAD, auto="colour-deviation", threshold=6.0)
    assert kept_pct(usable, POT_BODY) < 50.0          # missed, as expected
    assert kept_pct(aggressive, POT_BODY) > 90.0      # caught, but...
    assert tiled_pct(aggressive, BARE_FLOOR) < 60.0   # ...floor is destroyed


def test_drop_small_components_removes_specks_and_keeps_blobs():
    m = np.zeros((100, 100), np.uint8)
    m[10:14, 10:14] = 255          # 16 px speck
    m[50:80, 50:80] = 255          # 900 px blob
    out = drop_small_components(m, 400)
    assert out[12, 12] == 0
    assert out[60, 60] == 255
