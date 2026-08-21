import numpy as np
from renderer.compose import lighting_gain, render, soft_clip
from renderer.geometry import quad_mask_of
from renderer.masking import build_mask
from renderer.tiles import build_tile_sheet
from tests.fixtures import BARE_FLOOR, QUAD, room_with_objects


def test_soft_clip_never_exceeds_255():
    v = np.linspace(0, 900, 500, dtype=np.float32)
    assert soft_clip(v.copy()).max() <= 255.0


def test_soft_clip_is_monotonic():
    v = np.linspace(0, 900, 500, dtype=np.float32)
    out = soft_clip(v.copy())
    assert (np.diff(out) >= -1e-4).all()


def test_soft_clip_leaves_midtones_alone():
    v = np.array([0.0, 100.0, 150.0], np.float32)
    assert np.allclose(soft_clip(v.copy()), v, atol=1e-3)


def test_lighting_gain_is_centred_on_one_inside_the_mask():
    base = room_with_objects()
    mask = quad_mask_of(base.shape, QUAD)
    gain = lighting_gain(base, mask)
    assert 0.85 < float(gain[mask > 127].mean()) < 1.15


def test_lighting_gain_is_darker_where_the_shadow_is():
    base = room_with_objects()
    mask = quad_mask_of(base.shape, QUAD)
    gain = lighting_gain(base, mask)
    assert gain[770, 1010] < gain[770, 300]     # shadow vs window side


def test_render_leaves_pixels_outside_the_mask_untouched():
    base = room_with_objects()
    sheet = build_tile_sheet([np.full((600, 600, 3), 235, np.uint8)],
                             (600, 600), (4200, 3200), 0.35)
    mask = build_mask(base, QUAD)
    out = render(base, sheet, QUAD, mask=mask)
    assert np.array_equal(out[100, :, :], base[100, :, :])   # above far edge


def test_render_changes_the_floor():
    base = room_with_objects()
    sheet = build_tile_sheet([np.full((600, 600, 3), 235, np.uint8)],
                             (600, 600), (4200, 3200), 0.35)
    out = render(base, sheet, QUAD, mask=build_mask(base, QUAD))
    assert not np.array_equal(out[BARE_FLOOR], base[BARE_FLOOR])
    assert out[BARE_FLOOR].mean() > base[BARE_FLOOR].mean()   # cream over grey


def test_excluded_region_is_not_painted_over():
    base = room_with_objects()
    sheet = build_tile_sheet([np.full((600, 600, 3), 235, np.uint8)],
                             (600, 600), (4200, 3200), 0.35)
    poly = [(820, 500), (1045, 500), (1045, 694), (820, 694)]
    out = render(base, sheet, QUAD, mask=build_mask(base, QUAD, exclude=[poly]))
    assert np.array_equal(out[620:680, 850:1010], base[620:680, 850:1010])
