import numpy as np
import pytest
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


def test_soft_clip_compresses_above_the_knee_rather_than_clamping():
    """The discriminating test. Every other soft_clip test here also passes
    against a plain np.clip(v, 0, 255), so without this one the roll-off has
    no regression cover at all -- and losing the roll-off returns the render
    to flat blown highlights with the tile veining erased.
    """
    # Just above the knee (0.80 * 255 = 204): compressed, but not clamped.
    above_knee = float(soft_clip(np.array([220.0], np.float32))[0])
    assert 204.0 < above_knee < 220.0

    # Far above white: a clamp yields exactly 255; the roll-off asymptotes
    # toward it and must stay strictly below.
    far = float(soft_clip(np.array([400.0], np.float32))[0])
    assert far < 254.9
    assert far > above_knee


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


def test_uniform_floor_light_ignores_bright_excluded_pixels():
    base = np.full((240, 360, 3), 255, np.uint8)
    mask = np.zeros(base.shape[:2], np.uint8)
    mask[50:220, 30:330] = 255
    mask[90:170, 150:210] = 0
    base[mask > 0] = 120

    gain = lighting_gain(base, mask)

    assert np.allclose(gain[mask > 0], 1.0, atol=1e-4)


def test_lighting_retains_a_narrow_shadow_without_old_floor_grain():
    height, width = 320, 480
    y, x = np.mgrid[:height, :width]
    grain = 18 * np.cos(x * np.pi / 2) * np.cos(y * np.pi / 2)
    shadow = 0.5 * np.exp(-((y - 160) / 6) ** 2)
    gray = np.clip((150 + grain) * (1 - shadow), 0, 255).astype(np.uint8)
    base = np.repeat(gray[..., None], 3, axis=2)
    mask = np.full((height, width), 255, np.uint8)

    gain = lighting_gain(base, mask)

    assert gain[160, 240] < 0.85 * gain[100, 240]
    assert gain[40:100, 80:400].std() < 0.01


def test_lighting_gain_preserves_deep_shadow_irradiance():
    base = np.full((160, 400, 3), 200, np.uint8)
    base[:, 240:] = 80
    mask = np.full(base.shape[:2], 255, np.uint8)

    gain = lighting_gain(base, mask, sigma=12)

    # sRGB 80 and 200 represent linear light levels 0.0802 and 0.5776.
    assert 0.12 < gain[80, 350] / gain[80, 80] < 0.16


def test_lighting_gain_keeps_contact_shadow_depth():
    y = np.arange(240, dtype=np.float32)[:, None]
    gray = 150 * (1 - 0.5 * np.exp(-((y - 120) / 6) ** 2))
    base = np.repeat(np.repeat(gray[..., None], 3, axis=2), 320, axis=1)
    mask = np.full(base.shape[:2], 255, np.uint8)

    gain = lighting_gain(base.astype(np.uint8), mask)

    assert gain[120, 160] / gain[60, 160] < 0.40


@pytest.mark.parametrize("color", [(235, 235, 235), (32, 110, 235), (20, 40, 60)])
def test_render_preserves_material_color_in_uniform_light(color):
    base = np.full((100, 140, 3), 120, np.uint8)
    sheet = np.full((40, 60, 3), color, np.uint8)
    quad = [(10, 10), (130, 10), (130, 90), (10, 90)]

    out = render(base, sheet, quad)

    assert np.allclose(out[50, 70], color, atol=1)


def test_render_blends_partial_coverage_in_linear_light():
    base = np.zeros((80, 100, 3), np.uint8)
    sheet = np.full((40, 60, 3), 255, np.uint8)
    mask = np.full(base.shape[:2], 128, np.uint8)
    quad = [(0, 0), (100, 0), (100, 80), (0, 80)]

    out = render(base, sheet, quad, mask=mask, strength=0)

    # Half the light from white is approximately sRGB 188, not sRGB 128.
    assert np.allclose(out[40, 50], 188, atol=2)


def test_render_resamples_texture_edges_in_linear_light():
    base = np.full((20, 20, 3), 120, np.uint8)
    sheet = np.zeros((2, 2, 3), np.uint8)
    sheet[:, 1] = 255
    quad = [(0, 0), (4, 0), (4, 4), (0, 4)]

    out = render(base, sheet, quad, feather=0)

    # Photo x=1 maps halfway between black and white texture pixels.
    assert np.allclose(out[2, 1], 188, atol=1)


def test_render_is_invariant_to_unselected_photo_padding():
    y, x = np.mgrid[:120, :180]
    base = np.repeat((80 + x // 2 + y // 4).astype(np.uint8)[..., None], 3, axis=2)
    mask = np.zeros(base.shape[:2], np.uint8)
    mask[25:105, 35:165] = 255
    quad = [(35, 25), (165, 25), (165, 105), (35, 105)]
    sheet_y, sheet_x = np.mgrid[:60, :90]
    sheet = np.stack((40 + sheet_x, 80 + sheet_y, 120 + sheet_x), axis=2).astype(np.uint8)
    offset_y, offset_x = 150, 90
    padded = np.pad(base, ((offset_y, 210), (offset_x, 260), (0, 0)), constant_values=255)
    padded_mask = np.pad(mask, ((offset_y, 210), (offset_x, 260)))
    padded_quad = [(x + offset_x, y + offset_y) for x, y in quad]

    out = render(base, sheet, quad, mask=mask)
    padded_out = render(padded, sheet, padded_quad, mask=padded_mask)

    assert np.array_equal(out, padded_out[offset_y:offset_y + 120,
                                          offset_x:offset_x + 180])


def test_render_empty_mask_returns_original_pixels():
    base = np.full((80, 100, 3), 120, np.uint8)
    mask = np.zeros(base.shape[:2], np.uint8)
    sheet = np.full((40, 60, 3), 235, np.uint8)

    out = render(base, sheet, [(0, 0), (100, 0), (100, 80), (0, 80)], mask=mask)

    assert np.array_equal(out, base)


def test_render_preserves_every_unselected_pixel_at_mask_boundaries():
    base = np.full((120, 160, 3), 120, np.uint8)
    mask = np.zeros(base.shape[:2], np.uint8)
    mask[30:110, 20:150] = 255
    mask[60:80, 70:90] = 0
    sheet = np.full((80, 130, 3), 210, np.uint8)
    quad = [(20, 30), (150, 30), (150, 110), (20, 110)]

    out = render(base, sheet, quad, mask=mask)

    assert np.array_equal(out[mask == 0], base[mask == 0])


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
