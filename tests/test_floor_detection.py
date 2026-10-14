import cv2
import numpy as np
import pytest

from renderer.masking import build_mask


def test_logits_select_floor_at_the_photo_aspect_ratio():
    from renderer.floor_detection import floor_mask_from_logits

    logits = np.zeros((6, 8, 6), np.float32)
    logits[0] = 2
    logits[3, :, 3:] = 5
    logits[5, 4:, 3:] = 8

    mask = floor_mask_from_logits(logits, (300, 400))

    assert mask.shape == (400, 300)
    assert mask.dtype == np.uint8
    assert mask[100, 50] == 0
    assert mask[100, 250] == 255
    assert mask[350, 250] == 0


def test_model_input_keeps_the_photo_aspect_ratio():
    from renderer.floor_detection import model_input_size

    assert model_input_size(3024, 4032) == (480, 640)
    assert model_input_size(4032, 3024) == (640, 480)
    assert model_input_size(1400, 933) == (640, 416)


def test_coarse_floor_edge_snaps_to_the_real_edge_in_the_photo():
    """The model sees 1/4 resolution; its edge can sit a cell off the wall."""
    from renderer.floor_detection import floor_mask_from_logits

    guide = np.full((256, 256, 3), (60, 70, 120), np.uint8)   # skirting board
    guide[:, 100:] = (170, 190, 205)                          # carpet
    noise = np.random.default_rng(1).normal(0, 6, guide.shape)
    guide = np.clip(guide + noise, 0, 255).astype(np.uint8)
    logits = np.zeros((4, 32, 32), np.float32)
    logits[0] = 2
    logits[3, :, 14:] = 6                                     # edge at x ~ 112

    plain = floor_mask_from_logits(logits, (256, 256))
    snapped = floor_mask_from_logits(logits, (256, 256), guide)

    plain_edge = np.argmax(plain[128] > 0)
    snapped_edge = np.argmax(snapped[128] > 0)
    assert abs(plain_edge - 100) > 6
    assert abs(snapped_edge - 100) <= 2
    rows = np.argmax(snapped > 0, axis=1)
    assert rows.max() - rows.min() <= 2


def test_detected_outline_preserves_notches_and_object_holes_at_photo_scale():
    from renderer.floor_detection import outlines_from_mask

    mask = np.zeros((200, 300), np.uint8)
    points = [[20, 20], [130, 20], [130, 100], [280, 100],
              [280, 180], [20, 180]]
    cv2.fillPoly(mask, [np.int32(points)], 255)
    mask[120:160, 60:100] = 0
    mask[5:8, 290:293] = 255

    result = outlines_from_mask(mask, (600, 400))
    base = np.zeros((400, 600, 3), np.uint8)
    restored = build_mask(base, [(0, 0), (599, 0), (599, 399), (0, 399)],
                          floor_boundary=result["floor_boundary"],
                          exclude=result["exclude"])

    assert len(result["floor_boundary"]) >= 6
    assert restored[100, 400] == 0
    assert restored[280, 160] == 0
    assert restored[300, 440] == 255
    assert restored[10, 585] == 0


def test_outline_reaches_the_photo_edges_without_leaving_a_border():
    from renderer.floor_detection import outlines_from_mask

    mask = np.zeros((512, 512), np.uint8)
    mask[300:, :] = 255
    result = outlines_from_mask(mask, (3024, 4032))

    assert max(p[0] for p in result["floor_boundary"]) == 3023
    assert max(p[1] for p in result["floor_boundary"]) == 4031


def test_no_floor_returns_a_clear_manual_selection_fallback():
    from renderer.floor_detection import outlines_from_mask

    with pytest.raises(ValueError, match="No floor"):
        outlines_from_mask(np.zeros((512, 512), np.uint8), (600, 400))


def test_pinholes_in_the_floor_are_not_reported_as_objects():
    from renderer.floor_detection import outlines_from_mask

    mask = np.zeros((1024, 768), np.uint8)
    mask[500:, :] = 255
    mask[700:704, 300:304] = 0       # filter speckle, a few photo pixels
    mask[800:860, 400:460] = 0       # a real object on the floor

    result = outlines_from_mask(mask, (3024, 4032))

    assert len(result["exclude"]) == 1
