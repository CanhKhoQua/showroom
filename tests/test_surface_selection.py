import importlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from renderer.masking import build_mask


@pytest.fixture
def selector(monkeypatch):
    module = importlib.import_module("renderer.surface_selection")

    class Encoder:
        def __init__(self):
            self.inputs = []

        def run(self, _, inputs):
            self.inputs.append(inputs["image"].copy())
            return [np.array([len(self.inputs)], np.float32)]

    class Decoder:
        def __init__(self):
            self.inputs = []
            self.scores = np.float32([[0.9]])
            self.masks = np.full((1, 1, 256, 256), -10, np.float32)
            self.masks[:, :, 128:, :] = 10

        def run(self, _, inputs):
            self.inputs.append(inputs)
            return [self.scores, self.masks]

    encoder, decoder = Encoder(), Decoder()
    monkeypatch.setattr(module, "_sessions", lambda: (encoder, decoder))
    monkeypatch.setattr(module, "_embedding", None)
    return module, encoder, decoder


def test_encoder_uses_rgb_normalization_aspect_ratio_and_zero_padding(selector):
    module, encoder, decoder = selector
    image = np.full((80, 60, 3), [10, 20, 30], np.uint8)

    result = module.select_surface(image, [[30, 70]], [1])

    pixels = encoder.inputs[0]
    assert pixels.shape == (1, 3, 1024, 1024)
    assert pixels.dtype == np.float32
    np.testing.assert_allclose(pixels[0, :, 0, 0], [-1.6041614, -1.6855743, -1.6301525])
    assert np.all(pixels[:, :, :, 768:] == 0)
    np.testing.assert_array_equal(decoder.inputs[0]["point_coords"], [[[384, 896]]])
    assert decoder.inputs[0]["point_labels"].dtype == np.float32
    assert max(point[0] for point in result["floor_boundary"]) == 59
    assert max(point[1] for point in result["floor_boundary"]) == 79


def test_cache_reuses_only_the_current_photo_embedding(selector):
    module, encoder, decoder = selector
    image = np.zeros((80, 60, 3), np.uint8)
    module.select_surface(image, [[30, 70]], [1])
    module.select_surface(image.copy(), [[35, 65]], [1])
    assert len(encoder.inputs) == 1
    image[0, 0, 0] = 1
    module.select_surface(image, [[30, 70]], [1])
    module.select_surface(np.zeros_like(image), [[30, 70]], [1])
    assert len(encoder.inputs) == 3
    assert [int(item["image_embeddings"][0]) for item in decoder.inputs] == [1, 1, 2, 3]


def test_concurrent_requests_keep_each_photo_with_its_embedding(selector):
    module, encoder, decoder = selector
    images = [np.full((80, 60, 3), value, np.uint8) for value in (0, 20)]
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(
            lambda image: module.select_surface(image, [[30, 70]], [1]), images))
    assert all(result["floor_boundary"] for result in results)
    assert len(encoder.inputs) == 2
    assert sorted(int(item["image_embeddings"][0]) for item in decoder.inputs) == [1, 2]


def test_prompt_consistent_candidate_wins_over_a_higher_score(selector):
    module, _, decoder = selector
    decoder.scores = np.float32([[0.99, 0.8, 0.7]])
    decoder.masks = np.full((1, 3, 256, 256), -10, np.float32)
    decoder.masks[:, :, 128:, :] = 10
    decoder.masks[0, 1, 180:220, 80:120] = -10
    decoder.masks[0, 2, 180:220, 140:180] = -10

    result = module.select_surface(np.zeros((256, 256, 3), np.uint8),
                                   [[180, 180], [100, 200]], [1, 0])
    restored = build_mask(np.zeros((256, 256, 3), np.uint8),
                          [(0, 0), (255, 0), (255, 255), (0, 255)],
                          floor_boundary=result["floor_boundary"],
                          exclude=result["exclude"])
    assert restored[180, 180] == 255
    assert restored[200, 100] == 0
    assert restored[200, 160] == 255
    assert len(result["exclude"]) == 1


def test_outline_keeps_notches_and_holes_at_original_photo_size(selector):
    module, _, decoder = selector
    decoder.masks[0, 0, 160:200, 80:120] = -10
    decoder.masks[0, 0, 128:160, 140:] = -10
    result = module.select_surface(np.zeros((512, 512, 3), np.uint8),
                                   [[80, 400]], [1])
    restored = build_mask(np.zeros((512, 512, 3), np.uint8),
                          [(0, 0), (511, 0), (511, 511), (0, 511)],
                          floor_boundary=result["floor_boundary"],
                          exclude=result["exclude"])
    assert restored[360, 200] == 0
    assert restored[280, 400] == 0
    assert restored[450, 400] == 255


def test_clicked_component_wins_over_a_larger_disconnected_region(selector):
    module, _, decoder = selector
    decoder.masks[:] = -10
    decoder.masks[0, 0, 10:110, :] = 10
    decoder.masks[0, 0, 180:240, 80:140] = 10

    result = module.select_surface(np.zeros((256, 256, 3), np.uint8),
                                   [[100, 200]], [1])

    assert min(point[1] for point in result["floor_boundary"]) >= 179
    assert max(point[1] for point in result["floor_boundary"]) >= 238


def test_candidate_without_an_include_point_cannot_win_on_negative_points(selector):
    module, _, decoder = selector
    decoder.scores = np.float32([[0.99, 0.8]])
    decoder.masks = np.full((1, 2, 256, 256), -10, np.float32)
    decoder.masks[0, 1, 128:] = 10

    result = module.select_surface(np.zeros((256, 256, 3), np.uint8),
                                   [[100, 200], [150, 200], [180, 200]], [1, 0, 0])

    assert max(point[1] for point in result["floor_boundary"]) == 255


def test_no_region_error_is_usable_for_both_floor_and_wall(selector):
    module, _, decoder = selector
    decoder.masks[:] = -10
    with pytest.raises(ValueError, match="No surface") as error:
        module.select_surface(np.zeros((80, 60, 3), np.uint8), [[30, 70]], [1])
    assert "floor" not in str(error.value).lower()


@pytest.mark.parametrize("points,labels", [
    ([], []), ([[30, 70]], []), ([[30, 70]], [0]),
    ([[30, 70]], [2]), ([[60, 70]], [1]), ([[-1, 70]], [1]),
    ([[30, 80]], [1]), ([[float("nan"), 70]], [1]),
    ([[30, 70, 1]], [1]), ([[30, 70]], [[1]]),
])
def test_invalid_prompts_are_rejected_before_inference(selector, points, labels):
    module, encoder, _ = selector
    with pytest.raises(ValueError):
        module.select_surface(np.zeros((80, 60, 3), np.uint8), points, labels)
    assert not encoder.inputs


@pytest.mark.parametrize("image", [
    np.zeros((80, 60), np.uint8), np.zeros((80, 60, 4), np.uint8),
    np.zeros((80, 60, 3), np.float32), np.zeros((0, 60, 3), np.uint8),
])
def test_invalid_images_are_rejected_before_inference(selector, image):
    module, encoder, _ = selector
    with pytest.raises(ValueError):
        module.select_surface(image, [[30, 70]], [1])
    assert not encoder.inputs
