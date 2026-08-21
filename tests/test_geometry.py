import cv2
import numpy as np
from renderer.geometry import (lerp, near_edge_band, quad_mask_of,
                               sheet_to_photo_homography)

QUAD = [(335, 432), (952, 432), (1245, 828), (58, 828)]
SHAPE = (860, 1280)


def test_homography_maps_sheet_corners_onto_quad():
    sheet_shape = (400, 600, 3)
    h = sheet_to_photo_homography(sheet_shape, QUAD)
    corners = np.float32([[0, 0], [600, 0], [600, 400], [0, 400]]).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(corners, h).reshape(-1, 2)
    assert np.allclose(mapped, np.float32(QUAD), atol=1e-3)


def test_lerp_endpoints_and_midpoint():
    assert lerp((0, 0), (10, 20), 0.0) == (0, 0)
    assert lerp((0, 0), (10, 20), 1.0) == (10, 20)
    assert lerp((0, 0), (10, 20), 0.5) == (5, 10)


def test_quad_mask_is_binary_and_covers_interior():
    m = quad_mask_of(SHAPE, QUAD)
    assert set(np.unique(m)).issubset({0, 255})
    assert m[800, 650] == 255      # well inside
    assert m[100, 650] == 0        # above the far edge


def test_near_edge_band_sits_inside_quad_and_near_the_bottom():
    qm = quad_mask_of(SHAPE, QUAD)
    band = near_edge_band(QUAD, qm)
    assert band.any()
    assert not (band[qm == 0] > 0).any()          # never escapes the quad
    ys = np.where(band > 0)[0]
    assert ys.min() > 750                          # only near the NEAR edge


def test_deeper_band_covers_more_area():
    qm = quad_mask_of(SHAPE, QUAD)
    shallow = (near_edge_band(QUAD, qm, depth=0.10) > 0).sum()
    deep = (near_edge_band(QUAD, qm, depth=0.30) > 0).sum()
    assert deep > shallow
