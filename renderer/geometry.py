"""Perspective mapping between the top-down tile sheet and the photo.

Quad order is FAR-LEFT, FAR-RIGHT, NEAR-RIGHT, NEAR-LEFT everywhere.
"""
import cv2
import numpy as np


def quad_mask_of(shape, quad):
    mask = np.zeros(shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, np.int32(quad), 255)
    return mask


def sheet_to_photo_homography(sheet_shape, quad):
    height, width = sheet_shape[:2]
    src = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    return cv2.getPerspectiveTransform(src, np.float32(quad))


def lerp(a, b, t):
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def near_edge_band(quad, quad_mask, depth=0.14):
    """Strip along the NEAR edge. Assumption: this is certainly bare floor.

    Used as the seed for the colour model in renderer.masking.
    """
    band = np.int32([quad[3], quad[2],
                     lerp(quad[2], quad[1], depth), lerp(quad[3], quad[0], depth)])
    strip = np.zeros(quad_mask.shape[:2], np.uint8)
    cv2.fillPoly(strip, [band], 255)
    return cv2.bitwise_and(strip, quad_mask)
