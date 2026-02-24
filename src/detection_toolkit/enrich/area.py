"""Pure area computation functions — no FiftyOne dependency."""

import logging
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def bbox_area(
    bounding_box: List[float],
    img_width: int,
    img_height: int,
) -> Tuple[float, float]:
    """Compute area from a FiftyOne bounding box ``[x, y, w, h]`` (relative).

    Returns:
        ``(area_px, area_rel)``
    """
    _, _, w, h = bounding_box
    area_rel = w * h
    area_px = area_rel * img_width * img_height
    return area_px, area_rel


def mask_area(
    mask: np.ndarray,
    bounding_box: List[float],
    img_width: int,
    img_height: int,
) -> Tuple[float, float]:
    """Compute area from a detection mask + its bounding box.

    The mask is a 2-D array whose dimensions may differ from the actual
    bounding-box pixel size, so we use the ratio of non-zero pixels to
    total mask pixels, scaled by the bounding-box area.

    Returns:
        ``(area_px, area_rel)``
    """
    total_mask_pixels = mask.shape[0] * mask.shape[1]
    if total_mask_pixels == 0:
        return 0.0, 0.0
    mask_ratio = np.count_nonzero(mask) / total_mask_pixels
    bbox_px, bbox_rel = bbox_area(bounding_box, img_width, img_height)
    return bbox_px * mask_ratio, bbox_rel * mask_ratio


def polyline_area(
    points: List[List[List[float]]],
    img_width: int,
    img_height: int,
) -> Tuple[float, float]:
    """Compute area from polyline points (list of sub-polygons).

    Each sub-polygon is a list of ``[x, y]`` pairs in relative coordinates.
    Uses Shapely for robust polygon area calculation.

    Returns:
        ``(area_px, area_rel)``
    """
    from shapely.geometry import Polygon as ShapelyPolygon
    from shapely.validation import make_valid

    area_rel = 0.0
    for ring in points:
        if len(ring) < 3:
            continue
        try:
            poly = ShapelyPolygon(ring)
            if not poly.is_valid:
                poly = make_valid(poly)
            area_rel += poly.area
        except Exception as exc:
            logger.debug("Skipping degenerate polygon: %s", exc)
            continue

    area_px = area_rel * img_width * img_height
    return area_px, area_rel
