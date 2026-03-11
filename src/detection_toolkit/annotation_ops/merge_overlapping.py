"""Merge overlapping same-class annotations within a FiftyOne dataset."""

import logging
from typing import List, Set

logger = logging.getLogger(__name__)


def _find_connected_components(geometries) -> List[Set[int]]:
    """Find connected components among geometries that overlap in area.

    Uses Union-Find for O(n² α(n)) performance.

    Args:
        geometries: List of Shapely geometries.

    Returns:
        List of sets, each set containing indices of a connected component.
    """
    n = len(geometries)
    if n == 0:
        return []

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if geometries[i].intersection(geometries[j]).area > 0:
                union(i, j)

    groups = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, set()).add(i)

    return list(groups.values())


def merge_overlapping(dataset, label_field=None):
    """Merge overlapping same-class annotations in-place.

    Args:
        dataset: FiftyOne Dataset or DatasetView.
        label_field: Label field name. Auto-resolved if None.

    Returns:
        The dataset (mutated in-place).
    """
    raise NotImplementedError


from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union


def _polyline_to_shapely(polyline):
    """Convert fo.Polyline exterior ring to a Shapely Polygon."""
    ring = polyline.points[0]
    coords = [(x, y) for x, y in ring]
    poly = Polygon(coords)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly


def _shapely_to_rings(poly):
    """Convert a Shapely Polygon to FiftyOne points (list of rings)."""
    rings = []
    exterior = list(poly.exterior.coords)
    rings.append([(x, y) for x, y in exterior])
    for interior in poly.interiors:
        rings.append([(x, y) for x, y in interior.coords])
    return rings


def _merge_polygons(polylines):
    """Merge a list of same-class fo.Polylines via Shapely unary_union.

    Args:
        polylines: List of fo.Polyline instances (same class).

    Returns:
        List of fo.Polyline instances (one per resulting polygon).
    """
    if len(polylines) == 1:
        return polylines

    shapely_polys = [_polyline_to_shapely(p) for p in polylines]
    merged = unary_union(shapely_polys)

    first = polylines[0]
    label = first.label
    # Collect custom attributes from first polyline
    extra_attrs = {}
    for attr in first.field_names:
        if attr not in ("id", "attributes", "tags", "label", "points", "closed", "filled", "index"):
            extra_attrs[attr] = first[attr]

    if isinstance(merged, Polygon):
        geoms = [merged]
    elif isinstance(merged, MultiPolygon):
        geoms = list(merged.geoms)
    else:
        logger.warning("Unexpected geometry type from unary_union: %s", type(merged))
        return polylines

    import fiftyone as fo

    result = []
    for geom in geoms:
        if geom.is_empty:
            continue
        pl = fo.Polyline(
            label=label,
            points=_shapely_to_rings(geom),
            closed=True,
            filled=True,
            **extra_attrs,
        )
        result.append(pl)

    return result


def _merge_bboxes(detections):
    """Merge bbox-only Detections into one by taking the union rectangle.

    Args:
        detections: List of fo.Detection instances (same class, no masks).

    Returns:
        A single fo.Detection with the union bounding box.
    """
    if len(detections) == 1:
        return detections[0]

    x_min = min(d.bounding_box[0] for d in detections)
    y_min = min(d.bounding_box[1] for d in detections)
    x_max = max(d.bounding_box[0] + d.bounding_box[2] for d in detections)
    y_max = max(d.bounding_box[1] + d.bounding_box[3] for d in detections)

    first = detections[0]
    extra_attrs = {}
    for attr in first.field_names:
        if attr not in ("id", "attributes", "tags", "label", "bounding_box", "mask", "mask_path", "index"):
            extra_attrs[attr] = first[attr]

    import fiftyone as fo

    return fo.Detection(
        label=first.label,
        bounding_box=[x_min, y_min, x_max - x_min, y_max - y_min],
        **extra_attrs,
    )


import numpy as np


def _merge_masks(detections, width, height):
    """Merge masked Detections by rasterizing to image-space and bitwise-OR.

    Args:
        detections: List of fo.Detection instances with masks (same class).
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        A single fo.Detection with merged mask and tight bounding box.
    """
    if len(detections) == 1:
        return detections[0]

    # Rasterize all masks into a single image-space canvas
    canvas = np.zeros((height, width), dtype=bool)
    for det in detections:
        bx, by, bw, bh = det.bounding_box
        x0 = int(round(bx * width))
        y0 = int(round(by * height))
        x1 = int(round((bx + bw) * width))
        y1 = int(round((by + bh) * height))
        # Resize mask to bbox pixel dimensions if needed
        mask = det.mask
        roi_h, roi_w = y1 - y0, x1 - x0
        if mask.shape != (roi_h, roi_w):
            import cv2

            mask = cv2.resize(
                mask.astype(np.uint8), (roi_w, roi_h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        canvas[y0:y1, x0:x1] |= mask

    # Find tight bounding box of merged mask
    rows = np.any(canvas, axis=1)
    cols = np.any(canvas, axis=0)
    if not rows.any():
        return detections[0]

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    y_max += 1
    x_max += 1

    # Crop mask and compute relative bbox
    cropped = canvas[y_min:y_max, x_min:x_max]
    rel_bbox = [
        x_min / width,
        y_min / height,
        (x_max - x_min) / width,
        (y_max - y_min) / height,
    ]

    first = detections[0]
    extra_attrs = {}
    for attr in first.field_names:
        if attr not in ("id", "attributes", "tags", "label", "bounding_box", "mask", "mask_path", "index"):
            extra_attrs[attr] = first[attr]

    import fiftyone as fo

    return fo.Detection(
        label=first.label,
        bounding_box=rel_bbox,
        mask=cropped,
        **extra_attrs,
    )
