"""Extract and convert FiftyOne annotations for tiling workers.

Converts FiftyOne bulk ``values()`` output into plain-Python dicts that
can be passed to :class:`~concurrent.futures.ProcessPoolExecutor` workers.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _get(item, key):
    """Get a field from either a FiftyOne label object or a plain dict."""
    if isinstance(item, dict):
        return item.get(key)
    return getattr(item, key, None)


def build_class_map(
    classes_dict: Optional[Dict[str, List[str]]],
) -> Dict[str, int]:
    """Build ``{class_name: class_id}`` from a FiftyOne ``dataset.classes`` dict.

    FiftyOne stores classes as ``{label_field: [name, ...]}``. We take the
    first (and typically only) entry and assign 0-based IDs.
    """
    if not classes_dict:
        return {}
    class_list = list(classes_dict.values())[0]
    return {name: idx for idx, name in enumerate(class_list)}


def extract_annotations(
    sample_labels: Optional[Dict[str, Any]],
    container_type: str,
    class_map: Dict[str, int],
    img_w: int,
    img_h: int,
) -> List[Dict]:
    """Convert one sample's FiftyOne labels to plain-Python annotation dicts.

    Args:
        sample_labels: The raw dict from ``dataset.values("<field>")[i]``.
            Contains either ``"detections"`` or ``"polylines"`` key.
        container_type: ``"detections"`` or ``"polylines"``.
        class_map: ``{class_name: class_id}`` mapping.
        img_w: Image width in pixels.
        img_h: Image height in pixels.

    Returns:
        List of annotation dicts, each with keys:
        - ``class_id`` (int)
        - ``type`` (``"bbox"`` or ``"polygon"``)
        - ``bbox`` ([x, y, w, h] in absolute pixels) or
          ``polygons`` ([[(x, y), ...], ...] in absolute pixels)
    """
    if sample_labels is None:
        return []

    items = sample_labels.get(container_type, [])
    if not items:
        return []

    result: List[Dict] = []

    for item in items:
        label = _get(item, "label")
        if label not in class_map:
            continue
        class_id = class_map[label]

        if container_type == "detections":
            bbox_rel = _get(item, "bounding_box")
            if bbox_rel is None or len(bbox_rel) != 4:
                continue
            # FiftyOne bbox: [x, y, w, h] relative to image size
            abs_x = bbox_rel[0] * img_w
            abs_y = bbox_rel[1] * img_h
            abs_w = bbox_rel[2] * img_w
            abs_h = bbox_rel[3] * img_h

            mask = _get(item, "mask")
            if mask is not None:
                # Convert instance mask to polygon(s) in absolute image coords.
                # FiftyOne mask shape is (mask_h, mask_w) in bbox-local pixels.
                from cocoyolo.geometry import mask_to_polygons
                import numpy as np
                mask_h, mask_w = mask.shape
                raw_polys = mask_to_polygons(mask.astype(np.uint8))
                if raw_polys:
                    abs_polygons = [
                        [
                            (abs_x + pt[0] * abs_w / mask_w,
                             abs_y + pt[1] * abs_h / mask_h)
                            for pt in poly
                        ]
                        for poly in raw_polys
                    ]
                    result.append({
                        "class_id": class_id,
                        "type": "polygon",
                        "polygons": abs_polygons,
                    })
                    continue
            # No mask (or empty mask) — fall back to bbox
            result.append({
                "class_id": class_id,
                "type": "bbox",
                "bbox": [abs_x, abs_y, abs_w, abs_h],
            })

        elif container_type == "polylines":
            points_groups = _get(item, "points")
            if not points_groups:
                continue
            # FiftyOne Polyline.points: list of rings, each a list of
            # (x, y) relative coords.  Extract all non-empty rings.
            abs_polygons = []
            for ring in points_groups:
                if ring:
                    abs_polygons.append([
                        (px * img_w, py * img_h) for px, py in ring
                    ])
            if not abs_polygons:
                continue
            result.append({
                "class_id": class_id,
                "type": "polygon",
                "polygons": abs_polygons,
            })

    return result
