"""Extract and convert FiftyOne annotations for tiling workers.

Converts FiftyOne bulk ``values()`` output into plain-Python dicts that
can be passed to :class:`~concurrent.futures.ProcessPoolExecutor` workers.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


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
          ``polygon`` ([(x, y), ...] in absolute pixels)
    """
    if sample_labels is None:
        return []

    items = sample_labels.get(container_type, [])
    if not items:
        return []

    result: List[Dict] = []

    for item in items:
        label = item.get("label")
        if label not in class_map:
            continue
        class_id = class_map[label]

        if container_type == "detections":
            bbox_rel = item.get("bounding_box")
            if bbox_rel is None or len(bbox_rel) != 4:
                continue
            # FiftyOne bbox: [x, y, w, h] relative to image size
            abs_bbox = [
                bbox_rel[0] * img_w,
                bbox_rel[1] * img_h,
                bbox_rel[2] * img_w,
                bbox_rel[3] * img_h,
            ]
            # TODO: mask support — when Detection has .mask, convert via
            # cocoyolo.geometry.mask_to_polygons() instead of using bbox.
            result.append({
                "class_id": class_id,
                "type": "bbox",
                "bbox": abs_bbox,
            })

        elif container_type == "polylines":
            points_groups = item.get("points")
            if not points_groups or not points_groups[0]:
                continue
            # FiftyOne Polyline.points: list of list of (x, y) relative
            # We take the first (outer) ring.
            rel_points = points_groups[0]
            abs_points = [
                (px * img_w, py * img_h) for px, py in rel_points
            ]
            result.append({
                "class_id": class_id,
                "type": "polygon",
                "polygon": abs_points,
            })

    return result
