"""Unified annotation handling for YOLO and COCO formats."""

from typing import Dict, List, Optional, Tuple

from cocoyolo.annotations import parse_yolo_line as _parse_yolo_line
from cocoyolo.annotations import polygon_area as _polygon_area
from cocoyolo.annotations import recompute_bbox_from_polygon as _recompute_bbox


class AnnotationHandler:
    """Parse and normalise YOLO / COCO annotations to a common dict format.

    Every ``parse_*`` method returns a dict with at least:

    - ``class_id`` (int)
    - ``type`` — one of ``"bbox"``, ``"polygon"``, ``"rle"``
    - ``bbox`` — ``[x, y, w, h]`` in absolute pixels (COCO convention)
    - ``area`` — annotation area in pixels²

    For segmentation annotations the dict also contains ``polygon`` (list of
    ``(x, y)`` tuples in absolute pixels) or ``rle`` (raw COCO RLE dict).
    """

    @staticmethod
    def parse_yolo_line(
        line: str, img_width: int, img_height: int
    ) -> Optional[Dict]:
        """Parse one line from a YOLO ``.txt`` label file.

        Delegates to ``cocoyolo.annotations.parse_yolo_line``.
        """
        return _parse_yolo_line(line, img_width, img_height)

    @staticmethod
    def parse_coco_annotation(
        ann: Dict, img_width: int, img_height: int
    ) -> Dict:
        """Normalise a single COCO annotation dict to the common format.

        Handles bbox-only, polygon segmentation, and RLE segmentation.
        """
        result: Dict = {
            "class_id": ann["category_id"],
            "area": float(ann.get("area", 0)),
        }

        seg = ann.get("segmentation")
        if seg:
            if isinstance(seg, dict):
                result["type"] = "rle"
                result["rle"] = seg
            elif isinstance(seg, list) and seg:
                result["type"] = "polygon"
                flat = seg[0]
                result["polygon"] = [
                    (flat[i], flat[i + 1]) for i in range(0, len(flat), 2)
                ]
                if len(seg) > 1:
                    result["all_polygons"] = [
                        [(s[i], s[i + 1]) for i in range(0, len(s), 2)]
                        for s in seg
                    ]
            else:
                result["type"] = "bbox"
        else:
            result["type"] = "bbox"

        if "bbox" in ann:
            x, y, w, h = ann["bbox"]
            result["bbox"] = [float(x), float(y), float(w), float(h)]
            if result["area"] == 0:
                result["area"] = float(w * h)

        return result

    @staticmethod
    def recompute_bbox_from_polygon(
        points: List[Tuple[float, float]],
    ) -> List[float]:
        """Compute ``[x, y, w, h]`` bounding box from polygon points."""
        return _recompute_bbox(points)

    @staticmethod
    def polygon_area(points: List[Tuple[float, float]]) -> float:
        """Polygon area via the shoelace formula."""
        return _polygon_area(points)
