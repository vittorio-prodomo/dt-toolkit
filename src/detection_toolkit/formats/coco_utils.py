"""COCO format utilities: RLE decoding, task detection, image lookup, mask topology."""

import json
import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

from cocoyolo.io_utils import decode_rle  # noqa: F401 — re-export
from cocoyolo.io_utils import find_coco_image as find_coco_image_file  # noqa: F401

from ..core.models import TaskType

logger = logging.getLogger(__name__)


def detect_task_type(input_path: Union[str, Path]) -> TaskType:
    """Detect detection vs segmentation from a COCO JSON or dataset directory.

    *input_path* may be a ``.json`` file or a directory containing one.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if input_path.is_file():
        if input_path.suffix.lower() != ".json":
            raise ValueError(f"Unsupported file type: {input_path.suffix}")
        json_file = input_path
    else:
        json_file = _find_json_in_dir(input_path)
        if json_file is None:
            raise RuntimeError(
                f"No JSON annotation files found under {input_path}"
            )

    return _examine_json(json_file)


def analyze_mask_topology(binary_mask: np.ndarray) -> Tuple[bool, bool]:
    """Detect multiple disjoint shapes and holes in a binary mask.

    Args:
        binary_mask: ``uint8`` array of 0s and 1s (or 0 / 255).

    Returns:
        ``(has_multiple_disjoint_shapes, has_holes)``
    """
    import cv2

    mask = binary_mask.astype(np.uint8) if binary_mask.dtype != np.uint8 else binary_mask
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return False, False

    top_level = 0
    has_holes = False
    for h in hierarchy[0]:
        if h[3] == -1:
            top_level += 1
            if h[2] != -1:
                has_holes = True

    return top_level > 1, has_holes


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------

def _find_json_in_dir(folder: Path) -> Optional[Path]:
    """Find a COCO JSON file in *folder*."""
    ann_dir = folder / "annotations"
    if ann_dir.is_dir():
        jsons = sorted(ann_dir.glob("*.json"))
        if jsons:
            return jsons[0]
    # Fallback: rglob for _annotations*.json or any .json
    for p in folder.rglob("*.json"):
        if "_annotations" in p.name or p.parent.name == "annotations":
            return p
    jsons = sorted(folder.rglob("*.json"))
    return jsons[0] if jsons else None


def _examine_json(json_path: Path) -> TaskType:
    with open(json_path) as fh:
        data = json.load(fh)

    for ann in data.get("annotations", []):
        if ann.get("segmentation"):
            return TaskType.SEGMENTATION
        if ann.get("bbox"):
            return TaskType.DETECTION

    raise ValueError(f"No valid annotations found in {json_path}")
