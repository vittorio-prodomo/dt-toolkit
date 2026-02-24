"""Format conversion and format-specific utilities.

Public API::

    from detection_toolkit.formats import (
        COCOToYOLOConverter,
        YOLOToCOCOConverter,
    )

    from detection_toolkit.formats.yolo_utils import (
        create_data_yaml,
        create_data_yaml_from_dataset_info,
        remap_yolo_annotations,
        detect_task_type,  # YOLO-specific
    )

    from detection_toolkit.formats.coco_utils import (
        find_coco_image_file,
        decode_rle,
        detect_task_type,   # COCO-specific
        analyze_mask_topology,
    )
"""

from .coco_to_yolo import COCOToYOLOConverter
from .yolo_to_coco import YOLOToCOCOConverter

__all__ = [
    "COCOToYOLOConverter",
    "YOLOToCOCOConverter",
]
