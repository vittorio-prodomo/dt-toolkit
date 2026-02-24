"""Convert YOLO Ultralytics format datasets to COCO format.

Delegates to ``cocoyolo.yolo_to_coco.convert_yolo_to_coco`` for the actual
conversion logic, preserving the class-based API for backward compatibility
within detection_toolkit.
"""

import logging
from pathlib import Path
from typing import Union

from ..core.loader import DatasetLoader
from ..core.models import DatasetFormat, DatasetInfo

logger = logging.getLogger(__name__)


class YOLOToCOCOConverter:
    """Convert a YOLO dataset to COCO format.

    Usage::

        converter = YOLOToCOCOConverter()
        converter.convert(Path("yolo_dataset"), Path("coco_output"))
    """

    def __init__(self, verbose: bool = True) -> None:
        self.verbose = verbose

    def convert(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        keep_zero_indexing: bool = False,
        min_bbox_size: float = 1.0,
        image_mode: str = "symlink",
    ) -> DatasetInfo:
        """Run the conversion.

        Args:
            input_path: Root of the source YOLO dataset.
            output_path: Where to write the COCO dataset.
            keep_zero_indexing: If *False* (default), shift category IDs
                from YOLO 0-based to COCO 1-based.
            min_bbox_size: Drop bounding boxes whose width or height is
                smaller than this value in pixels.
            image_mode: How to transfer images (``"copy"``, ``"symlink"``,
                ``"hardlink"``).  Defaults to ``"symlink"``.

        Returns:
            A :class:`DatasetInfo` describing the newly created COCO dataset.
        """
        from cocoyolo.yolo_to_coco import convert_yolo_to_coco

        input_path = Path(input_path)
        output_path = Path(output_path)

        logger.info("YOLO → COCO: %s → %s", input_path, output_path)

        _src_info, _stats = convert_yolo_to_coco(
            input_path=input_path,
            output_path=output_path,
            keep_zero_indexing=keep_zero_indexing,
            min_bbox_size=min_bbox_size,
            image_mode=image_mode,
            verbose=self.verbose,
        )

        logger.info("YOLO → COCO conversion complete")
        out_loader = DatasetLoader(output_path, verbose=False)
        return out_loader.load_dataset(format=DatasetFormat.COCO)
