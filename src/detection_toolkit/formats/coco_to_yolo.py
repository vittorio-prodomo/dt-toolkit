"""Convert COCO format datasets to YOLO Ultralytics format.

Delegates to ``cocoyolo.coco_to_yolo.convert`` for the actual conversion
logic, preserving the class-based API for backward compatibility within
detection_toolkit.
"""

import logging
import shutil
from pathlib import Path
from typing import Union

from ..core.loader import DatasetLoader
from ..core.models import DatasetFormat, DatasetInfo

logger = logging.getLogger(__name__)


class COCOToYOLOConverter:
    """Convert a COCO dataset to YOLO Ultralytics format.

    Usage::

        converter = COCOToYOLOConverter()
        converter.convert(Path("coco_dataset"), Path("yolo_output"))
    """

    VALID_HOLE_STRATEGIES = ("fill", "bridge")
    VALID_DISJOINT_STRATEGIES = ("split", "bridge")

    def __init__(self, verbose: bool = True) -> None:
        self.verbose = verbose

    def convert(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        contour_approx_factor: float = 0.001,
        hole_strategy: str = "bridge",
        disjoint_strategy: str = "bridge",
        image_mode: str = "symlink",
        task: str = "auto",
        workers: int = None,
    ) -> DatasetInfo:
        """Run the conversion.

        Args:
            input_path: Root of the source COCO dataset.
            output_path: Where to write the YOLO dataset.
            contour_approx_factor: ``cv2.approxPolyDP`` epsilon as a fraction
                of the contour arc length.
            hole_strategy: How to handle holes in RLE masks.
                ``"fill"`` ignores holes; ``"bridge"`` preserves them via
                inverse bridges.
            disjoint_strategy: How to handle disjoint regions.
                ``"split"`` writes each region as a separate annotation;
                ``"bridge"`` connects them via nearest-neighbour chain
                with zero-width bridges.
            image_mode: How to transfer images (``"copy"``, ``"symlink"``,
                ``"hardlink"``).  Defaults to ``"symlink"``.
            task: YOLO output task type (``"auto"``, ``"detect"``,
                ``"segment"``).
            workers: Number of parallel workers (``None`` = all CPUs).

        Returns:
            A :class:`DatasetInfo` describing the newly created YOLO dataset.
        """
        if hole_strategy not in self.VALID_HOLE_STRATEGIES:
            raise ValueError(
                f"hole_strategy must be one of {self.VALID_HOLE_STRATEGIES}, "
                f"got {hole_strategy!r}"
            )
        if disjoint_strategy not in self.VALID_DISJOINT_STRATEGIES:
            raise ValueError(
                f"disjoint_strategy must be one of {self.VALID_DISJOINT_STRATEGIES}, "
                f"got {disjoint_strategy!r}"
            )

        from cocoyolo.coco_to_yolo import convert as _cocoyolo_convert

        input_path = Path(input_path)
        output_path = Path(output_path)

        logger.info("COCO → YOLO: %s → %s", input_path, output_path)

        try:
            _src_info, _stats = _cocoyolo_convert(
                input_path=input_path,
                output_path=output_path,
                contour_approx_factor=contour_approx_factor,
                hole_strategy=hole_strategy,
                disjoint_strategy=disjoint_strategy,
                task=task,
                image_mode=image_mode,
                workers=workers,
                verbose=self.verbose,
            )
        except Exception:
            if output_path.exists():
                shutil.rmtree(output_path, ignore_errors=True)
            raise

        logger.info("COCO → YOLO conversion complete")
        out_loader = DatasetLoader(output_path, verbose=False)
        return out_loader.load_dataset(format=DatasetFormat.YOLO)
