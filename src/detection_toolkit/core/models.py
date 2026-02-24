"""Core data models for dataset representation."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional


class DatasetFormat(str, Enum):
    """Supported dataset annotation formats."""

    YOLO = "yolo"
    COCO = "coco"


class TaskType(str, Enum):
    """Supported computer vision task types."""

    DETECTION = "detection"
    SEGMENTATION = "segmentation"


@dataclass
class DatasetSplit:
    """One split (train/val/test) of a dataset.

    Carries both YOLO paths (label_paths) and COCO paths (coco_ann_path),
    with unused fields being empty/None depending on the format.
    """

    name: str
    image_paths: List[Path]
    label_paths: List[Path] = field(default_factory=list)
    coco_ann_path: Optional[Path] = None
    coco_images_root: Optional[Path] = None

    @property
    def num_images(self) -> int:
        return len(self.image_paths)

    @property
    def num_labels(self) -> int:
        return len(self.label_paths)


@dataclass
class DatasetInfo:
    """Complete dataset metadata — format-agnostic.

    All downstream code works exclusively with this structure.
    The on-disk directory variant is never leaked beyond the loader.
    """

    format: DatasetFormat
    root_dir: Path
    splits: Dict[str, DatasetSplit]
    class_names: Dict[int, str]
    total_images: int
    total_instances: int = 0
    task_type: TaskType = TaskType.DETECTION

    @property
    def class_list(self) -> List[str]:
        """Ordered class names (sorted by ID)."""
        return [self.class_names[k] for k in sorted(self.class_names)]

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def split_names(self) -> List[str]:
        return list(self.splits.keys())
