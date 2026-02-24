"""Adapter functions mapping cocoyolo data models to detection_toolkit models."""

from cocoyolo.dataset import COCODatasetInfo
from cocoyolo.dataset_yolo import YOLODatasetInfo

from ..core.models import DatasetFormat, DatasetInfo, DatasetSplit


def coco_info_to_dataset_info(info: COCODatasetInfo) -> DatasetInfo:
    """Convert a cocoyolo ``COCODatasetInfo`` to a ``DatasetInfo``."""
    splits = {}
    for name, coco_split in info.splits.items():
        splits[name] = DatasetSplit(
            name=coco_split.name,
            image_paths=list(coco_split.image_paths),
            coco_ann_path=coco_split.ann_path,
            coco_images_root=coco_split.images_root,
        )
    return DatasetInfo(
        format=DatasetFormat.COCO,
        root_dir=info.root_dir,
        splits=splits,
        class_names=dict(info.class_names),
        total_images=info.total_images,
    )


def yolo_info_to_dataset_info(info: YOLODatasetInfo) -> DatasetInfo:
    """Convert a cocoyolo ``YOLODatasetInfo`` to a ``DatasetInfo``."""
    splits = {}
    for name, yolo_split in info.splits.items():
        splits[name] = DatasetSplit(
            name=yolo_split.name,
            image_paths=list(yolo_split.image_paths),
            label_paths=list(yolo_split.label_paths),
        )
    return DatasetInfo(
        format=DatasetFormat.YOLO,
        root_dir=info.root_dir,
        splits=splits,
        class_names=dict(info.class_names),
        total_images=info.total_images,
    )
