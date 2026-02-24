"""Core data models, dataset loading, and file operations.

Public API::

    from detection_toolkit.core import (
        DatasetFormat, TaskType, DatasetSplit, DatasetInfo,
        DatasetLoader,
        AnnotationHandler,
        FileOperations,
        get_image_size, find_image_file, find_images_in_dir,
    )
"""

from .annotations import AnnotationHandler
from .file_ops import FileOperations
from .image_utils import find_image_file, find_images_in_dir, get_image_size
from .loader import DatasetLoader
from .models import DatasetFormat, DatasetInfo, DatasetSplit, TaskType

__all__ = [
    "DatasetFormat",
    "TaskType",
    "DatasetSplit",
    "DatasetInfo",
    "DatasetLoader",
    "AnnotationHandler",
    "FileOperations",
    "get_image_size",
    "find_image_file",
    "find_images_in_dir",
]
