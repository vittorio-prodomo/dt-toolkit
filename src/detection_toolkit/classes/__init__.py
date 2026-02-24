"""Class filtering and merging utilities.

Public API::

    from detection_toolkit.classes import (
        ClassFilter,
        ClassFilterConfig,
        ClassMerger,
        normalize_class_name,
    )
"""

from .filter import ClassFilter, ClassFilterConfig
from .merger import ClassMerger
from .utils import normalize_class_name

__all__ = [
    "ClassFilter",
    "ClassFilterConfig",
    "ClassMerger",
    "normalize_class_name",
]
