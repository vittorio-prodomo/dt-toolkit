"""Dataset merging with class remapping and conflict resolution.

Public API::

    from detection_toolkit.merge import (
        MergeConfig,
        DatasetMerger,
    )
"""

from .config import MergeConfig
from .merger import DatasetMerger

__all__ = [
    "MergeConfig",
    "DatasetMerger",
]
