"""Dataset tiling with polygon processing and parallel execution.

Public API::

    from detection_toolkit.tiling import (
        TilingConfig,
        DatasetTiler,
    )
"""

from .config import TilingConfig
from .tiler import DatasetTiler

__all__ = [
    "TilingConfig",
    "DatasetTiler",
]
