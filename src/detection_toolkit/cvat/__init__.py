"""CVAT integration — task management via the CVAT SDK."""

from detection_toolkit.cvat.client import CvatClient
from detection_toolkit.cvat.splitter import SplitResult, TaskSplitter

__all__ = ["CvatClient", "SplitResult", "TaskSplitter"]
