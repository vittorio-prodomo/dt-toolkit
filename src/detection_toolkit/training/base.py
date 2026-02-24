"""Abstract base class for all training backends."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

from .config import TrainingConfig

logger = logging.getLogger(__name__)


class TrainerBase(ABC):
    """Strategy interface for model training.

    Each backend (Ultralytics, Detectron2, RF-DETR, …) implements this
    interface so that callers can switch backends without changing code.

    Usage::

        from detection_toolkit.training import get_trainer, TrainingConfig

        config = TrainingConfig(model_name="yolo11m-seg", ...)
        trainer = get_trainer(config)
        trainer.setup()
        results = trainer.train()
    """

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.model = None
        self.results: Optional[Dict[str, Any]] = None

    @abstractmethod
    def setup(self) -> None:
        """Prepare the model, dataset, and output directories.

        Called once before :meth:`train` or :meth:`resume`.
        """

    @abstractmethod
    def train(self) -> Dict[str, Any]:
        """Run the full training loop.

        Returns:
            A dict of final metrics (backend-specific keys).
        """

    @abstractmethod
    def resume(self, checkpoint: Optional[Path] = None) -> Dict[str, Any]:
        """Resume training from a checkpoint.

        Args:
            checkpoint: Explicit checkpoint path.  If ``None``, the
                backend should locate the latest checkpoint automatically.

        Returns:
            A dict of final metrics.
        """

    def validate(self) -> Dict[str, Any]:
        """Run validation only (optional override).

        Returns:
            A dict of validation metrics.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support standalone validation"
        )

    # ------------------------------------------------------------------
    # Helpers available to all backends
    # ------------------------------------------------------------------

    def _resolve_project_dir(self) -> Path:
        """Return the output directory for this run, creating it if needed."""
        if self.config.project:
            project = Path(self.config.project)
        else:
            project = Path("runs") / self.config.model_name

        run_dir = project / self.config.name
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _needs_format_conversion(self, required: str) -> bool:
        """Check if the dataset format needs conversion for this backend."""
        return self.config.dataset_format.value != required
