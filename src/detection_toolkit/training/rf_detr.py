"""RF-DETR training backend.

RF-DETR (Roboflow DETR) is a real-time object detection transformer.
Uses the ``rfdetr`` Python package from Roboflow.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from .base import TrainerBase
from .config import TrainingConfig

logger = logging.getLogger(__name__)


class RFDETRTrainer(TrainerBase):
    """Training backend for RF-DETR models.

    RF-DETR supports both YOLO and COCO dataset formats natively,
    so no automatic format conversion is performed.

    Supported model patterns::

        rf-detr-base, rf-detr-large, rfdetr-base, rfdetr-large

    Requires the ``rfdetr`` package::

        pip install rfdetr
    """

    def setup(self) -> None:
        """Prepare the RF-DETR model and dataset."""
        from rfdetr import RFDETRBase, RFDETRLarge

        model_lower = self.config.model_name.lower().replace("-", "").replace("_", "")
        if "large" in model_lower:
            self.model = RFDETRLarge()
            logger.info("RF-DETR setup: RFDETRLarge")
        else:
            self.model = RFDETRBase()
            logger.info("RF-DETR setup: RFDETRBase")

    def train(self) -> Dict[str, Any]:
        """Run the RF-DETR training loop."""
        if self.model is None:
            self.setup()

        cfg = self.config
        output_dir = str(self._resolve_project_dir())

        kwargs: Dict[str, Any] = {
            "dataset_dir": str(cfg.dataset_path),
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "imgsz": cfg.imgsz,
            "lr": cfg.lr0,
            "output_dir": output_dir,
            "device": cfg.device,
            "seed": cfg.seed,
        }

        if cfg.fraction < 1.0:
            kwargs["data_fraction"] = cfg.fraction

        # Pass through any extra kwargs
        kwargs.update(cfg.extra)

        logger.info(
            "Starting RF-DETR training: %d epochs, batch=%d, imgsz=%d",
            cfg.epochs, cfg.batch_size, cfg.imgsz,
        )
        self.model.train(**kwargs)
        self.results = {"output_dir": output_dir}
        return self.results

    def resume(self, checkpoint: Optional[Path] = None) -> Dict[str, Any]:
        """Resume RF-DETR training from a checkpoint."""
        if self.model is None:
            self.setup()

        cfg = self.config
        output_dir = str(self._resolve_project_dir())

        kwargs: Dict[str, Any] = {
            "dataset_dir": str(cfg.dataset_path),
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "imgsz": cfg.imgsz,
            "lr": cfg.lr0,
            "output_dir": output_dir,
            "device": cfg.device,
            "seed": cfg.seed,
        }

        if checkpoint is not None:
            kwargs["resume"] = str(checkpoint)
        else:
            kwargs["resume"] = True

        kwargs.update(cfg.extra)

        self.model.train(**kwargs)
        self.results = {"output_dir": output_dir}
        return self.results

