"""Ultralytics YOLO training backend.

Supports YOLOv8, YOLOv11, YOLO26 and all Ultralytics model variants
(detect, segment, classify, pose, obb).
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from .base import TrainerBase
from .config import TrainingConfig

logger = logging.getLogger(__name__)

# Suffix → Ultralytics task mapping
_TASK_MAP = {
    "-seg": "segment",
    "-cls": "classify",
    "-pose": "pose",
    "-obb": "obb",
}


def _detect_task(model_name: str) -> str:
    """Infer the Ultralytics task from the model name."""
    lower = model_name.lower()
    for suffix, task in _TASK_MAP.items():
        if suffix in lower:
            return task
    return "detect"


class UltralyticsTrainer(TrainerBase):
    """Training backend for the Ultralytics YOLO family.

    Wraps :class:`ultralytics.YOLO` and delegates all training to the
    native Ultralytics engine.

    Supported model patterns::

        yolo11n.pt, yolo11m-seg.pt, yolov8x-seg.pt, yolo26n-seg.pt, ...

    The dataset must be in YOLO format with a ``data.yaml`` file.
    If the dataset is in COCO format, it will be auto-converted first.
    """

    def setup(self) -> None:
        """Load the Ultralytics model and resolve the dataset path."""
        from ultralytics import YOLO

        model_path = self._resolve_model_path()
        self.model = YOLO(str(model_path))
        self.task = _detect_task(self.config.model_name)
        self.data_yaml = self._resolve_data_yaml()

        logger.info(
            "Ultralytics setup: model=%s, task=%s, data=%s",
            model_path.name, self.task, self.data_yaml,
        )

    def train(self) -> Dict[str, Any]:
        """Run the Ultralytics training loop."""
        if self.model is None:
            self.setup()

        kwargs = self._build_train_kwargs()
        logger.info(
            "Starting Ultralytics training: %d epochs, batch=%d, imgsz=%d",
            self.config.epochs, self.config.batch_size, self.config.imgsz,
        )
        results = self.model.train(**kwargs)
        self.results = {"ultralytics_results": results}
        return self.results

    def resume(self, checkpoint: Optional[Path] = None) -> Dict[str, Any]:
        """Resume Ultralytics training from a checkpoint."""
        from ultralytics import YOLO

        if checkpoint is not None:
            self.model = YOLO(str(checkpoint))
        elif self.model is None:
            self.setup()

        kwargs = self._build_train_kwargs()
        kwargs["resume"] = True
        results = self.model.train(**kwargs)
        self.results = {"ultralytics_results": results}
        return self.results

    def validate(self) -> Dict[str, Any]:
        """Run Ultralytics validation."""
        if self.model is None:
            self.setup()

        results = self.model.val(data=str(self.data_yaml))
        return {"ultralytics_val_results": results}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_model_path(self) -> Path:
        """Find the model weights file."""
        name = self.config.model_name
        # Add .pt extension if missing
        if not name.endswith(".pt"):
            name += ".pt"

        # Check common locations
        candidates = [
            Path(name),
            self.config.dataset_path.parent / "models" / "yolo" / name,
            Path("models") / "yolo" / name,
        ]
        for c in candidates:
            if c.exists():
                return c

        # Ultralytics will download if not found locally
        return Path(name)

    def _resolve_data_yaml(self) -> Path:
        """Find or create the data.yaml for training."""
        ds = self.config.dataset_path

        if self._needs_format_conversion("yolo"):
            logger.info("Dataset is COCO format — converting to YOLO for Ultralytics")
            from ..formats.coco_to_yolo import COCOToYOLOConverter
            import tempfile

            tmp = Path(tempfile.mkdtemp(prefix="ultralytics_yolo_"))
            converter = COCOToYOLOConverter(verbose=False)
            converter.convert(ds, tmp)
            ds = tmp

        # Look for data.yaml
        yaml_path = ds / "data.yaml"
        if yaml_path.exists():
            return yaml_path

        # Try .yml
        yml_path = ds / "data.yml"
        if yml_path.exists():
            return yml_path

        raise FileNotFoundError(
            f"No data.yaml found in {ds}. "
            "Ultralytics requires a YOLO-format dataset."
        )

    def _build_train_kwargs(self) -> Dict[str, Any]:
        """Build the kwargs dict for ``model.train()``."""
        cfg = self.config
        project_dir = self._resolve_project_dir()

        kwargs: Dict[str, Any] = {
            "data": str(self.data_yaml),
            "epochs": cfg.epochs,
            "batch": cfg.batch_size,
            "imgsz": cfg.imgsz,
            "patience": cfg.patience,
            "optimizer": cfg.optimizer,
            "lr0": cfg.lr0,
            "momentum": cfg.momentum,
            "weight_decay": cfg.weight_decay,
            "warmup_epochs": cfg.warmup_epochs,
            "device": cfg.device,
            "workers": cfg.workers,
            "project": str(project_dir.parent),
            "name": project_dir.name,
            "seed": cfg.seed,
            "amp": cfg.amp,
            "fraction": cfg.fraction,
            "save_period": cfg.save_period,
            "exist_ok": True,
            "verbose": True,
        }

        # Pretrained handling
        if cfg.pretrained.lower() in ("true", "yes", "1"):
            kwargs["pretrained"] = True
        elif cfg.pretrained.lower() in ("false", "no", "0"):
            kwargs["pretrained"] = False
        else:
            # Treat as a path to custom weights
            kwargs["pretrained"] = cfg.pretrained

        # Pass through any extra kwargs
        kwargs.update(cfg.extra)

        return kwargs
