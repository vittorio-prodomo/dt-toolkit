"""Unified multi-backend training abstraction.

Public API::

    from detection_toolkit.training import (
        TrainingConfig,
        TrainerBase,
        get_trainer,
        TRAINER_REGISTRY,
    )

The :func:`get_trainer` factory auto-detects the correct backend from
the model name or uses the explicit ``backend`` field in the config.
"""

import logging
import re
from typing import Dict, Type

from .base import TrainerBase
from .config import TrainingConfig

logger = logging.getLogger(__name__)

# Backend name → Trainer class (lazy import to avoid hard dependencies)
TRAINER_REGISTRY: Dict[str, str] = {
    "ultralytics": "detection_toolkit.training.ultralytics.UltralyticsTrainer",
    "mask-rcnn": "detection_toolkit.training.mask_rcnn.MaskRCNNTrainer",
    "rf-detr": "detection_toolkit.training.rf_detr.RFDETRTrainer",
}

# Model name patterns → backend name
_BACKEND_PATTERNS = [
    (re.compile(r"^(yolo|yolov)\d", re.IGNORECASE), "ultralytics"),
    (re.compile(r"^(rf[_-]?detr|rfdetr)", re.IGNORECASE), "rf-detr"),
    (re.compile(r"^(mask[_-]?rcnn|faster[_-]?rcnn|retinanet)", re.IGNORECASE), "mask-rcnn"),
]


def detect_backend(model_name: str) -> str:
    """Infer the training backend from a model name.

    Raises:
        ValueError: If no backend matches *model_name*.
    """
    for pattern, backend in _BACKEND_PATTERNS:
        if pattern.search(model_name):
            return backend
    raise ValueError(
        f"Cannot auto-detect backend for model {model_name!r}. "
        f"Set config.backend explicitly. Known backends: {list(TRAINER_REGISTRY)}"
    )


def _import_class(dotted_path: str) -> Type[TrainerBase]:
    """Import a trainer class from its dotted module path."""
    module_path, class_name = dotted_path.rsplit(".", 1)
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


def get_trainer(config: TrainingConfig) -> TrainerBase:
    """Create a trainer instance for the given config.

    If ``config.backend`` is ``None``, the backend is inferred from
    ``config.model_name`` using pattern matching.

    Returns:
        An initialised (but not yet ``setup()``-ed) trainer.

    Raises:
        ValueError: If the backend cannot be determined.
        KeyError: If the backend is not in :data:`TRAINER_REGISTRY`.
    """
    backend = config.backend or detect_backend(config.model_name)
    logger.info("Using training backend: %s", backend)

    if backend not in TRAINER_REGISTRY:
        raise KeyError(
            f"Unknown backend {backend!r}. "
            f"Registered: {list(TRAINER_REGISTRY)}"
        )

    cls = _import_class(TRAINER_REGISTRY[backend])
    return cls(config)


__all__ = [
    "TrainingConfig",
    "TrainerBase",
    "get_trainer",
    "detect_backend",
    "TRAINER_REGISTRY",
]
