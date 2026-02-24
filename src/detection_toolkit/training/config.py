"""Universal training configuration dataclass."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from ..core.models import DatasetFormat


@dataclass
class TrainingConfig:
    """Backend-agnostic training configuration.

    Common parameters are explicit fields; backend-specific pass-through
    options go in :attr:`extra`.

    Args:
        model_name: Model identifier (e.g. ``"yolo11m-seg"``,
            ``"rf-detr-base"``, ``"mask_rcnn_R_50_FPN_3x"``).
        backend: Training backend name.  Auto-detected from *model_name*
            if ``None``.
        dataset_path: Root directory of the dataset.
        dataset_format: ``YOLO`` or ``COCO``.
        epochs: Number of training epochs.
        batch_size: Per-GPU batch size.
        imgsz: Input image resolution (square).
        patience: Early-stopping patience (epochs with no improvement).
        optimizer: Optimizer name (``"AdamW"``, ``"SGD"``, ``"Adam"``, …).
        lr0: Initial learning rate.
        momentum: SGD momentum / Adam beta1.
        weight_decay: L2 regularisation.
        warmup_epochs: Linear warmup epochs.
        device: Device specification (e.g. ``0``, ``"0,1"``, ``"cpu"``).
        workers: Data-loader workers per GPU.
        project: Output directory for this training run.
        name: Run name inside *project*.
        resume: Resume from the last checkpoint.
        amp: Automatic mixed precision.
        seed: Random seed.
        pretrained: Pretrained weights strategy.
        fraction: Dataset fraction (for quick tests).
        save_period: Checkpoint save period in epochs (``-1`` = best only).
        extra: Backend-specific options passed through verbatim.
    """

    model_name: str
    backend: Optional[str] = None
    dataset_path: Path = Path(".")
    dataset_format: DatasetFormat = DatasetFormat.YOLO
    epochs: int = 200
    batch_size: int = 32
    imgsz: int = 640
    patience: int = 35
    optimizer: str = "AdamW"
    lr0: float = 0.001
    momentum: float = 0.937
    weight_decay: float = 0.0005
    warmup_epochs: float = 3.0
    device: str = "0"
    workers: int = 16
    project: Optional[str] = None
    name: str = "run"
    resume: bool = False
    amp: bool = False
    seed: int = 0
    pretrained: str = "true"
    fraction: float = 1.0
    save_period: int = -1
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.dataset_path = Path(self.dataset_path)
        if self.epochs <= 0:
            raise ValueError(f"epochs must be > 0, got {self.epochs}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")
        if self.imgsz <= 0:
            raise ValueError(f"imgsz must be > 0, got {self.imgsz}")
        if self.lr0 <= 0:
            raise ValueError(f"lr0 must be > 0, got {self.lr0}")

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "TrainingConfig":
        """Load from a YAML file."""
        with open(path) as fh:
            data = yaml.safe_load(fh)
        if "dataset_path" in data:
            data["dataset_path"] = Path(data["dataset_path"])
        if "dataset_format" in data:
            data["dataset_format"] = DatasetFormat(data["dataset_format"])
        return cls(**data)

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Serialise to a YAML file."""
        from dataclasses import asdict

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        d = asdict(self)
        d["dataset_path"] = str(d["dataset_path"])
        d["dataset_format"] = d["dataset_format"]
        with open(path, "w") as fh:
            yaml.dump(d, fh, default_flow_style=False)
