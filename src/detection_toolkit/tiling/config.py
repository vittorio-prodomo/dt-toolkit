"""Configuration dataclass for dataset tiling."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import yaml


@dataclass
class TilingConfig:
    """Configuration for :class:`~detection_toolkit.tiling.DatasetTiler`.

    Args:
        tile_size: Width and height of each square tile in pixels.
        overlap_percent: Fractional overlap between adjacent tiles (0.0–1.0).
        edge_behavior: How to handle tiles at image boundaries.

            * ``"crop"`` — edge tiles may be smaller than *tile_size*.
            * ``"padding"`` — pad edge tiles to full *tile_size*.

        padding_value: RGB fill colour when *edge_behavior* is ``"padding"``.
        min_area_ratio: Minimum annotation area (relative to tile area) to keep.
        max_area_ratio: Maximum annotation area (relative to tile area) to keep.
        negative_sample_ratio: Ratio of empty tiles to annotated tiles.
            ``-1`` keeps all empty tiles.
        densify_factor: Polygon densification factor — smaller values add
            more intermediate points (distance = perimeter × factor).
        smoothing_tolerance: Shapely ``simplify`` tolerance for polygon smoothing.
        compression: JPEG quality (0–100). Only used when writing JPEG tiles.
        max_processes: Maximum parallel workers. ``None`` = auto.
        random_seed: Seed for negative-sample sampling reproducibility.
        fraction: Process only this fraction of images (for quick tests).
    """

    tile_size: int
    overlap_percent: float = 0.0
    edge_behavior: str = "crop"
    padding_value: Tuple[int, int, int] = (153, 153, 153)
    min_area_ratio: float = 0.1
    max_area_ratio: float = 1.0
    densify_factor: float = 0.5
    smoothing_tolerance: float = 0.1
    negative_sample_ratio: float = 0.1
    compression: int = 90
    max_processes: Optional[int] = None
    random_seed: Optional[int] = None
    fraction: Optional[float] = None
    flatten: bool = False

    def __post_init__(self) -> None:
        if self.tile_size <= 0:
            raise ValueError(f"tile_size must be > 0, got {self.tile_size}")
        if not 0.0 <= self.overlap_percent <= 1.0:
            raise ValueError(
                f"overlap_percent must be in [0, 1], got {self.overlap_percent}"
            )
        if self.edge_behavior not in ("crop", "padding"):
            raise ValueError(
                f"Invalid edge_behavior: {self.edge_behavior!r}"
            )
        if not 0.0 <= self.min_area_ratio <= 1.0:
            raise ValueError(
                f"min_area_ratio must be in [0, 1], got {self.min_area_ratio}"
            )
        if not 0.0 <= self.max_area_ratio <= 1.0:
            raise ValueError(
                f"max_area_ratio must be in [0, 1], got {self.max_area_ratio}"
            )
        if self.min_area_ratio > self.max_area_ratio:
            raise ValueError(
                f"min_area_ratio ({self.min_area_ratio}) > "
                f"max_area_ratio ({self.max_area_ratio})"
            )
        if not 0 <= self.compression <= 100:
            raise ValueError(
                f"compression must be in [0, 100], got {self.compression}"
            )
        if self.fraction is not None and self.fraction <= 0:
            raise ValueError(f"fraction must be > 0, got {self.fraction}")

    def to_dict(self) -> Dict:
        """Serialise to a plain dict (pickle-safe for multiprocessing)."""
        from dataclasses import asdict
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "TilingConfig":
        """Reconstruct from a dict (e.g. in a worker process)."""
        return cls(**d)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "TilingConfig":
        """Load from a YAML file."""
        with open(path) as fh:
            data = yaml.safe_load(fh)
        if "padding_value" in data and isinstance(data["padding_value"], list):
            data["padding_value"] = tuple(data["padding_value"])
        return cls(**data)

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Serialise to a YAML file."""
        from dataclasses import asdict

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        d = asdict(self)
        d["padding_value"] = list(d["padding_value"])
        with open(path, "w") as fh:
            yaml.dump(d, fh, default_flow_style=False)
