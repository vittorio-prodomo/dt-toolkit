"""Configuration dataclass for dataset merging."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


@dataclass
class MergeConfig:
    """Configuration for :class:`~detection_toolkit.merge.DatasetMerger`.

    Args:
        label_mapping: ``{canonical_name: [variant_names]}``.
            Class names found across datasets are matched against variants
            and remapped to the canonical name.
        missing_splits: How to handle datasets that lack some splits.

            * ``"include_all"`` — include all splits found in *any* dataset.
            * ``"common_only"`` — only include splits present in *all* datasets.
            * ``"error"`` — raise if datasets have different split sets.

        image_conflicts: How to handle duplicate image filenames.

            * ``"prefix_dataset"`` — prepend the dataset directory name.
            * ``"error"`` — raise on the first conflict.

        unmapped_classes: What to do with classes not in *label_mapping*.

            * ``"include"`` — keep as-is (the class name becomes its own canonical).
            * ``"skip"`` — silently drop annotations for unmapped classes.
            * ``"error"`` — raise.

        filter_classes: If set, only these canonical class names survive.
        output_splits: Final split layout.

            * ``"preserve_splits"`` — keep original split structure.
            * ``"merge_to_train"`` — collapse everything into ``train``.
    """

    label_mapping: Dict[str, List[str]]
    missing_splits: str = "include_all"
    image_conflicts: str = "prefix_dataset"
    unmapped_classes: str = "include"
    filter_classes: Optional[List[str]] = None
    output_splits: str = "preserve_splits"

    def __post_init__(self) -> None:
        if self.missing_splits not in ("include_all", "common_only", "error"):
            raise ValueError(f"Invalid missing_splits: {self.missing_splits}")
        if self.image_conflicts not in ("prefix_dataset", "error"):
            raise ValueError(
                f"Invalid image_conflicts: {self.image_conflicts}"
            )
        if self.unmapped_classes not in ("include", "skip", "error"):
            raise ValueError(
                f"Invalid unmapped_classes: {self.unmapped_classes}"
            )
        if self.output_splits not in ("preserve_splits", "merge_to_train"):
            raise ValueError(
                f"Invalid output_splits: {self.output_splits}"
            )

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "MergeConfig":
        """Load a config from a YAML file."""
        with open(path) as fh:
            data = yaml.safe_load(fh)
        return cls(**data)

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Serialise this config to a YAML file."""
        from dataclasses import asdict

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            yaml.dump(asdict(self), fh, default_flow_style=False)
