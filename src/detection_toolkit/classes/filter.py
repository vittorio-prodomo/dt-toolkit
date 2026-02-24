"""Filter classes within a dataset (keep or remove specific classes).

Supports both YOLO and COCO formats.  Images are always preserved
(those that lose all annotations become negative samples).
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

from tqdm import tqdm

from ..core.file_ops import FileOperations
from ..core.loader import DatasetLoader
from ..core.models import DatasetFormat, DatasetInfo, DatasetSplit
from ..formats.coco_utils import find_coco_image_file
from ..formats.yolo_utils import create_data_yaml, remap_yolo_annotations

logger = logging.getLogger(__name__)


@dataclass
class ClassFilterConfig:
    """Configuration for class filtering.

    Args:
        action: ``"keep"`` (whitelist) or ``"remove"`` (blacklist).
        classes: Class names to keep or remove.
    """

    action: str
    classes: List[str]

    def __post_init__(self) -> None:
        if self.action not in ("keep", "remove"):
            raise ValueError(
                f"action must be 'keep' or 'remove', got {self.action!r}"
            )
        if not self.classes:
            raise ValueError("At least one class must be specified")

    def validate(self, available: Set[str]) -> None:
        """Raise ``ValueError`` if any listed class is missing from *available*."""
        missing = set(self.classes) - available
        if missing:
            raise ValueError(
                f"Classes not found in dataset: {sorted(missing)}. "
                f"Available: {sorted(available)}"
            )

    def classes_to_keep(self, all_classes: Set[str]) -> Set[str]:
        """Return the set of classes that survive filtering."""
        if self.action == "keep":
            return set(self.classes) & all_classes
        return all_classes - set(self.classes)


class ClassFilter:
    """Filter classes in a YOLO or COCO dataset.

    Usage::

        cfg = ClassFilterConfig(action="keep", classes=["crack", "spall"])
        flt = ClassFilter(cfg)
        flt.filter(Path("input"), Path("output"))
    """

    def __init__(
        self, config: ClassFilterConfig, verbose: bool = True
    ) -> None:
        self.config = config
        self.verbose = verbose
        self.file_ops = FileOperations(verbose=False)

    def filter(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        format: Optional[DatasetFormat] = None,
    ) -> DatasetInfo:
        """Filter classes and write the result to *output_path*.

        Args:
            input_path: Root of the source dataset.
            output_path: Where to write the filtered dataset.
            format: Explicit format; auto-detected if *None*.

        Returns:
            A :class:`DatasetInfo` describing the filtered dataset.
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        loader = DatasetLoader(input_path, verbose=self.verbose)
        info = loader.load_dataset(format=format)

        available = set(info.class_names.values())
        self.config.validate(available)
        keep = self.config.classes_to_keep(available)

        logger.info(
            "Filtering %s dataset: %d -> %d classes",
            info.format.value.upper(),
            len(available),
            len(keep),
        )

        try:
            if info.format == DatasetFormat.YOLO:
                self._filter_yolo(info, output_path, keep)
            else:
                self._filter_coco(info, output_path, keep)

            out_loader = DatasetLoader(output_path, verbose=False)
            return out_loader.load_dataset(format=info.format)
        except Exception:
            self.file_ops.rollback_operations()
            raise

    # ------------------------------------------------------------------
    # YOLO
    # ------------------------------------------------------------------

    def _filter_yolo(
        self,
        info: DatasetInfo,
        output: Path,
        keep: Set[str],
    ) -> None:
        # Build old_id -> new_id mapping.  -1 = drop.
        sorted_keep = sorted(keep)
        name_to_new = {n: i for i, n in enumerate(sorted_keep)}
        id_map: Dict[int, int] = {}
        for old_id, name in info.class_names.items():
            id_map[old_id] = name_to_new.get(name, -1)

        for split_name, split in info.splits.items():
            img_dir = output / "images" / split_name
            lbl_dir = output / "labels" / split_name
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)

            label_index = {lp.stem: lp for lp in split.label_paths}

            for img in tqdm(
                split.image_paths,
                desc=f"  {split_name}",
                disable=not self.verbose,
            ):
                self.file_ops.create_file_link(
                    img, img_dir / img.name
                )
                src_lbl = label_index.get(img.stem)
                if src_lbl and src_lbl.exists():
                    dst_lbl = lbl_dir / (img.stem + ".txt")
                    written = remap_yolo_annotations(
                        src_lbl, dst_lbl, id_map, skip_unmapped=True
                    )
                    # If nothing was written, remove the empty file
                    if written == 0 and dst_lbl.exists():
                        dst_lbl.unlink()

        create_data_yaml(output, sorted_keep, list(info.splits.keys()))

    # ------------------------------------------------------------------
    # COCO
    # ------------------------------------------------------------------

    def _filter_coco(
        self,
        info: DatasetInfo,
        output: Path,
        keep: Set[str],
    ) -> None:
        (output / "images").mkdir(parents=True, exist_ok=True)
        (output / "annotations").mkdir(parents=True, exist_ok=True)

        sorted_keep = sorted(keep)
        name_to_new = {n: i for i, n in enumerate(sorted_keep)}
        id_map: Dict[int, int] = {}
        for old_id, name in info.class_names.items():
            id_map[old_id] = name_to_new.get(name, -1)

        for split_name, split in info.splits.items():
            if split.coco_ann_path is None:
                continue

            with open(split.coco_ann_path) as fh:
                coco = json.load(fh)

            # Filter annotations
            new_anns = []
            for ann in coco.get("annotations", []):
                new_cid = id_map.get(ann["category_id"], -1)
                if new_cid >= 0:
                    a = ann.copy()
                    a["category_id"] = new_cid
                    new_anns.append(a)

            # Link images (all images kept — empty ones become negatives)
            root = split.coco_images_root or info.root_dir
            for img_entry in tqdm(
                coco.get("images", []),
                desc=f"  {split_name}",
                disable=not self.verbose,
            ):
                src = find_coco_image_file(img_entry["file_name"], root)
                if src:
                    self.file_ops.create_file_link(
                        src, output / "images" / img_entry["file_name"]
                    )

            # Write filtered JSON
            new_cats = [
                {"id": i, "name": n, "supercategory": ""}
                for i, n in enumerate(sorted_keep)
            ]
            out_json = (
                output / "annotations" / f"instances_{split_name}.json"
            )
            with open(out_json, "w") as fh:
                json.dump(
                    {
                        "images": coco.get("images", []),
                        "annotations": new_anns,
                        "categories": new_cats,
                    },
                    fh,
                    indent=2,
                )

            logger.info(
                "Split %s: %d images, %d -> %d annotations",
                split_name,
                len(coco.get("images", [])),
                len(coco.get("annotations", [])),
                len(new_anns),
            )
