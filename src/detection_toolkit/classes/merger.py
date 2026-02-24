"""Merge semantically similar classes within a single dataset.

For example, merge ``mild_crack`` + ``splitting_crack`` into ``crack``.
Supports both YOLO and COCO formats.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

from tqdm import tqdm

from ..core.file_ops import FileOperations
from ..core.loader import DatasetLoader
from ..core.models import DatasetFormat, DatasetInfo
from ..formats.coco_utils import find_coco_image_file
from ..formats.yolo_utils import create_data_yaml, remap_yolo_annotations

logger = logging.getLogger(__name__)


class ClassMerger:
    """Merge multiple class names into canonical names within one dataset.

    Args:
        config: ``{canonical_name: [variant_names]}``.
            Every *variant_name* found in the dataset will be remapped to
            *canonical_name*.  Classes not mentioned are kept as-is.

    Usage::

        merger = ClassMerger({
            "crack": ["mild_crack", "splitting_crack", "crack_map"],
            "spall": ["mechanical_spall", "spall"],
        })
        merger.merge(Path("input"), Path("output"))
    """

    def __init__(
        self,
        config: Dict[str, List[str]],
        verbose: bool = True,
    ) -> None:
        self.config = config
        self.verbose = verbose
        self.file_ops = FileOperations(verbose=False)

    def merge(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        format: Optional[DatasetFormat] = None,
    ) -> DatasetInfo:
        """Run the class merge and write to *output_path*.

        Returns:
            A :class:`DatasetInfo` describing the output dataset.
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        loader = DatasetLoader(input_path, verbose=self.verbose)
        info = loader.load_dataset(format=format)

        self._validate(info)
        id_map, final_classes = self._build_mapping(info)

        logger.info(
            "Merging classes: %d -> %d (%s)",
            len(info.class_names),
            len(final_classes),
            ", ".join(final_classes),
        )

        try:
            if info.format == DatasetFormat.YOLO:
                self._merge_yolo(info, output_path, id_map, final_classes)
            else:
                self._merge_coco(info, output_path, id_map, final_classes)

            out_loader = DatasetLoader(output_path, verbose=False)
            return out_loader.load_dataset(format=info.format)
        except Exception:
            self.file_ops.rollback_operations()
            raise

    # ------------------------------------------------------------------
    # Mapping
    # ------------------------------------------------------------------

    def _validate(self, info: DatasetInfo) -> None:
        """Ensure every variant listed in the config exists in the dataset."""
        available = set(info.class_names.values())
        for target, variants in self.config.items():
            for v in variants:
                if v not in available:
                    raise ValueError(
                        f"Variant '{v}' (target='{target}') not found. "
                        f"Available: {sorted(available)}"
                    )

    def _build_mapping(
        self, info: DatasetInfo
    ) -> Tuple[Dict[int, int], List[str]]:
        """Build ``{old_class_id: new_class_id}`` and the final class list.

        Returns:
            ``(id_map, final_classes)``
        """
        # Resolve each original name to its canonical name
        name_map: Dict[str, str] = {}
        for target, variants in self.config.items():
            for v in variants:
                name_map[v] = target
            # The target itself might also be an existing class name
            if target in set(info.class_names.values()):
                name_map[target] = target

        # Build the final class set
        final_names: Set[str] = set()
        for _id, name in info.class_names.items():
            final_names.add(name_map.get(name, name))

        final_classes = sorted(final_names)
        name_to_new_id = {n: i for i, n in enumerate(final_classes)}

        id_map: Dict[int, int] = {}
        for old_id, old_name in info.class_names.items():
            canonical = name_map.get(old_name, old_name)
            id_map[old_id] = name_to_new_id[canonical]

        return id_map, final_classes

    # ------------------------------------------------------------------
    # YOLO
    # ------------------------------------------------------------------

    def _merge_yolo(
        self,
        info: DatasetInfo,
        output: Path,
        id_map: Dict[int, int],
        final_classes: List[str],
    ) -> None:
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
                self.file_ops.create_file_link(img, img_dir / img.name)
                src_lbl = label_index.get(img.stem)
                if src_lbl and src_lbl.exists():
                    dst_lbl = lbl_dir / (img.stem + ".txt")
                    remap_yolo_annotations(
                        src_lbl, dst_lbl, id_map, skip_unmapped=False
                    )

        create_data_yaml(output, final_classes, list(info.splits.keys()))

    # ------------------------------------------------------------------
    # COCO
    # ------------------------------------------------------------------

    def _merge_coco(
        self,
        info: DatasetInfo,
        output: Path,
        id_map: Dict[int, int],
        final_classes: List[str],
    ) -> None:
        (output / "images").mkdir(parents=True, exist_ok=True)
        (output / "annotations").mkdir(parents=True, exist_ok=True)

        new_cats = [
            {"id": i, "name": n, "supercategory": ""}
            for i, n in enumerate(final_classes)
        ]

        for split_name, split in info.splits.items():
            if split.coco_ann_path is None:
                continue

            with open(split.coco_ann_path) as fh:
                coco = json.load(fh)

            # Remap annotations
            new_anns = []
            for ann in coco.get("annotations", []):
                new_cid = id_map.get(ann["category_id"])
                if new_cid is not None:
                    a = ann.copy()
                    a["category_id"] = new_cid
                    new_anns.append(a)

            # Link images
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
                "Split %s: %d images, %d annotations",
                split_name,
                len(coco.get("images", [])),
                len(new_anns),
            )
