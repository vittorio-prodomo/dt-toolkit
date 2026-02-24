"""Merge multiple datasets into one with class remapping and conflict resolution.

Supports same-format and cross-format merging for YOLO and COCO datasets.
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from tqdm import tqdm

from ..classes.utils import normalize_class_name
from ..core.file_ops import FileOperations
from ..core.loader import DatasetLoader
from ..core.models import DatasetFormat, DatasetInfo, DatasetSplit
from ..formats.coco_utils import find_coco_image_file
from ..formats.yolo_utils import create_data_yaml, remap_yolo_annotations
from .config import MergeConfig

logger = logging.getLogger(__name__)


class DatasetMerger:
    """Merge multiple datasets into a single unified dataset.

    Usage::

        config = MergeConfig(
            label_mapping={
                "crack": ["mild_crack", "splitting_crack"],
                "spall": ["mechanical_spall", "spall"],
            },
        )
        merger = DatasetMerger(config)
        result = merger.merge(
            [Path("ds_a"), Path("ds_b")],
            Path("merged_output"),
        )
    """

    def __init__(
        self, config: MergeConfig, verbose: bool = True
    ) -> None:
        self.config = config
        self.verbose = verbose
        self.file_ops = FileOperations(verbose=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def merge(
        self,
        input_paths: List[Union[str, Path]],
        output_path: Union[str, Path],
        format: Optional[DatasetFormat] = None,
    ) -> DatasetInfo:
        """Merge datasets that share the same format.

        Args:
            input_paths: Paths to each source dataset.
            output_path: Where to write the merged result.
            format: Explicit format; auto-detected if *None*.
                All datasets must share this format.

        Returns:
            A :class:`DatasetInfo` for the merged dataset.
        """
        input_paths = [Path(p) for p in input_paths]
        output_path = Path(output_path)

        for p in input_paths:
            if not p.exists():
                raise FileNotFoundError(f"Dataset not found: {p}")

        # Detect / validate format
        fmt = self._resolve_format(input_paths, format)
        logger.info("Merging %d datasets (%s) -> %s", len(input_paths), fmt.value, output_path)

        # Load all dataset infos
        infos = self._load_all(input_paths, fmt)

        # Build class mapping
        class_map, final_classes, ds_id_maps = self._analyze_classes(infos)

        # Determine splits
        all_splits = self._resolve_splits(infos)

        # Check filename conflicts
        has_conflicts = self._has_filename_conflicts(infos, all_splits)

        try:
            if fmt == DatasetFormat.YOLO:
                self._merge_yolo(
                    infos, output_path, final_classes, ds_id_maps,
                    all_splits, has_conflicts,
                )
            else:
                self._merge_coco(
                    infos, output_path, final_classes, ds_id_maps,
                    all_splits, has_conflicts,
                )

            out_loader = DatasetLoader(output_path, verbose=False)
            result = out_loader.load_dataset(format=fmt)
            logger.info(
                "Merge complete: %d classes, %d images",
                result.num_classes,
                result.total_images,
            )
            return result
        except Exception:
            self.file_ops.rollback_operations()
            raise

    def merge_cross_format(
        self,
        input_paths: List[Union[str, Path]],
        output_path: Union[str, Path],
        output_format: DatasetFormat = DatasetFormat.COCO,
    ) -> DatasetInfo:
        """Merge datasets of mixed formats.

        YOLO datasets are converted to COCO first via
        :class:`~detection_toolkit.formats.YOLOToCOCOConverter`, then all
        COCO datasets are merged.  If *output_format* is ``YOLO``, the
        merged COCO result is converted back.

        Returns:
            A :class:`DatasetInfo` for the merged dataset.
        """
        import shutil
        import tempfile

        input_paths = [Path(p) for p in input_paths]
        output_path = Path(output_path)

        from ..formats.yolo_to_coco import YOLOToCOCOConverter
        converter = YOLOToCOCOConverter(verbose=self.verbose)

        temp_dirs: List[Path] = []
        coco_paths: List[Path] = []

        try:
            for ds_path in input_paths:
                loader = DatasetLoader(ds_path, verbose=False)
                fmt = loader.detect_format()
                if fmt == DatasetFormat.YOLO:
                    tmp = Path(tempfile.mkdtemp(prefix="merge_yolo2coco_"))
                    temp_dirs.append(tmp)
                    converter.convert(ds_path, tmp)
                    coco_paths.append(tmp)
                else:
                    coco_paths.append(ds_path)

            result = self.merge(coco_paths, output_path, DatasetFormat.COCO)

            if output_format == DatasetFormat.YOLO:
                from ..formats.coco_to_yolo import COCOToYOLOConverter
                yolo_conv = COCOToYOLOConverter(verbose=self.verbose)
                tmp_yolo = Path(tempfile.mkdtemp(prefix="merge_coco2yolo_"))
                temp_dirs.append(tmp_yolo)
                yolo_conv.convert(output_path, tmp_yolo)
                shutil.rmtree(output_path)
                shutil.move(str(tmp_yolo), str(output_path))
                temp_dirs.remove(tmp_yolo)

                out_loader = DatasetLoader(output_path, verbose=False)
                result = out_loader.load_dataset(format=DatasetFormat.YOLO)

            return result
        finally:
            for tmp in temp_dirs:
                if tmp.exists():
                    shutil.rmtree(tmp, ignore_errors=True)

    # ------------------------------------------------------------------
    # Format resolution
    # ------------------------------------------------------------------

    def _resolve_format(
        self,
        paths: List[Path],
        explicit: Optional[DatasetFormat],
    ) -> DatasetFormat:
        """Auto-detect or validate that all datasets share one format."""
        if explicit is not None:
            return explicit

        formats = set()
        for p in paths:
            loader = DatasetLoader(p, verbose=False)
            formats.add(loader.detect_format())

        if len(formats) > 1:
            raise ValueError(
                f"Mixed formats detected ({formats}). Use merge_cross_format() "
                "or specify format explicitly."
            )
        return formats.pop()

    def _load_all(
        self, paths: List[Path], fmt: DatasetFormat
    ) -> Dict[Path, DatasetInfo]:
        infos: Dict[Path, DatasetInfo] = {}
        for p in paths:
            loader = DatasetLoader(p, verbose=False)
            infos[p] = loader.load_dataset(format=fmt)
        return infos

    # ------------------------------------------------------------------
    # Class analysis
    # ------------------------------------------------------------------

    def _analyze_classes(
        self, infos: Dict[Path, DatasetInfo]
    ) -> Tuple[Dict[str, str], List[str], Dict[Path, Dict[int, int]]]:
        """Build the canonical class mapping and per-dataset ID maps.

        Returns:
            ``(name_map, final_classes, ds_id_maps)`` where:

            * *name_map*: ``{original_name: canonical_name}``
            * *final_classes*: sorted list of canonical names
            * *ds_id_maps*: ``{dataset_path: {old_id: new_id}}``
        """
        # Collect all original class names
        all_names: Set[str] = set()
        for info in infos.values():
            all_names.update(info.class_names.values())

        # Build variant → canonical lookup from the config
        variant_to_canonical: Dict[str, str] = {}
        for canonical, variants in self.config.label_mapping.items():
            canon_norm = normalize_class_name(canonical)
            variant_to_canonical[canon_norm] = canonical
            for v in variants:
                variant_to_canonical[normalize_class_name(v)] = canonical

        # Map every original name → canonical
        name_map: Dict[str, str] = {}
        for orig in all_names:
            norm = normalize_class_name(orig)
            if norm in variant_to_canonical:
                name_map[orig] = variant_to_canonical[norm]
            elif self.config.unmapped_classes == "include":
                name_map[orig] = orig
            elif self.config.unmapped_classes == "skip":
                logger.info("Skipping unmapped class: %s", orig)
            elif self.config.unmapped_classes == "error":
                raise ValueError(f"Unmapped class: {orig!r}")

        # Apply filter_classes
        if self.config.filter_classes:
            filter_set = set(self.config.filter_classes)
            name_map = {
                k: v for k, v in name_map.items() if v in filter_set
            }

        # Final sorted class list
        final_classes = sorted(set(name_map.values()))
        name_to_id = {n: i for i, n in enumerate(final_classes)}

        # Per-dataset old_id → new_id
        ds_id_maps: Dict[Path, Dict[int, int]] = {}
        for ds_path, info in infos.items():
            m: Dict[int, int] = {}
            for old_id, old_name in info.class_names.items():
                if old_name in name_map:
                    m[old_id] = name_to_id[name_map[old_name]]
                else:
                    m[old_id] = -1  # skip
            ds_id_maps[ds_path] = m

        if self.verbose:
            logger.info(
                "Class analysis: %d original -> %d canonical",
                len(all_names),
                len(final_classes),
            )

        return name_map, final_classes, ds_id_maps

    # ------------------------------------------------------------------
    # Splits
    # ------------------------------------------------------------------

    def _resolve_splits(
        self, infos: Dict[Path, DatasetInfo]
    ) -> Set[str]:
        """Determine which splits to include based on config."""
        all_splits: Set[str] = set()
        per_ds: Dict[Path, Set[str]] = {}
        for ds_path, info in infos.items():
            s = set(info.splits.keys())
            all_splits |= s
            per_ds[ds_path] = s

        if self.config.missing_splits == "include_all":
            result = all_splits
        elif self.config.missing_splits == "common_only":
            result = set.intersection(*per_ds.values()) if per_ds else set()
        elif self.config.missing_splits == "error":
            first = next(iter(per_ds.values()), set())
            for ds_path, s in per_ds.items():
                if s != first:
                    raise ValueError(
                        f"Split mismatch: {ds_path} has {sorted(s)}, "
                        f"expected {sorted(first)}"
                    )
            result = first
        else:
            result = all_splits

        if self.config.output_splits == "merge_to_train":
            return {"train"}

        return result

    # ------------------------------------------------------------------
    # Filename conflicts
    # ------------------------------------------------------------------

    def _has_filename_conflicts(
        self,
        infos: Dict[Path, DatasetInfo],
        splits: Set[str],
    ) -> bool:
        """Return True if any image filename appears in more than one dataset."""
        seen: Dict[str, Path] = {}
        for ds_path, info in infos.items():
            for split_name, split in info.splits.items():
                if split_name not in splits and self.config.output_splits != "merge_to_train":
                    continue
                for img in split.image_paths:
                    key = img.name
                    if key in seen and seen[key] != ds_path:
                        logger.info("Filename conflict detected: %s", key)
                        return True
                    seen[key] = ds_path
        return False

    def _resolve_filename(
        self, img_name: str, ds_path: Path, has_conflicts: bool
    ) -> str:
        """Return the output filename, possibly prefixed."""
        if not has_conflicts:
            return img_name
        if self.config.image_conflicts == "prefix_dataset":
            return f"{ds_path.name}_{img_name}"
        if self.config.image_conflicts == "error":
            raise ValueError(
                f"Filename conflict for '{img_name}' and "
                f"image_conflicts='error'"
            )
        return img_name

    # ------------------------------------------------------------------
    # YOLO merge
    # ------------------------------------------------------------------

    def _merge_yolo(
        self,
        infos: Dict[Path, DatasetInfo],
        output: Path,
        final_classes: List[str],
        ds_id_maps: Dict[Path, Dict[int, int]],
        splits: Set[str],
        has_conflicts: bool,
    ) -> None:
        merge_all = self.config.output_splits == "merge_to_train"

        # Create directories
        for split in splits:
            (output / "images" / split).mkdir(parents=True, exist_ok=True)
            (output / "labels" / split).mkdir(parents=True, exist_ok=True)

        for ds_path, info in infos.items():
            id_map = ds_id_maps[ds_path]

            for split_name, split in info.splits.items():
                out_split = "train" if merge_all else split_name
                if out_split not in splits:
                    continue

                label_index = {lp.stem: lp for lp in split.label_paths}

                for img in tqdm(
                    split.image_paths,
                    desc=f"  {ds_path.name}/{split_name}",
                    disable=not self.verbose,
                ):
                    new_name = self._resolve_filename(
                        img.name, ds_path, has_conflicts
                    )
                    dst_img = output / "images" / out_split / new_name
                    self.file_ops.create_file_link(img, dst_img)

                    src_lbl = label_index.get(img.stem)
                    if src_lbl and src_lbl.exists():
                        new_stem = Path(new_name).stem
                        dst_lbl = (
                            output / "labels" / out_split / (new_stem + ".txt")
                        )
                        remap_yolo_annotations(
                            src_lbl, dst_lbl, id_map, skip_unmapped=True
                        )

        create_data_yaml(output, final_classes, sorted(splits))

    # ------------------------------------------------------------------
    # COCO merge
    # ------------------------------------------------------------------

    def _merge_coco(
        self,
        infos: Dict[Path, DatasetInfo],
        output: Path,
        final_classes: List[str],
        ds_id_maps: Dict[Path, Dict[int, int]],
        splits: Set[str],
        has_conflicts: bool,
    ) -> None:
        (output / "images").mkdir(parents=True, exist_ok=True)
        (output / "annotations").mkdir(parents=True, exist_ok=True)

        merge_all = self.config.output_splits == "merge_to_train"
        cats = [
            {"id": i, "name": n, "supercategory": ""}
            for i, n in enumerate(final_classes)
        ]

        # Group work by output split
        split_data: Dict[str, Dict] = {}
        for out_split in splits:
            split_data[out_split] = {
                "images": [],
                "annotations": [],
                "categories": cats,
            }

        img_counter = {s: 1 for s in splits}
        ann_counter = {s: 1 for s in splits}

        for ds_path, info in infos.items():
            id_map = ds_id_maps[ds_path]

            for split_name, split in info.splits.items():
                out_split = "train" if merge_all else split_name
                if out_split not in splits:
                    continue
                if split.coco_ann_path is None:
                    continue

                with open(split.coco_ann_path) as fh:
                    coco = json.load(fh)

                old_to_new_img: Dict[int, int] = {}
                root = split.coco_images_root or info.root_dir

                for img_entry in tqdm(
                    coco.get("images", []),
                    desc=f"  {ds_path.name}/{split_name}",
                    disable=not self.verbose,
                ):
                    old_fn = img_entry["file_name"]
                    new_fn = self._resolve_filename(
                        old_fn, ds_path, has_conflicts
                    )

                    src = find_coco_image_file(old_fn, root)
                    if src is None:
                        continue

                    dst = output / "images" / new_fn
                    self.file_ops.create_file_link(src, dst)

                    new_img_id = img_counter[out_split]
                    img_counter[out_split] += 1

                    new_entry = img_entry.copy()
                    new_entry["id"] = new_img_id
                    new_entry["file_name"] = new_fn
                    split_data[out_split]["images"].append(new_entry)
                    old_to_new_img[img_entry["id"]] = new_img_id

                for ann in coco.get("annotations", []):
                    new_img_id = old_to_new_img.get(ann["image_id"])
                    if new_img_id is None:
                        continue
                    new_cid = id_map.get(ann["category_id"], -1)
                    if new_cid < 0:
                        continue

                    new_ann = ann.copy()
                    new_ann["id"] = ann_counter[out_split]
                    ann_counter[out_split] += 1
                    new_ann["image_id"] = new_img_id
                    new_ann["category_id"] = new_cid
                    split_data[out_split]["annotations"].append(new_ann)

        # Write per-split JSONs
        for out_split, data in split_data.items():
            out_json = output / "annotations" / f"instances_{out_split}.json"
            with open(out_json, "w") as fh:
                json.dump(data, fh, indent=2)
            logger.info(
                "Split %s: %d images, %d annotations",
                out_split,
                len(data["images"]),
                len(data["annotations"]),
            )
