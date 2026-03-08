"""High-level dataset tiler for YOLO and COCO datasets.

Orchestrates tile grid computation, parallel image/label processing,
and negative-sample control.
"""

import json
import logging
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from tqdm import tqdm

from ..core.loader import DatasetLoader
from ..core.models import DatasetFormat, DatasetInfo
from ..formats.yolo_utils import create_data_yaml
from .config import TilingConfig
from .geometry import (
    bbox_area_ratio,
    bbox_to_yolo_center,
    clip_bbox_to_tile,
    clip_polygon_to_tile,
    compute_tile_grid,
    normalize_to_yolo,
    polygon_area_ratio,
    to_tile_coords,
)
from .parallel import optimal_process_count, tile_single_yolo_image, tile_single_image, tile_single_fo_sample

logger = logging.getLogger(__name__)


class DatasetTiler:
    """Tile a YOLO, COCO, or unlabeled image dataset into fixed-size patches.

    Usage::

        config = TilingConfig(tile_size=640, overlap_percent=0.3)
        tiler = DatasetTiler(config)
        result = tiler.tile(Path("input_dataset"), Path("output_tiled"))

    Args:
        config: Tiling configuration.
        verbose: Show progress bars and informational messages.
    """

    def __init__(
        self, config: TilingConfig, verbose: bool = True
    ) -> None:
        self.config = config
        self.verbose = verbose

    def tile(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        format: Optional[DatasetFormat] = None,
    ) -> Optional[DatasetInfo]:
        """Tile a dataset and write to *output_path*.

        If *input_path* is a recognised YOLO/COCO dataset it is tiled with
        annotations.  Otherwise it is treated as an unlabeled image
        directory.

        Args:
            input_path: Source dataset or image directory.
            output_path: Where to write the tiled output.
            format: Explicit format. Auto-detected if *None*.

        Returns:
            A :class:`DatasetInfo` for labeled datasets, or *None* for
            unlabeled image directories.
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        # Try labeled dataset first
        try:
            loader = DatasetLoader(input_path, verbose=False)
            info = loader.load_dataset(format=format)
        except (FileNotFoundError, ValueError):
            if format is not None:
                raise  # user explicitly requested a format — don't swallow
            logger.info(
                "No labeled dataset detected — tiling as unlabeled images"
            )
            self._tile_unlabeled(input_path, output_path)
            return None

        if self.config.flatten:
            logger.warning("--flatten is ignored for labeled datasets")

        logger.info(
            "Tiling %s dataset (%d images, tile_size=%d, overlap=%.0f%%)",
            info.format.value,
            info.total_images,
            self.config.tile_size,
            self.config.overlap_percent * 100,
        )

        pad_width = self._calc_pad_width(info)

        if info.format == DatasetFormat.YOLO:
            self._tile_yolo(info, output_path, pad_width)
        else:
            self._tile_coco(info, output_path, pad_width)

        out_loader = DatasetLoader(output_path, verbose=False)
        result = out_loader.load_dataset(format=info.format)
        logger.info(
            "Tiling complete: %d images -> %d tiles",
            info.total_images,
            result.total_images,
        )
        return result

    # ------------------------------------------------------------------
    # FiftyOne dataset tiling
    # ------------------------------------------------------------------

    def tile_fo(
        self,
        dataset,
        output_path: Union[str, Path],
        output_format: str = "yolo",
        label_field: Optional[str] = None,
        reimport: bool = True,
        dataset_name: Optional[str] = None,
    ) -> Optional[DatasetInfo]:
        """Tile a FiftyOne dataset and write to disk.

        Args:
            dataset: FiftyOne Dataset or DatasetView.
            output_path: Where to write the tiled output.
            output_format: ``"yolo"`` or ``"coco"``.
            label_field: Label field name; auto-resolved if *None*.
            reimport: Reimport tiled result into FiftyOne after writing.
            dataset_name: Name for reimported dataset. Defaults to
                ``"{source}_tiled_{tile_size}"``.

        Returns:
            A :class:`DatasetInfo` for the tiled output, or *None* if
            reimport is disabled.
        """
        # Lazy imports — fiftyone_hub.utils imports fiftyone at module level.
        # We bind into the module namespace so that unittest.mock.patch can
        # intercept them in tests (with create=True).  If a mock has already
        # set them, we honour the mock instead.
        import detection_toolkit.tiling.tiler as _self_mod
        if not hasattr(_self_mod, "resolve_label_field"):
            from ..fiftyone_hub.utils import resolve_label_field as _rlf
            _self_mod.resolve_label_field = _rlf
        if not hasattr(_self_mod, "_detect_container_type"):
            from ..fiftyone_hub.utils import _detect_container_type as _dct
            _self_mod._detect_container_type = _dct

        from .fo_extract import build_class_map, extract_annotations

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Resolve label field and container type
        resolved_field = _self_mod.resolve_label_field(dataset, label_field)
        container_type = _self_mod._detect_container_type(dataset, resolved_field)

        # Build class mapping
        class_map = build_class_map(dataset.classes)
        class_list = list(class_map.keys())
        if not class_list:
            logger.warning("No classes found in dataset — tiling without annotations")

        # Determine splits
        has_splits = dataset.has_sample_field("split")
        if has_splits:
            split_names = dataset.distinct("split")
        else:
            split_names = ["train"]

        # Bulk-extract all fields
        sub_field = f"{resolved_field}.{container_type}"
        fields = ["filepath", "metadata.width", "metadata.height", sub_field]
        if has_splits:
            fields.append("split")

        bulk_data = dataset.values(fields)
        filepaths = bulk_data[0]
        widths = bulk_data[1]
        heights = bulk_data[2]
        all_labels = bulk_data[3]
        splits = bulk_data[4] if has_splits else ["train"] * len(filepaths)

        num_samples = len(filepaths)
        logger.info(
            "Tiling FiftyOne dataset '%s' (%d samples, tile_size=%d, format=%s)",
            getattr(dataset, "name", "view"),
            num_samples,
            self.config.tile_size,
            output_format,
        )

        # Convert annotations to plain-Python dicts per sample
        sample_annotations = []
        for i in range(num_samples):
            anns = extract_annotations(
                all_labels[i],
                container_type=container_type,
                class_map=class_map,
                img_w=widths[i] or 0,
                img_h=heights[i] or 0,
            )
            sample_annotations.append(anns)

        # Estimate pad width
        pad_width = self._estimate_pad_width(
            [Path(fp) for fp in filepaths[:5]]
        )

        cfg_dict = self.config.to_dict()

        # Group by split and dispatch workers
        for split_name in split_names:
            indices = [
                i for i in range(num_samples) if splits[i] == split_name
            ]
            if not indices:
                continue

            tasks = []
            for i in indices:
                tasks.append((
                    filepaths[i],
                    sample_annotations[i],
                    str(output_path),
                    split_name,
                    Path(filepaths[i]).stem,
                    pad_width,
                    class_list,
                    cfg_dict,
                    output_format,
                ))

            num_procs = optimal_process_count(
                len(tasks), self.config.max_processes,
            )

            total_positive = 0
            total_negative = 0
            all_coco_entries: List[Dict] = []

            if num_procs <= 1:
                for task in tqdm(
                    tasks,
                    desc=f"  {split_name}",
                    disable=not self.verbose,
                ):
                    pos, neg, coco_entries = tile_single_fo_sample(task)
                    total_positive += pos
                    total_negative += neg
                    all_coco_entries.extend(coco_entries)
            else:
                logger.info(
                    "Tiling %s with %d workers (%d samples)",
                    split_name, num_procs, len(tasks),
                )
                with ProcessPoolExecutor(max_workers=num_procs) as pool:
                    futures = {
                        pool.submit(tile_single_fo_sample, t): t
                        for t in tasks
                    }
                    for future in tqdm(
                        as_completed(futures),
                        total=len(futures),
                        desc=f"  {split_name}",
                        disable=not self.verbose,
                    ):
                        pos, neg, coco_entries = future.result()
                        total_positive += pos
                        total_negative += neg
                        all_coco_entries.extend(coco_entries)

            # Post-process output
            if output_format == "yolo":
                lbl_dir = output_path / "labels" / split_name
                img_dir = output_path / "images" / split_name
                kept_neg = self._apply_negative_sampling(
                    lbl_dir, img_dir, total_positive, total_negative,
                )
                logger.info(
                    "Split %s: %d positive, %d negative kept",
                    split_name, total_positive, kept_neg,
                )
            elif output_format == "coco":
                self._assemble_coco_json(
                    all_coco_entries, class_map, split_name, output_path,
                    total_positive, total_negative,
                )

        # Write data.yaml for YOLO
        if output_format == "yolo":
            class_names = {v: k for k, v in class_map.items()}
            create_data_yaml(output_path, class_names, split_names)

        logger.info("FiftyOne tiling complete -> %s", output_path)

        # Reimport
        result = None
        if reimport:
            result = self._reimport_fo(
                output_path, output_format, dataset, dataset_name,
            )

        return result

    def _assemble_coco_json(
        self,
        coco_entries: List[Dict],
        class_map: Dict[str, int],
        split_name: str,
        output_path: Path,
        total_positive: int,
        total_negative: int,
    ) -> None:
        """Assemble COCO JSON from worker entries and apply negative sampling."""
        categories = [
            {"id": idx + 1, "name": name, "supercategory": ""}
            for name, idx in sorted(class_map.items(), key=lambda x: x[1])
        ]

        images = []
        annotations = []
        img_counter = 1
        ann_counter = 1

        for entry in coco_entries:
            img_entry = entry["image"]
            img_entry["id"] = img_counter
            images.append(img_entry)

            for ann in entry["annotations"]:
                ann["id"] = ann_counter
                ann["image_id"] = img_counter
                annotations.append(ann)
                ann_counter += 1

            img_counter += 1

        # Apply negative sampling
        images, annotations = self._apply_negative_sampling_coco(
            images, annotations, total_positive, total_negative,
        )

        ann_dir = output_path / "annotations"
        ann_dir.mkdir(parents=True, exist_ok=True)
        out_json = ann_dir / f"instances_{split_name}.json"
        with open(out_json, "w") as fh:
            json.dump(
                {
                    "images": images,
                    "annotations": annotations,
                    "categories": categories,
                },
                fh,
                indent=2,
            )

        logger.info(
            "Split %s: %d tiles (%d annotated)",
            split_name, len(images), total_positive,
        )

    def _reimport_fo(
        self,
        output_path: Path,
        output_format: str,
        source_dataset,
        dataset_name: Optional[str],
    ) -> Optional[DatasetInfo]:
        """Reimport tiled output into FiftyOne."""
        from ..fiftyone_hub.loader import FiftyOneDatasetLoader

        if dataset_name is None:
            src_name = getattr(source_dataset, "name", "dataset")
            dataset_name = f"{src_name}_tiled_{self.config.tile_size}"

        fmt = DatasetFormat(output_format)
        loader = FiftyOneDatasetLoader(verbose=self.verbose)

        logger.info(
            "Reimporting tiled dataset as '%s'", dataset_name,
        )
        loader.load(
            output_path,
            dataset_name,
            format=fmt,
            persistent=True,
            overwrite=True,
        )

        # Load and return DatasetInfo
        out_loader = DatasetLoader(output_path, verbose=False)
        return out_loader.load_dataset(format=fmt)

    # ------------------------------------------------------------------
    # Padding width calculation
    # ------------------------------------------------------------------

    def _calc_pad_width(self, info: DatasetInfo) -> int:
        """Calculate zero-padding width for tile index in filenames."""
        sample_paths: List[Path] = []
        for split in info.splits.values():
            sample_paths.extend(split.image_paths[:5])
        return self._estimate_pad_width(sample_paths)

    # ------------------------------------------------------------------
    # Unlabeled tiling (parallel)
    # ------------------------------------------------------------------

    def _tile_unlabeled(
        self,
        input_path: Path,
        output_path: Path,
    ) -> None:
        """Tile an unlabeled image directory using multiprocessing."""
        from ..core.image_utils import find_images_in_dir

        images = self._sample_images(find_images_in_dir(input_path))
        if not images:
            logger.warning("No images found in %s", input_path)
            return

        logger.info(
            "Tiling %d unlabeled images (tile_size=%d, overlap=%.0f%%, flatten=%s)",
            len(images),
            self.config.tile_size,
            self.config.overlap_percent * 100,
            self.config.flatten,
        )

        pad_width = self._calc_pad_width_unlabeled(images)
        cfg_dict = self.config.to_dict()

        # Build worker args
        tasks = []
        for img in images:
            rel = img.relative_to(input_path)
            if self.config.flatten:
                # Prefix from relative path for disambiguation
                parts = rel.parent.parts
                prefix = "_".join(parts) + "_" + rel.stem if parts else rel.stem
                out_dir = str(output_path)
            else:
                out_dir = str(output_path / rel.parent)
                prefix = rel.stem
            tasks.append((str(img), out_dir, prefix, pad_width, cfg_dict))

        num_procs = optimal_process_count(len(tasks), self.config.max_processes)
        total_tiles = 0

        if num_procs <= 1:
            for task in tqdm(
                tasks, desc="  tiling", disable=not self.verbose,
            ):
                total_tiles += tile_single_image(task)
        else:
            logger.info(
                "Tiling with %d workers (%d images)", num_procs, len(tasks),
            )
            with ProcessPoolExecutor(max_workers=num_procs) as pool:
                futures = {
                    pool.submit(tile_single_image, t): t for t in tasks
                }
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="  tiling",
                    disable=not self.verbose,
                ):
                    total_tiles += future.result()

        logger.info(
            "Tiling complete: %d images -> %d tiles", len(images), total_tiles,
        )

    def _calc_pad_width_unlabeled(self, images: List[Path]) -> int:
        """Calculate zero-padding width for tile index in filenames."""
        return self._estimate_pad_width(images)

    def _estimate_pad_width(self, sample_paths: List[Path]) -> int:
        """Estimate zero-pad width from a sample of image paths."""
        from ..core.image_utils import get_image_size

        max_tiles = 0
        for img_path in sample_paths[:5]:
            try:
                w, h = get_image_size(img_path)
                n = len(compute_tile_grid(
                    w, h, self.config.tile_size,
                    self.config.overlap_percent, self.config.edge_behavior,
                ))
                max_tiles = max(max_tiles, n)
            except Exception:
                pass
        return max(1, len(str(max(max_tiles - 1, 0))))

    # ------------------------------------------------------------------
    # YOLO tiling (parallel)
    # ------------------------------------------------------------------

    def _tile_yolo(
        self,
        info: DatasetInfo,
        output: Path,
        pad_width: int,
    ) -> None:
        """Tile all splits of a YOLO dataset using multiprocessing."""
        cfg_dict = self.config.to_dict()

        for split_name, split in info.splits.items():
            img_dir = output / "images" / split_name
            lbl_dir = output / "labels" / split_name
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)

            images = self._sample_images(split.image_paths)
            label_index = {lp.stem: lp for lp in split.label_paths}

            # Build worker args
            tasks = []
            for img in images:
                lbl = label_index.get(img.stem)
                tasks.append((
                    str(img),
                    str(lbl) if lbl else None,
                    str(output),
                    split_name,
                    pad_width,
                    cfg_dict,
                ))

            num_procs = optimal_process_count(
                len(tasks), self.config.max_processes,
            )

            total_positive = 0
            total_negative = 0

            if num_procs <= 1:
                # Serial fallback
                for task in tqdm(
                    tasks,
                    desc=f"  {split_name}",
                    disable=not self.verbose,
                ):
                    pos, neg = tile_single_yolo_image(task)
                    total_positive += pos
                    total_negative += neg
            else:
                logger.info(
                    "Tiling %s with %d workers (%d images)",
                    split_name, num_procs, len(tasks),
                )
                with ProcessPoolExecutor(max_workers=num_procs) as pool:
                    futures = {
                        pool.submit(tile_single_yolo_image, t): t
                        for t in tasks
                    }
                    for future in tqdm(
                        as_completed(futures),
                        total=len(futures),
                        desc=f"  {split_name}",
                        disable=not self.verbose,
                    ):
                        pos, neg = future.result()
                        total_positive += pos
                        total_negative += neg

            # Apply negative sampling
            kept_neg = self._apply_negative_sampling(
                lbl_dir, img_dir, total_positive, total_negative,
            )

            logger.info(
                "Split %s: %d positive tiles, %d negative tiles kept",
                split_name, total_positive, kept_neg,
            )

        create_data_yaml(output, info.class_names, list(info.splits.keys()))

    # ------------------------------------------------------------------
    # COCO tiling (serial)
    # ------------------------------------------------------------------

    def _tile_coco(
        self,
        info: DatasetInfo,
        output: Path,
        pad_width: int,
    ) -> None:
        """Tile all splits of a COCO dataset (serial processing)."""
        (output / "annotations").mkdir(parents=True, exist_ok=True)

        cats = [
            {"id": cid, "name": cname, "supercategory": ""}
            for cid, cname in sorted(info.class_names.items())
        ]

        for split_name, split in info.splits.items():
            img_dir = output / "images" / split_name
            img_dir.mkdir(parents=True, exist_ok=True)

            if split.coco_ann_path is None:
                continue

            with open(split.coco_ann_path) as fh:
                coco = json.load(fh)

            # Build lookups
            img_map = {img["id"]: img for img in coco.get("images", [])}
            anns_by_img: Dict[int, List[Dict]] = {}
            for ann in coco.get("annotations", []):
                anns_by_img.setdefault(ann["image_id"], []).append(ann)

            images_list = list(img_map.values())
            images_list = self._sample_images_list(images_list)

            root = split.coco_images_root or info.root_dir
            new_images: List[Dict] = []
            new_annotations: List[Dict] = []
            img_counter = 1
            ann_counter = 1
            total_positive = 0
            total_negative = 0

            for img_entry in tqdm(
                images_list,
                desc=f"  {split_name}",
                disable=not self.verbose,
            ):
                img_id = img_entry["id"]
                img_anns = anns_by_img.get(img_id, [])
                img_w = img_entry["width"]
                img_h = img_entry["height"]

                from ..formats.coco_utils import find_coco_image_file
                src_file = find_coco_image_file(img_entry["file_name"], root)
                if src_file is None:
                    continue

                tiles = compute_tile_grid(
                    img_w, img_h, self.config.tile_size,
                    self.config.overlap_percent, self.config.edge_behavior,
                )

                for idx, (tx, ty, tw, th) in enumerate(tiles):
                    tile_name = f"{src_file.stem}_tile_{idx:0{pad_width}d}"
                    tile_ext = src_file.suffix

                    # Clip annotations
                    tile_anns = self._clip_coco_annotations(
                        img_anns, tx, ty, tw, th, img_w, img_h,
                    )

                    # Write tile image
                    tile_img_path = img_dir / f"{tile_name}{tile_ext}"
                    self._write_coco_tile_image(
                        src_file, tx, ty, tw, th, tile_img_path,
                    )

                    # Add to output JSON
                    new_img_id = img_counter
                    img_counter += 1

                    new_images.append({
                        "id": new_img_id,
                        "file_name": f"{tile_name}{tile_ext}",
                        "width": tw,
                        "height": th,
                    })

                    if tile_anns:
                        total_positive += 1
                        for ta in tile_anns:
                            ta["id"] = ann_counter
                            ann_counter += 1
                            ta["image_id"] = new_img_id
                            new_annotations.append(ta)
                    else:
                        total_negative += 1

            # Apply negative sampling to the COCO output
            new_images, new_annotations = self._apply_negative_sampling_coco(
                new_images, new_annotations, total_positive, total_negative,
            )

            out_json = output / "annotations" / f"instances_{split_name}.json"
            with open(out_json, "w") as fh:
                json.dump(
                    {
                        "images": new_images,
                        "annotations": new_annotations,
                        "categories": cats,
                    },
                    fh,
                    indent=2,
                )

            logger.info(
                "Split %s: %d tiles (%d annotated)",
                split_name,
                len(new_images),
                total_positive,
            )

    # ------------------------------------------------------------------
    # COCO annotation clipping
    # ------------------------------------------------------------------

    def _clip_coco_annotations(
        self,
        annotations: List[Dict],
        tx: int,
        ty: int,
        tw: int,
        th: int,
        img_w: int,
        img_h: int,
    ) -> List[Dict]:
        """Clip COCO annotations to a tile, returning new annotation dicts."""
        result: List[Dict] = []

        for ann in annotations:
            seg = ann.get("segmentation")

            if seg and isinstance(seg, list) and seg:
                # Polygon segmentation
                for flat_poly in seg:
                    points = [
                        (flat_poly[i], flat_poly[i + 1])
                        for i in range(0, len(flat_poly), 2)
                    ]
                    clipped = clip_polygon_to_tile(
                        points, tx, ty, tw, th,
                        densify_factor=self.config.densify_factor,
                        smoothing_tolerance=self.config.smoothing_tolerance,
                    )
                    if not clipped:
                        continue

                    for poly_coords in clipped:
                        local = to_tile_coords(poly_coords, tx, ty)
                        ratio = polygon_area_ratio(local, tw, th)
                        if not (self.config.min_area_ratio <= ratio <= self.config.max_area_ratio):
                            continue

                        # Build new annotation
                        flat = []
                        for x, y in local:
                            flat.extend([
                                max(0, min(tw, x)),
                                max(0, min(th, y)),
                            ])

                        from ..core.annotations import AnnotationHandler
                        bbox = AnnotationHandler.recompute_bbox_from_polygon(local)
                        area = AnnotationHandler.polygon_area(local)

                        result.append({
                            "category_id": ann["category_id"],
                            "segmentation": [flat],
                            "bbox": bbox,
                            "area": area,
                            "iscrowd": ann.get("iscrowd", 0),
                        })

            elif seg and isinstance(seg, dict):
                # RLE — crop the mask to tile
                clipped_rle = self._clip_rle_to_tile(
                    seg, tx, ty, tw, th,
                )
                if clipped_rle is None:
                    continue

                # Check area ratio
                from ..formats.coco_utils import decode_rle
                mask = decode_rle(clipped_rle)
                mask_area = float(np.sum(mask > 0))
                tile_area = tw * th
                ratio = mask_area / tile_area if tile_area > 0 else 0
                if not (self.config.min_area_ratio <= ratio <= self.config.max_area_ratio):
                    continue

                result.append({
                    "category_id": ann["category_id"],
                    "segmentation": clipped_rle,
                    "bbox": [0, 0, tw, th],  # approximate
                    "area": mask_area,
                    "iscrowd": ann.get("iscrowd", 0),
                })

            else:
                # Bbox only
                bbox = ann.get("bbox")
                if bbox is None:
                    continue

                clipped = clip_bbox_to_tile(bbox, tx, ty, tw, th)
                if clipped is None:
                    continue

                ratio = bbox_area_ratio(clipped, tw, th)
                if not (self.config.min_area_ratio <= ratio <= self.config.max_area_ratio):
                    continue

                result.append({
                    "category_id": ann["category_id"],
                    "bbox": clipped,
                    "area": clipped[2] * clipped[3],
                    "iscrowd": ann.get("iscrowd", 0),
                })

        return result

    def _clip_rle_to_tile(
        self,
        rle_data: Dict,
        tx: int,
        ty: int,
        tw: int,
        th: int,
    ) -> Optional[Dict]:
        """Crop an RLE mask to a tile region, re-encode."""
        try:
            from pycocotools import mask as coco_mask_util
            from ..formats.coco_utils import decode_rle

            full_mask = decode_rle(rle_data)
            h_mask, w_mask = full_mask.shape

            # Extract tile region
            y1, y2 = ty, min(ty + th, h_mask)
            x1, x2 = tx, min(tx + tw, w_mask)
            if y1 >= y2 or x1 >= x2:
                return None

            tile_mask = full_mask[y1:y2, x1:x2]

            # Pad if needed
            if tile_mask.shape != (th, tw):
                padded = np.zeros((th, tw), dtype=np.uint8)
                padded[:tile_mask.shape[0], :tile_mask.shape[1]] = tile_mask
                tile_mask = padded

            if np.sum(tile_mask) == 0:
                return None

            # Re-encode
            tile_mask = np.asfortranarray(tile_mask.astype(np.uint8))
            rle = coco_mask_util.encode(tile_mask)
            rle["counts"] = rle["counts"].decode("utf-8")
            return rle

        except Exception as exc:
            logger.debug("RLE clipping failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # COCO tile image writing
    # ------------------------------------------------------------------

    def _write_coco_tile_image(
        self,
        src_path: Path,
        tx: int,
        ty: int,
        tw: int,
        th: int,
        out_path: Path,
    ) -> None:
        """Read a tile region and write it out."""
        import warnings
        import rasterio
        from rasterio.windows import Window
        warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

        with rasterio.open(src_path) as src:
            read_w = min(tw, src.width - tx)
            read_h = min(th, src.height - ty)
            window = Window(tx, ty, read_w, read_h)
            data = src.read(window=window)
            bands = src.count

        if self.config.edge_behavior == "padding" and (read_w < tw or read_h < th):
            padded = np.full((bands, th, tw), fill_value=0, dtype=data.dtype)
            for b in range(min(bands, 3)):
                padded[b, :, :] = self.config.padding_value[b]
            if bands > 3:
                padded[3:, :, :] = 255
            padded[:, :read_h, :read_w] = data
            data = padded

        ext = out_path.suffix.lower()
        driver_map = {
            ".jpg": "JPEG", ".jpeg": "JPEG",
            ".png": "PNG",
            ".tif": "GTiff", ".tiff": "GTiff",
            ".bmp": "BMP",
        }
        driver = driver_map.get(ext, "JPEG")

        kwargs: Dict[str, Any] = {
            "driver": driver,
            "width": data.shape[2],
            "height": data.shape[1],
            "count": bands,
            "dtype": data.dtype,
        }
        if driver == "JPEG":
            kwargs["quality"] = self.config.compression

        import rasterio
        with rasterio.open(out_path, "w", **kwargs) as dst:
            dst.write(data)

    # ------------------------------------------------------------------
    # Negative sampling
    # ------------------------------------------------------------------

    def _apply_negative_sampling(
        self,
        lbl_dir: Path,
        img_dir: Path,
        total_positive: int,
        total_negative: int,
    ) -> int:
        """Remove excess negative tiles from a YOLO split.

        Returns the number of negative tiles kept.
        """
        ratio = self.config.negative_sample_ratio
        if ratio < 0 or total_negative == 0:
            return total_negative

        if total_positive == 0:
            target = total_negative
        else:
            target = int(total_positive * ratio / (1 - ratio)) if ratio < 1 else total_negative

        target = min(target, total_negative)

        if target >= total_negative:
            return total_negative

        # Find negative tiles (empty label files)
        negatives = [
            p for p in sorted(lbl_dir.glob("*.txt"))
            if p.stat().st_size == 0
        ]

        if len(negatives) <= target:
            return len(negatives)

        # Sample which to keep
        rng = random.Random(self.config.random_seed)
        to_remove = rng.sample(negatives, len(negatives) - target)

        for lbl_path in to_remove:
            lbl_path.unlink()
            # Remove corresponding image
            for ext_try in (".jpg", ".jpeg", ".png", ".tif", ".bmp"):
                img_path = img_dir / (lbl_path.stem + ext_try)
                if img_path.exists():
                    img_path.unlink()
                    break

        kept = total_negative - len(to_remove)
        logger.info(
            "Negative sampling: %d/%d negatives kept (target ratio=%.2f)",
            kept, total_negative, ratio,
        )
        return kept

    def _apply_negative_sampling_coco(
        self,
        images: List[Dict],
        annotations: List[Dict],
        total_positive: int,
        total_negative: int,
    ) -> Tuple[List[Dict], List[Dict]]:
        """Filter negative images from COCO output lists."""
        ratio = self.config.negative_sample_ratio
        if ratio < 0 or total_negative == 0:
            return images, annotations

        if total_positive == 0:
            target = total_negative
        else:
            target = int(total_positive * ratio / (1 - ratio)) if ratio < 1 else total_negative

        target = min(target, total_negative)

        if target >= total_negative:
            return images, annotations

        # Find which image IDs have annotations
        annotated_ids = {a["image_id"] for a in annotations}
        positive_imgs = [img for img in images if img["id"] in annotated_ids]
        negative_imgs = [img for img in images if img["id"] not in annotated_ids]

        rng = random.Random(self.config.random_seed)
        kept_negatives = rng.sample(negative_imgs, min(target, len(negative_imgs)))

        final_images = positive_imgs + kept_negatives
        final_img_ids = {img["id"] for img in final_images}
        final_annotations = [a for a in annotations if a["image_id"] in final_img_ids]

        return final_images, final_annotations

    # ------------------------------------------------------------------
    # Image sampling (fraction)
    # ------------------------------------------------------------------

    def _sample_images(self, paths: List[Path]) -> List[Path]:
        """Optionally sub-sample the image list by *config.fraction*."""
        if self.config.fraction is None:
            return paths
        rng = random.Random(self.config.random_seed)
        n = max(1, int(len(paths) * self.config.fraction))
        return rng.sample(paths, min(n, len(paths)))

    def _sample_images_list(self, images: List[Dict]) -> List[Dict]:
        """Optionally sub-sample COCO image entries."""
        if self.config.fraction is None:
            return images
        rng = random.Random(self.config.random_seed)
        n = max(1, int(len(images) * self.config.fraction))
        return rng.sample(images, min(n, len(images)))
