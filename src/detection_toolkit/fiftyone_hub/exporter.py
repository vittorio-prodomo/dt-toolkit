"""Export FiftyOne datasets to disk in YOLO or COCO format.

Supports per-split export, area-ratio filtering, negative sampling,
and tag-based exclusion.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

import fiftyone as fo
import yaml
from fiftyone import ViewField as F
from fiftyone.utils.coco import COCODetectionDatasetExporter
from fiftyone.utils.yolo import YOLOv5DatasetExporter

from ..core.models import DatasetFormat
from .utils import NegativeSampler, filter_by_area_ratio, resolve_label_field

logger = logging.getLogger(__name__)


def _write_weights_json(filepaths, scores, output_dir):
    """Write per-split weights.json with normalized scores.

    Args:
        filepaths: List of absolute file paths for this split's samples.
        scores: List of raw score values (same length as filepaths).
        output_dir: Directory to write weights.json into.
    """
    scores_arr = np.array(scores, dtype=np.float64)
    s_min, s_max = scores_arr.min(), scores_arr.max()
    if s_max - s_min > 1e-12:
        weights = (scores_arr - s_min) / (s_max - s_min)
    else:
        weights = np.ones_like(scores_arr)

    mapping = {}
    for fp, w in zip(filepaths, weights):
        filename = Path(fp).name
        mapping[filename] = round(float(w), 6)

    out_path = Path(output_dir) / "weights.json"
    with open(out_path, "w") as f:
        json.dump(mapping, f, indent=2)
    logger.info("Wrote %d weights to %s", len(mapping), out_path)


class FiftyOneDatasetExporter:
    """Export a FiftyOne dataset to YOLO or COCO format on disk.

    Usage::

        exporter = FiftyOneDatasetExporter()
        exporter.export(
            dataset,
            Path("output/yolo_dataset"),
            DatasetFormat.YOLO,
        )
    """

    def export(
        self,
        dataset: fo.Dataset,
        output_dir: Union[str, Path],
        format: DatasetFormat,
        export_media: Union[str, bool] = "symlink",
        splits: Optional[List[str]] = None,
        label_field: Optional[str] = None,
        min_area_ratio: float = 0.0,
        max_area_ratio: float = 1.0,
        negative_ratio: Optional[float] = None,
        negative_strategy: Optional[str] = None,
        negative_tags: Optional[List[str]] = None,
        negative_default_ratio: float = 0.3,
        exclude_tags: Optional[List[str]] = None,
        weights_field: Optional[str] = None,
    ) -> None:
        """Export *dataset* to disk.

        Args:
            dataset: FiftyOne dataset or view.
            output_dir: Target directory.
            format: Output format (:attr:`DatasetFormat.YOLO` or
                :attr:`DatasetFormat.COCO`).
            export_media: Media handling — ``"symlink"``, ``"copy"``,
                ``"move"``, or ``False``.
            splits: Splits to export; ``None`` means all available.
            label_field: Label field; auto-resolved if *None*.
            min_area_ratio: Minimum annotation area ratio (0–1).
            max_area_ratio: Maximum annotation area ratio (0–1).
            negative_ratio: Target fraction of negatives in the final
                dataset (0–1).  ``None`` means no negative sampling.
            negative_strategy: ``"stratified"`` or ``"balanced"``.
            negative_tags: Tags for bucketed negative sampling.
            negative_default_ratio: Default-bucket share in stratified mode.
            exclude_tags: Sample-level tags to exclude before export.
            weights_field: Optional sample field containing importance scores.
                When provided, a ``weights.json`` file is written per split
                mapping filenames to ``[0, 1]`` normalized weights.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        view = dataset.view() if isinstance(dataset, fo.Dataset) else dataset

        # --- Tag exclusion ---
        if exclude_tags:
            sampler = NegativeSampler()
            view = sampler.apply_tag_exclusion(view, exclude_tags)

        # --- Area ratio filtering ---
        if min_area_ratio > 0.0 or max_area_ratio < 1.0:
            logger.info(
                "Applying area ratio filtering: min=%.3f, max=%.3f",
                min_area_ratio,
                max_area_ratio,
            )
            view = filter_by_area_ratio(
                view,
                min_area_ratio=min_area_ratio,
                max_area_ratio=max_area_ratio,
                label_field=label_field,
            )

        # --- Negative sampling ---
        if negative_ratio is not None:
            logger.info(
                "Applying negative sampling with ratio: %.2f", negative_ratio
            )
            view = self._apply_negative_sampling(
                view,
                label_field=label_field,
                negative_ratio=negative_ratio,
                negative_strategy=negative_strategy,
                negative_tags=negative_tags,
                negative_default_ratio=negative_default_ratio,
            )

        # --- Split filtering ---
        view = self._filter_splits(view, splits)

        # --- Export ---
        if format == DatasetFormat.YOLO:
            self._export_yolo(view, output_dir, export_media, weights_field)
        elif format == DatasetFormat.COCO:
            self._export_coco(view, output_dir, export_media, weights_field)
        else:
            raise ValueError(f"Unsupported export format: {format}")

        logger.info("Export complete -> %s", output_dir)

    # ------------------------------------------------------------------
    # Negative sampling
    # ------------------------------------------------------------------

    def _apply_negative_sampling(
        self,
        view: fo.DatasetView,
        label_field: Optional[str],
        negative_ratio: float,
        negative_strategy: Optional[str],
        negative_tags: Optional[List[str]],
        negative_default_ratio: float,
    ) -> fo.DatasetView:
        """Run per-split negative sampling and recombine."""
        resolved = resolve_label_field(view, label_field)
        sampler = NegativeSampler()

        has_splits = view.has_sample_field("split")
        if not has_splits:
            return self._negative_sample_group(
                view, resolved, sampler,
                negative_ratio, negative_strategy,
                negative_tags, negative_default_ratio,
            )

        # Per-split sampling
        split_names = view.distinct("split")
        merged = fo.Dataset(name="_export_neg_temp", overwrite=True)

        for split_name in split_names:
            split_view = view.match(F("split") == split_name)
            if len(split_view) == 0:
                continue

            merged_split = self._negative_sample_group(
                split_view, resolved, sampler,
                negative_ratio, negative_strategy,
                negative_tags, negative_default_ratio,
                merge_into=merged,
            )

        return merged.view()

    def _negative_sample_group(
        self,
        view,
        resolved_label_field: str,
        sampler: NegativeSampler,
        negative_ratio: float,
        negative_strategy: Optional[str],
        negative_tags: Optional[List[str]],
        negative_default_ratio: float,
        merge_into: Optional[fo.Dataset] = None,
    ):
        """Classify, sample, and optionally merge positive+negative."""
        positive, negative = sampler.classify_samples(
            view, resolved_label_field
        )
        sampled = sampler.sample_negatives(
            negative,
            negative_ratio,
            len(positive),
            tags=negative_tags,
            strategy=negative_strategy,
            default_ratio=negative_default_ratio,
        )

        if merge_into is not None:
            merge_into.merge_samples(positive)
            if isinstance(sampled, (list, tuple)):
                merge_into.add_samples(sampled)
            else:
                merge_into.merge_samples(sampled)
            return merge_into

        result = fo.Dataset(name="_export_neg_temp", overwrite=True)
        result.merge_samples(positive)
        if isinstance(sampled, (list, tuple)):
            result.add_samples(sampled)
        else:
            result.merge_samples(sampled)
        return result.view()

    # ------------------------------------------------------------------
    # Split helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _filter_splits(
        view: fo.DatasetView, splits: Optional[List[str]]
    ) -> fo.DatasetView:
        """Restrict *view* to the requested splits."""
        if splits is None or not view.has_sample_field("split"):
            return view

        available = view.distinct("split")
        avail_lower = {s.lower(): s for s in available}
        matched = []

        for s in splits:
            canonical = avail_lower.get(s.lower())
            if canonical:
                matched.append(canonical)
            else:
                logger.warning(
                    "Requested split '%s' not found in dataset", s
                )

        if not matched:
            raise ValueError(
                f"None of the requested splits {splits} found. "
                f"Available: {available}"
            )
        return view.match(F("split").is_in(matched))

    @staticmethod
    def _resolve_export_splits(view) -> List[str]:
        """Get the final list of splits to iterate over for export."""
        if view.has_sample_field("split"):
            return view.distinct("split")
        return ["train"]

    # ------------------------------------------------------------------
    # Format-specific export
    # ------------------------------------------------------------------

    def _export_yolo(
        self,
        view,
        output_dir: Path,
        export_media: Union[str, bool],
        weights_field=None,
    ) -> None:
        export_splits = self._resolve_export_splits(view)
        classes = self._resolve_classes(view)
        yaml_path = output_dir / "data.yaml"

        for split in export_splits:
            split_view = self._split_view(view, split)
            if len(split_view) == 0:
                logger.warning("No samples for YOLO split '%s'", split)
                continue

            split_export_media = export_media
            if self._should_skip_media(
                split_view, output_dir / "images" / split,
            ):
                split_export_media = False

            logger.info(
                "Exporting YOLO split: %s (%d samples)",
                split,
                len(split_view),
            )
            exporter = YOLOv5DatasetExporter(
                split=split,
                data_path=str(output_dir / "images" / split),
                labels_path=str(output_dir / "labels" / split),
                yaml_path=str(yaml_path),
                export_media=split_export_media,
                classes=classes,
            )
            split_view.export(dataset_exporter=exporter)
            if weights_field:
                _write_weights_json(
                    split_view.values("filepath"),
                    split_view.values(weights_field),
                    output_dir / "images" / split,
                )

        # Post-process the YAML
        self._fix_yolo_yaml(yaml_path, classes)

    def _export_coco(
        self,
        view,
        output_dir: Path,
        export_media: Union[str, bool],
        weights_field=None,
    ) -> None:
        export_splits = self._resolve_export_splits(view)
        classes = self._resolve_classes(view)

        for split in export_splits:
            split_view = self._split_view(view, split)
            if len(split_view) == 0:
                logger.warning("No samples for COCO split '%s'", split)
                continue

            split_export_media = export_media
            if self._should_skip_media(
                split_view, output_dir / "images" / split,
            ):
                split_export_media = False

            logger.info(
                "Exporting COCO split: %s (%d samples)",
                split,
                len(split_view),
            )
            exporter = COCODetectionDatasetExporter(
                data_path=str(output_dir / "images" / split),
                labels_path=str(
                    output_dir / "annotations" / f"instances_{split}.json"
                ),
                export_media=split_export_media,
                classes=classes,
                extra_attrs=True,
            )
            split_view.export(dataset_exporter=exporter)
            if weights_field:
                _write_weights_json(
                    split_view.values("filepath"),
                    split_view.values(weights_field),
                    output_dir / "images" / split,
                )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _split_view(view, split: str):
        """Get a view for a single split, or the full view if no split field."""
        if view.has_sample_field("split"):
            return view.match(F("split") == split)
        return view

    @staticmethod
    def _resolve_classes(view) -> List[str]:
        """Extract the class list from the dataset/view."""
        # DatasetView and Dataset both expose the .classes property
        classes = view.classes
        if classes:
            return list(classes.values())[0]
        return []

    @staticmethod
    def _should_skip_media(split_view, target_image_dir) -> bool:
        """Check if all sample images already live in the target directory.

        When source images resolve to paths inside *target_image_dir*,
        media export can be skipped to avoid overwriting or destroying
        the originals.

        Args:
            split_view: FiftyOne DatasetView for a single split.
            target_image_dir: Target directory for exported images.

        Returns:
            ``True`` if all filepaths resolve to inside *target_image_dir*.
        """
        filepaths = split_view.values("filepath")
        if not filepaths:
            return False

        resolved_target = Path(target_image_dir).resolve()

        overlap_count = 0
        for fp in filepaths:
            resolved = Path(fp).resolve()
            if resolved.is_relative_to(resolved_target):
                overlap_count += 1

        if overlap_count == len(filepaths):
            logger.info(
                "All %d images already in %s, skipping media export",
                len(filepaths),
                resolved_target,
            )
            return True

        if overlap_count > 0:
            logger.warning(
                "%d of %d images already in %s but others are not. "
                "Proceeding with media export for this split.",
                overlap_count,
                len(filepaths),
                resolved_target,
            )

        return False

    @staticmethod
    def _fix_yolo_yaml(yaml_path: Path, classes: List[str]) -> None:
        """Post-process the YOLO ``data.yaml`` written by FiftyOne.

        Removes the ``path`` key and ensures ``nc`` is present.
        """
        if not yaml_path.exists():
            return
        with open(yaml_path, "r") as fh:
            data = yaml.safe_load(fh)
        data.pop("path", None)
        if "nc" not in data:
            data["nc"] = len(classes)
        with open(yaml_path, "w") as fh:
            yaml.dump(data, fh, default_flow_style=False, sort_keys=False)
