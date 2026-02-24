"""FiftyOne utility functions: label field resolution, area filtering, negative sampling."""

import logging
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import fiftyone as fo
import fiftyone.core.labels as fol
from fiftyone import ViewField as F

logger = logging.getLogger(__name__)

# Default label fields to probe, in priority order.
DEFAULT_LABEL_FIELDS = ["segmentations", "ground_truth"]


# ------------------------------------------------------------------
# Label field resolution
# ------------------------------------------------------------------


def resolve_label_field(
    dataset: fo.Dataset,
    provided: Optional[str] = None,
) -> str:
    """Resolve the label field for a FiftyOne dataset.

    Resolution order:

    1. *provided* (if given and exists in the schema).
    2. First match from :data:`DEFAULT_LABEL_FIELDS`.
    3. First ``Detections`` or ``Polylines`` field in the schema.

    Raises:
        ValueError: If no suitable label field is found.
    """
    if provided:
        if dataset.has_sample_field(provided):
            logger.debug("Using provided label field: '%s'", provided)
            return provided
        raise ValueError(
            f"Provided label field '{provided}' does not exist in dataset"
        )

    for candidate in DEFAULT_LABEL_FIELDS:
        if dataset.has_sample_field(candidate):
            logger.debug("Using detected label field: '%s'", candidate)
            return candidate

    # Last resort: first label-container field in the schema.
    schema = dataset.get_field_schema()
    for field_name, field in schema.items():
        ftype = getattr(field, "document_type", None)
        if ftype in {
            getattr(fol, "Detections", object),
            getattr(fol, "Polylines", object),
        }:
            logger.debug("Using first label-like field: '%s'", field_name)
            return field_name

    raise ValueError(
        "Could not resolve a label field. Provide label_field explicitly "
        f"or ensure one of {DEFAULT_LABEL_FIELDS} exists."
    )


def _detect_container_type(
    dataset: fo.Dataset, label_field: str
) -> str:
    """Return ``'detections'`` or ``'polylines'`` for *label_field*."""
    field = dataset.get_field_schema().get(label_field)
    if field is None:
        raise ValueError(
            f"Label field '{label_field}' not found in dataset schema"
        )
    doc_type = getattr(field, "document_type", None)
    if doc_type is getattr(fol, "Detections", None):
        return "detections"
    if doc_type is getattr(fol, "Polylines", None):
        return "polylines"
    raise ValueError(
        f"Unsupported label container type for field '{label_field}'"
    )


# ------------------------------------------------------------------
# Area ratio filtering
# ------------------------------------------------------------------


def filter_by_area_ratio(
    dataset: fo.Dataset,
    min_area_ratio: float = 0.0,
    max_area_ratio: float = 1.0,
    label_field: Optional[str] = None,
) -> fo.DatasetView:
    """Filter individual annotations by relative area.

    Requires each annotation to have an ``area_rel`` attribute
    (typically computed by ``dataset.compute_metadata()`` or a custom step).
    Annotations without the attribute are kept.

    Args:
        dataset: Source dataset or view.
        min_area_ratio: Keep annotations with ``area_rel >= min_area_ratio``.
        max_area_ratio: Keep annotations with ``area_rel <= max_area_ratio``.
        label_field: Label field name; auto-resolved if *None*.

    Returns:
        Filtered :class:`fo.DatasetView`.
    """
    resolved = resolve_label_field(dataset, label_field)
    container = _detect_container_type(dataset, resolved)
    area_field = f"{resolved}.{container}.area_rel"

    if not dataset.has_sample_field(area_field):
        logger.warning(
            "area_rel field '%s' not found — skipping area filtering",
            area_field,
        )
        return dataset.view()

    before_count = dataset.aggregate(fo.Count(f"{resolved}.{container}"))
    logger.info("Annotations before area filtering: %d", before_count)

    subfield = f"{resolved}.{container}"
    expr = (
        (F("area_rel") >= min_area_ratio) & (F("area_rel") <= max_area_ratio)
    ) | ~F("area_rel").exists()
    filtered = dataset.filter_labels(subfield, expr, only_matches=False)

    after_count = filtered.aggregate(fo.Count(f"{resolved}.{container}"))
    logger.info(
        "Annotations after area filtering: %d (removed %d)",
        after_count,
        before_count - after_count,
    )
    return filtered


# ------------------------------------------------------------------
# Thumbnails
# ------------------------------------------------------------------


def create_thumbnails(
    dataset: fo.Dataset,
    output_dir: Optional[Union[str, "Path"]] = None,
    size: int = 256,
) -> None:
    """Create thumbnail images for grid display in the FiftyOne app.

    Writes thumbnails to a ``.thumbnails/`` hidden folder and configures
    the dataset's app config to use them in the grid view.

    Args:
        dataset: The FiftyOne dataset.
        output_dir: Parent directory for the ``.thumbnails/`` folder.
            If *None*, the common ancestor of all sample filepaths is used.
        size: Thumbnail height in pixels (aspect ratio is preserved).
    """
    import fiftyone.utils.image as foui
    from pathlib import Path
    import os

    if output_dir is None:
        # Derive from the common ancestor of all sample filepaths.
        filepaths = dataset.values("filepath")
        if not filepaths:
            logger.warning("Dataset is empty — skipping thumbnail creation")
            return
        common = os.path.commonpath(filepaths)
        # commonpath may return a file path if all samples share a dir;
        # ensure we have a directory.
        output_dir = Path(common) if os.path.isdir(common) else Path(common).parent

    output_dir = Path(output_dir)
    thumbnails_path = output_dir / ".thumbnails"

    if thumbnails_path.exists() and dataset.has_sample_field("thumbnail_path"):
        logger.info("Thumbnails already exist at %s, skipping", thumbnails_path)
        return

    # Try to reuse existing thumbnail files from a previous import.
    if thumbnails_path.exists() and not dataset.has_sample_field("thumbnail_path"):
        existing = {p.stem: p for p in thumbnails_path.iterdir() if p.is_file()}
        if existing:
            filepaths = dataset.values("filepath")
            matches = {
                fp: str(existing[Path(fp).stem])
                for fp in filepaths
                if Path(fp).stem in existing
            }
            if len(matches) == len(filepaths):
                logger.info(
                    "Reusing %d existing thumbnails from %s",
                    len(matches), thumbnails_path,
                )
                dataset.set_values(
                    "thumbnail_path",
                    [matches[fp] for fp in filepaths],
                )
                dataset.app_config.media_fields = ["filepath", "thumbnail_path"]
                dataset.app_config.grid_media_field = "thumbnail_path"
                dataset.save()
                return
            # Partial match — regenerate everything below.
            logger.info(
                "Found %d/%d cached thumbnails — regenerating all",
                len(matches), len(filepaths),
            )

    logger.info("Creating thumbnails (size=%d) in %s ...", size, thumbnails_path)
    foui.transform_images(
        dataset,
        size=(-1, size),
        output_field="thumbnail_path",
        output_dir=str(thumbnails_path),
    )

    dataset.app_config.media_fields = ["filepath", "thumbnail_path"]
    dataset.app_config.grid_media_field = "thumbnail_path"
    dataset.save()
    logger.info("Thumbnails saved to %s", thumbnails_path)


# ------------------------------------------------------------------
# Negative sampling
# ------------------------------------------------------------------


class NegativeSampler:
    """Sample negative (unannotated) images for inclusion in exported datasets.

    Supports three modes:

    * **random** — simple random sampling (default when no tags are given).
    * **stratified** — proportional representation per tag bucket.
    * **balanced** — equal representation per tag bucket.
    """

    def classify_samples(
        self,
        dataset: fo.Dataset,
        label_field: str,
    ) -> Tuple[fo.DatasetView, fo.DatasetView]:
        """Split *dataset* into positive (annotated) and negative (empty) views."""
        container = _detect_container_type(dataset, label_field)

        has_ann = F(f"{label_field}.{container}").exists()
        non_empty = F(f"{label_field}.{container}").length() > 0

        positive = dataset.match(has_ann & non_empty)
        negative = dataset.match(~has_ann | ~non_empty)

        logger.info(
            "Classified: %d positive, %d negative",
            len(positive),
            len(negative),
        )
        return positive, negative

    def sample_negatives(
        self,
        negative_view: fo.DatasetView,
        negative_ratio: float,
        positive_count: int,
        tags: Optional[List[str]] = None,
        strategy: Optional[str] = None,
        default_ratio: float = 0.3,
    ) -> fo.DatasetView:
        """Select a subset of negative samples.

        Args:
            negative_view: View containing all negative samples.
            negative_ratio: Target fraction of the *final* dataset that
                should be negative (0–1, exclusive).
            positive_count: Number of positive samples.
            tags: Tags to use for bucketed sampling.
            strategy: ``"stratified"`` or ``"balanced"``; *None* → random.
            default_ratio: In stratified mode, fraction of slots reserved
                for untagged negatives.

        Returns:
            A view or list of sampled negative samples.
        """
        total_available = len(negative_view)

        if negative_ratio <= 0.0:
            logger.info("Negative ratio <= 0 — no negatives included")
            return negative_view.limit(0)

        if negative_ratio >= 1.0:
            raise ValueError("negative_ratio must be < 1.0")

        # negative_count / (positive_count + negative_count) = negative_ratio
        required = int(
            (negative_ratio * positive_count) / (1 - negative_ratio)
        )
        if total_available < required:
            logger.warning(
                "Need %d negatives but only %d available — using all",
                required,
                total_available,
            )
            required = total_available

        logger.info(
            "Target negative ratio %.1f%% -> %d negatives (from %d available)",
            negative_ratio * 100,
            required,
            total_available,
        )

        # Simple random sampling when no strategy or tags
        if not strategy or not tags:
            return negative_view.shuffle().limit(required)

        # Tag-based bucketed sampling
        buckets = self._create_tag_buckets(negative_view, tags)

        if strategy == "stratified":
            return self._stratified_sample(buckets, required, default_ratio)
        if strategy == "balanced":
            return self._balanced_sample(buckets, required)
        raise ValueError(f"Unknown sampling strategy: {strategy!r}")

    def apply_tag_exclusion(
        self,
        dataset: fo.Dataset,
        exclude_tags: List[str],
    ) -> fo.DatasetView:
        """Exclude samples that carry any of *exclude_tags*."""
        if not exclude_tags:
            return dataset.view()

        before = len(dataset)
        filtered = dataset.match_tags(exclude_tags, bool=False)
        logger.info(
            "Tag exclusion: %d -> %d samples (%d removed)",
            before,
            len(filtered),
            before - len(filtered),
        )
        return filtered

    # ---- Private helpers ------------------------------------------------

    def _create_tag_buckets(
        self, negative_view: fo.DatasetView, tags: List[str]
    ) -> Dict[str, List]:
        """Assign each sample to exactly one bucket (rarest-tag priority)."""
        tag_counts = negative_view.aggregate(fo.CountValues("tags"))
        tag_counts = {t: c for t, c in tag_counts.items() if t in tags}

        buckets: Dict[str, List] = {t: [] for t in tags}
        buckets["default"] = []

        for sample in negative_view:
            sample_tags = [t for t in (sample.tags or []) if t in tags]
            if sample_tags:
                rarest = min(sample_tags, key=lambda t: tag_counts.get(t, 0))
                buckets[rarest].append(sample)
            else:
                buckets["default"].append(sample)

        for tag, items in buckets.items():
            logger.debug("Bucket '%s': %d samples", tag, len(items))
        return buckets

    def _stratified_sample(
        self,
        buckets: Dict[str, List],
        total: int,
        default_ratio: float,
    ) -> List:
        """Proportional representation per tag bucket."""
        default_slots = min(
            int(total * default_ratio), len(buckets.get("default", []))
        )
        tagged_slots = total - default_slots
        tagged_buckets = {
            t: s for t, s in buckets.items() if t != "default" and s
        }
        total_tagged = sum(len(s) for s in tagged_buckets.values())

        selected: List = []
        if total_tagged == 0:
            return self._random_pick(buckets.get("default", []), total)

        for tag, samples in tagged_buckets.items():
            n = min(
                int(len(samples) / total_tagged * tagged_slots), len(samples)
            )
            selected.extend(self._random_pick(samples, n))

        # Fill remaining with default-bucket samples
        selected.extend(
            self._random_pick(
                buckets.get("default", []),
                min(total - len(selected), default_slots),
            )
        )

        # Back-fill from any bucket if still short
        if len(selected) < total:
            selected_ids = {id(s) for s in selected}
            remaining = []
            for samples in buckets.values():
                remaining.extend(
                    s for s in samples if id(s) not in selected_ids
                )
            selected.extend(
                self._random_pick(remaining, total - len(selected))
            )

        return selected[:total]

    def _balanced_sample(
        self, buckets: Dict[str, List], total: int
    ) -> List:
        """Equal representation per tag bucket."""
        tagged_buckets = {
            t: s for t, s in buckets.items() if t != "default" and s
        }
        if not tagged_buckets:
            return self._random_pick(buckets.get("default", []), total)

        per_bucket = total // len(tagged_buckets)
        extra = total % len(tagged_buckets)
        selected: List = []

        for i, (tag, samples) in enumerate(
            sorted(tagged_buckets.items(), key=lambda x: len(x[1]))
        ):
            n = per_bucket + (1 if i < extra else 0)
            selected.extend(self._random_pick(samples, min(n, len(samples))))

        if len(selected) < total:
            selected.extend(
                self._random_pick(
                    buckets.get("default", []), total - len(selected)
                )
            )

        return selected[:total]

    @staticmethod
    def _random_pick(items: List, n: int) -> List:
        if n <= 0:
            return []
        if n >= len(items):
            return list(items)
        return random.sample(items, n)
