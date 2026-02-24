"""Orchestrate label enrichment on FiftyOne datasets."""

import logging
import time
from typing import List, Optional

import fiftyone as fo
import fiftyone.core.labels as fol

from .area import bbox_area, mask_area, polyline_area

logger = logging.getLogger(__name__)


def _get_label_fields(dataset: fo.Dataset) -> List[str]:
    """Return names of all Detections / Polylines fields in *dataset*."""
    schema = dataset.get_field_schema()
    label_types = {
        getattr(fol, "Detections", object),
        getattr(fol, "Polylines", object),
    }
    return [
        name for name, field in schema.items()
        if getattr(field, "document_type", None) in label_types
    ]


def _field_is_enriched(dataset: fo.Dataset, label_field: str) -> bool:
    """Check whether *label_field* already has ``area_px`` in its schema."""
    flat = dataset.get_field_schema(flat=True)
    return any(
        key.startswith(f"{label_field}.") and key.endswith(".area_px")
        for key in flat
    )


def _field_container_key(dataset: fo.Dataset, label_field: str) -> str:
    """Return ``'detections'`` or ``'polylines'`` for *label_field*."""
    doc_type = getattr(
        dataset.get_field_schema()[label_field], "document_type", None
    )
    if doc_type is getattr(fol, "Detections", None):
        return "detections"
    if doc_type is getattr(fol, "Polylines", None):
        return "polylines"
    raise ValueError(f"Unsupported label type for field '{label_field}'")


class DatasetEnricher:
    """Compute and store derived label attributes on a FiftyOne dataset.

    Usage::

        enricher = DatasetEnricher()
        enricher.enrich(dataset)
    """

    def __init__(self, verbose: bool = True) -> None:
        self.verbose = verbose

    def enrich(
        self,
        dataset: fo.Dataset,
        label_fields: Optional[List[str]] = None,
        batch_size: int = 1000,
        interactive: bool = True,
    ) -> None:
        """Compute ``area_px`` and ``area_rel`` for every label.

        Args:
            dataset: The FiftyOne dataset to enrich.
            label_fields: Label fields to process.  If *None*, auto-discover.
            batch_size: Batch size for mask processing.
            interactive: If *True*, prompt the user when multiple fields
                are found or when a field is already enriched.
        """
        # Ensure metadata is populated
        if dataset.exists("metadata").count() < len(dataset):
            logger.info("Computing missing metadata ...")
            dataset.compute_metadata(overwrite=False)

        # Resolve fields
        fields = label_fields or self._resolve_fields(dataset, interactive)
        if not fields:
            logger.warning("No label fields to enrich.")
            return

        for field in fields:
            # Idempotency check
            if _field_is_enriched(dataset, field):
                if interactive:
                    answer = input(
                        f"  Field '{field}' is already enriched. "
                        "Overwrite? [y/N] "
                    ).strip().lower()
                    if answer not in ("y", "yes"):
                        logger.info("Skipping field '%s'", field)
                        continue
                else:
                    logger.info(
                        "Field '%s' already enriched — overwriting", field
                    )

            self._enrich_field(dataset, field, batch_size)

    # ------------------------------------------------------------------
    # Field resolution
    # ------------------------------------------------------------------

    def _resolve_fields(
        self, dataset: fo.Dataset, interactive: bool
    ) -> List[str]:
        """Auto-discover label fields, prompting when needed."""
        all_fields = _get_label_fields(dataset)

        if not all_fields:
            return []

        if len(all_fields) == 1 or not interactive:
            if self.verbose:
                logger.info("Label fields to enrich: %s", all_fields)
            return all_fields

        # Multiple fields, interactive mode — prompt user
        print(f"\nFound {len(all_fields)} label fields:")
        for i, f in enumerate(all_fields, 1):
            enriched = _field_is_enriched(dataset, f)
            tag = " (already enriched)" if enriched else ""
            print(f"  {i}. {f}{tag}")

        selection = input(
            "\nEnter field numbers to enrich (comma-separated), "
            "or 'all': "
        ).strip().lower()

        if selection == "all":
            return all_fields

        try:
            indices = [int(s.strip()) - 1 for s in selection.split(",")]
            return [all_fields[i] for i in indices]
        except (ValueError, IndexError):
            logger.error("Invalid selection — aborting.")
            return []

    # ------------------------------------------------------------------
    # Per-field enrichment
    # ------------------------------------------------------------------

    def _enrich_field(
        self,
        dataset: fo.Dataset,
        label_field: str,
        batch_size: int,
    ) -> None:
        """Compute and write area attributes for a single label field."""
        container = _field_container_key(dataset, label_field)
        t0 = time.perf_counter()

        logger.info("Enriching field '%s' (%s) ...", label_field, container)

        # Bulk-read image dimensions
        widths = dataset.values("metadata.width")
        heights = dataset.values("metadata.height")

        if container == "detections":
            self._enrich_detections(
                dataset, label_field, widths, heights, batch_size
            )
        else:
            self._enrich_polylines(
                dataset, label_field, widths, heights
            )

        elapsed = time.perf_counter() - t0
        logger.info(
            "Enriched '%s' in %.1fs", label_field, elapsed
        )

    # ------------------------------------------------------------------
    # Detections
    # ------------------------------------------------------------------

    def _enrich_detections(
        self,
        dataset: fo.Dataset,
        label_field: str,
        widths: list,
        heights: list,
        batch_size: int,
    ) -> None:
        """Compute areas for a Detections field.

        Reads bounding boxes in bulk.  Masks are read in batches to cap
        memory usage.
        """
        prefix = f"{label_field}.detections"

        # Bulk-read bounding boxes (lightweight)
        all_bboxes = dataset.values(f"{prefix}.bounding_box")

        # Check if any labels have masks by probing the schema
        flat_schema = dataset.get_field_schema(flat=True)
        has_masks = f"{prefix}.mask" in flat_schema

        if has_masks:
            all_areas = self._compute_detection_areas_batched(
                dataset, label_field, all_bboxes, widths, heights, batch_size
            )
        else:
            all_areas = self._compute_detection_areas_bbox_only(
                all_bboxes, widths, heights
            )

        all_px, all_rel = zip(*all_areas) if all_areas else ([], [])
        dataset.set_values(f"{prefix}.area_px", list(all_px))
        dataset.set_values(f"{prefix}.area_rel", list(all_rel))

    def _compute_detection_areas_bbox_only(
        self,
        all_bboxes: list,
        widths: list,
        heights: list,
    ) -> list:
        """Fast path: no masks, pure bbox arithmetic."""
        results = []
        total = 0
        for bboxes, w, h in zip(all_bboxes, widths, heights):
            sample_areas = []
            if bboxes and w is not None and h is not None:
                for bb in bboxes:
                    sample_areas.append(bbox_area(bb, w, h))
                    total += 1
            results.append(sample_areas)
        logger.info("Computed bbox areas for %d detections", total)
        return results

    def _compute_detection_areas_batched(
        self,
        dataset: fo.Dataset,
        label_field: str,
        all_bboxes: list,
        widths: list,
        heights: list,
        batch_size: int,
    ) -> list:
        """Batched path: read masks in chunks to cap memory."""
        prefix = f"{label_field}.detections"
        n = len(dataset)
        all_results = []
        total = 0

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_view = dataset.skip(start).limit(end - start)
            batch_masks = batch_view.values(f"{prefix}.mask")
            batch_bboxes = all_bboxes[start:end]
            batch_widths = widths[start:end]
            batch_heights = heights[start:end]

            for masks, bboxes, w, h in zip(
                batch_masks, batch_bboxes, batch_widths, batch_heights
            ):
                sample_areas = []
                if bboxes and w is not None and h is not None:
                    for i, bb in enumerate(bboxes):
                        m = masks[i] if masks and i < len(masks) else None
                        if m is not None:
                            sample_areas.append(mask_area(m, bb, w, h))
                        else:
                            sample_areas.append(bbox_area(bb, w, h))
                        total += 1
                all_results.append(sample_areas)

            if self.verbose and n > batch_size:
                logger.info(
                    "  Processed %d / %d samples ...", end, n
                )

        logger.info(
            "Computed areas for %d detections (batched, mask-aware)", total
        )
        return all_results

    # ------------------------------------------------------------------
    # Polylines
    # ------------------------------------------------------------------

    def _enrich_polylines(
        self,
        dataset: fo.Dataset,
        label_field: str,
        widths: list,
        heights: list,
    ) -> None:
        """Compute areas for a Polylines field (always bulk, no batching)."""
        prefix = f"{label_field}.polylines"
        all_points = dataset.values(f"{prefix}.points")

        all_px = []
        all_rel = []
        total = 0

        for pts_list, w, h in zip(all_points, widths, heights):
            sample_px = []
            sample_rel = []
            if pts_list and w is not None and h is not None:
                for pts in pts_list:
                    px, rel = polyline_area(pts, w, h)
                    sample_px.append(px)
                    sample_rel.append(rel)
                    total += 1
            all_px.append(sample_px)
            all_rel.append(sample_rel)

        dataset.set_values(f"{prefix}.area_px", all_px)
        dataset.set_values(f"{prefix}.area_rel", all_rel)
        logger.info("Computed polygon areas for %d polylines", total)
