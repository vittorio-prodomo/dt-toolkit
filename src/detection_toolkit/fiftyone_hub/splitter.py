"""Stratified dataset splitting for FiftyOne datasets.

Splits a FiftyOne dataset into train/val/test sets with class-balanced
stratification using a greedy bin-packing algorithm.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Lazy import: populated on first use or replaced by tests via patching.
try:
    from .utils import resolve_label_field
except ImportError:  # fiftyone not installed
    resolve_label_field = None  # type: ignore[assignment]


@dataclass
class SplitResult:
    """Result of a dataset split operation.

    Attributes:
        split_counts: Number of images per split.
        class_distribution: Per-class instance counts per split.
        group_counts: Number of groups per split (only when ``group_by`` is used).
    """

    split_counts: Dict[str, int]
    class_distribution: Dict[str, Dict[str, int]]
    group_counts: Optional[Dict[str, int]] = None


def _greedy_assign(
    vectors: np.ndarray,
    ratios: Dict[str, float],
    seed: int = 42,
) -> Dict[str, List[int]]:
    """Assign items to splits using greedy bin-packing.

    Args:
        vectors: ``(N, C)`` array of per-item class count vectors.
        ratios: Target proportions per split (must sum to 1.0).
        seed: Random seed for tie-breaking and final shuffle.

    Returns:
        Mapping of split name to list of item indices.
    """
    n_items = vectors.shape[0]
    if n_items == 0:
        return {name: [] for name in ratios}

    rng = np.random.RandomState(seed)

    class_totals = vectors.sum(axis=0)
    weights = 1.0 / (class_totals + 1.0)

    rarity_scores = vectors @ weights

    jitter = rng.uniform(0, 1e-9, size=n_items)
    order = np.argsort(-(rarity_scores + jitter))

    split_names = list(ratios.keys())
    split_totals = {name: np.zeros(vectors.shape[1]) for name in split_names}
    split_counts = {name: 0 for name in split_names}
    assignment: Dict[str, List[int]] = {name: [] for name in split_names}

    target_fractions = np.array([ratios[name] for name in split_names])
    total_instances = vectors.sum()

    for idx in order:
        vec = vectors[idx]

        if total_instances > 0:
            current_totals = np.array(
                [split_totals[name].sum() for name in split_names]
            )
            grand_total = current_totals.sum() + vec.sum()
            if grand_total > 0:
                current_fractions = current_totals / grand_total
            else:
                current_fractions = np.zeros(len(split_names))
            deficits = target_fractions - current_fractions
        else:
            total_items_so_far = sum(split_counts.values()) + 1
            current_fractions = np.array(
                [split_counts[name] / total_items_so_far for name in split_names]
            )
            deficits = target_fractions - current_fractions

        best_split_idx = int(np.argmax(deficits))
        best_split = split_names[best_split_idx]

        assignment[best_split].append(int(idx))
        split_totals[best_split] += vec
        split_counts[best_split] += 1

    for name in split_names:
        rng.shuffle(assignment[name])

    return assignment


def _aggregate_groups(
    ids: List[str],
    vectors: np.ndarray,
    group_values: List[str],
) -> Tuple[List[str], np.ndarray, List[List[str]]]:
    """Aggregate per-image vectors into per-group vectors.

    Args:
        ids: Sample IDs, one per image.
        vectors: ``(N, C)`` count matrix.
        group_values: Group key per image (same length as *ids*).

    Returns:
        ``(group_keys, group_vectors, group_members)`` where
        *group_keys* is a list of unique group names,
        *group_vectors* is ``(G, C)``, and *group_members* maps each
        group index to a list of sample IDs.
    """
    groups: Dict[str, List[int]] = defaultdict(list)
    for i, gv in enumerate(group_values):
        groups[gv].append(i)

    group_keys = sorted(groups.keys())
    n_classes = vectors.shape[1]
    group_vectors = np.zeros((len(group_keys), n_classes))
    group_members: List[List[str]] = []

    for gi, gk in enumerate(group_keys):
        member_indices = groups[gk]
        group_vectors[gi] = vectors[member_indices].sum(axis=0)
        group_members.append([ids[i] for i in member_indices])

    return group_keys, group_vectors, group_members


def _detect_label_type(
    dataset,
    label_field: str,
) -> Tuple[str, str]:
    """Detect the label type and return ``(type_key, values_path)``.

    Supports Classification, Detections, Polylines, and Segmentation fields.

    Returns:
        A tuple of ``(type_key, values_path)`` where *type_key* is one of
        ``"classification"``, ``"detections"``, ``"polylines"``,
        ``"segmentation"`` and *values_path* is the field path to pass to
        ``dataset.values()``.
    """
    import fiftyone.core.labels as fol

    schema = dataset.get_field_schema()
    field = schema.get(label_field)
    if field is None:
        raise ValueError(
            f"Label field '{label_field}' not found in dataset schema"
        )

    doc_type = getattr(field, "document_type", None)

    if doc_type is getattr(fol, "Detections", None):
        return "detections", f"{label_field}.detections"
    if doc_type is getattr(fol, "Polylines", None):
        return "polylines", f"{label_field}.polylines"
    if doc_type is getattr(fol, "Classification", None):
        return "classification", label_field
    if doc_type is getattr(fol, "Segmentation", None):
        return "segmentation", label_field

    raise ValueError(
        f"Unsupported label type for field '{label_field}'. "
        f"Expected Classification, Detections, Polylines, or Segmentation."
    )


def _extract_feature_vectors(
    dataset,
    label_field: str,
    class_list: List[str],
) -> Tuple[List[str], np.ndarray, List[str]]:
    """Extract per-image class count vectors from a FiftyOne dataset.

    Args:
        dataset: FiftyOne Dataset or DatasetView.
        label_field: Resolved label field name.
        class_list: Ordered class names.

    Returns:
        ``(sample_ids, vectors, class_names)`` where *vectors* is
        ``(N, C+1)`` — one column per class plus a virtual negative class,
        and *class_names* includes ``"_negative"`` as the last entry.
    """
    label_type, values_path = _detect_label_type(dataset, label_field)
    class_to_idx = {name: i for i, name in enumerate(class_list)}
    n_classes = len(class_list)

    out_classes = list(class_list) + ["_negative"]
    neg_idx = n_classes

    ids, labels = dataset.values(["id", values_path])
    n_samples = len(ids)
    vectors = np.zeros((n_samples, n_classes + 1))

    for i, sample_labels in enumerate(labels):
        if sample_labels is None or (
            isinstance(sample_labels, list) and len(sample_labels) == 0
        ):
            vectors[i, neg_idx] = 1.0
            continue

        if label_type == "classification":
            cls_name = sample_labels.label
            idx = class_to_idx.get(cls_name)
            if idx is not None:
                vectors[i, idx] = 1.0
            else:
                vectors[i, neg_idx] = 1.0

        elif label_type in ("detections", "polylines"):
            has_any = False
            for det in sample_labels:
                idx = class_to_idx.get(det.label)
                if idx is not None:
                    vectors[i, idx] += 1.0
                    has_any = True
            if not has_any:
                vectors[i, neg_idx] = 1.0

        elif label_type == "segmentation":
            mask = sample_labels.mask
            unique_vals = set(np.unique(mask)) - {0}
            has_any = False
            for val in unique_vals:
                cls_idx = int(val) - 1
                if 0 <= cls_idx < n_classes:
                    vectors[i, cls_idx] = 1.0
                    has_any = True
            if not has_any:
                vectors[i, neg_idx] = 1.0

    return ids, vectors, out_classes


def _discover_classes(
    dataset, label_field: str,
) -> List[str]:
    """Auto-discover class names from a FiftyOne dataset.

    Tries, in order:
    1. ``dataset.classes[label_field]``
    2. ``dataset.default_classes``
    3. Distinct label values from the dataset
    """
    # 1. Per-field classes
    field_classes = dataset.classes.get(label_field)
    if field_classes:
        return list(field_classes)

    # 2. Default classes
    if dataset.default_classes:
        return list(dataset.default_classes)

    # 3. Discover from data
    label_type, values_path = _detect_label_type(dataset, label_field)
    if label_type == "classification":
        return sorted(dataset.distinct(f"{label_field}.label"))
    elif label_type in ("detections", "polylines"):
        container = "detections" if label_type == "detections" else "polylines"
        return sorted(dataset.distinct(f"{label_field}.{container}.label"))
    elif label_type == "segmentation":
        logger.warning(
            "Cannot discover class names for segmentation field '%s'. "
            "Set dataset.classes['%s'] = [...] before splitting.",
            label_field, label_field,
        )
        return []

    return []


class DatasetSplitter:
    """Split a FiftyOne dataset into train/val/test with stratified balancing.

    Usage::

        splitter = DatasetSplitter(ratios={"train": 0.8, "val": 0.2})
        result = splitter.split(dataset)

    Args:
        ratios: Target proportions per split. Values must be positive and
            sum to at most 1.0.
        label_field: Label field for stratification. Auto-resolved if *None*.
        group_by: Sample field for group-aware splitting.
        seed: Random seed for reproducibility.
        split_field: Name of the sample field to write.
    """

    def __init__(
        self,
        ratios: Dict[str, float],
        label_field: Optional[str] = None,
        group_by: Optional[str] = None,
        seed: int = 42,
        split_field: str = "split",
    ) -> None:
        for name, val in ratios.items():
            if val <= 0:
                raise ValueError(
                    f"All ratios must be positive, got {name}={val}"
                )
        ratio_sum = sum(ratios.values())
        if ratio_sum > 1.0 + 1e-9:
            raise ValueError(
                f"Ratios must sum to <= 1.0, got {ratio_sum:.4f}"
            )

        self.ratios = ratios
        self.label_field = label_field
        self.group_by = group_by
        self.seed = seed
        self.split_field = split_field

    def split(
        self,
        dataset,
        class_list: Optional[List[str]] = None,
    ) -> SplitResult:
        """Split the dataset and write the split field.

        Args:
            dataset: FiftyOne Dataset or DatasetView.
            class_list: Ordered class names for stratification.
                Auto-discovered from the dataset if *None*.

        Returns:
            :class:`SplitResult` with distribution statistics.
        """
        n_samples = len(dataset)
        if n_samples == 0:
            raise ValueError("Dataset is empty")

        # Validate group_by field exists
        if self.group_by and not dataset.has_sample_field(self.group_by):
            raise ValueError(
                f"Field '{self.group_by}' not found in dataset schema"
            )

        # Resolve label field (module-level name; patchable in tests)
        resolved_field = resolve_label_field(dataset, self.label_field)

        # Discover classes if not provided
        if class_list is None:
            class_list = _discover_classes(dataset, resolved_field)

        # Extract feature vectors
        ids, vectors, out_classes = _extract_feature_vectors(
            dataset, resolved_field, class_list,
        )

        # Group handling
        group_counts = None
        if self.group_by:
            group_values = dataset.values(self.group_by)
            group_keys, group_vecs, group_members = _aggregate_groups(
                ids, vectors, group_values,
            )

            # Warn about large groups
            min_split_ratio = min(self.ratios.values())
            target_min_split = int(n_samples * min_split_ratio)
            for gi, gk in enumerate(group_keys):
                group_size = len(group_members[gi])
                if group_size > target_min_split > 0:
                    logger.warning(
                        "Group '%s' has %d images, exceeding target for "
                        "smallest split (%d). Consider smaller groups or "
                        "a different grouping field.",
                        gk, group_size, target_min_split,
                    )

            # Assign groups to splits
            group_assignment = _greedy_assign(
                group_vecs, self.ratios, self.seed,
            )

            # Expand group assignment back to sample IDs
            id_to_split: Dict[str, str] = {}
            group_counts = {}
            for split_name, group_indices in group_assignment.items():
                group_counts[split_name] = len(group_indices)
                for gi in group_indices:
                    for sid in group_members[gi]:
                        id_to_split[sid] = split_name

            split_values = [id_to_split[sid] for sid in ids]
        else:
            assignment = _greedy_assign(vectors, self.ratios, self.seed)

            idx_to_split: Dict[int, str] = {}
            for split_name, indices in assignment.items():
                for idx in indices:
                    idx_to_split[idx] = split_name

            split_values = [idx_to_split[i] for i in range(n_samples)]

        # Write the split field
        dataset.set_values(self.split_field, split_values)

        # Compute result statistics
        split_counts_result: Dict[str, int] = {}
        class_dist: Dict[str, Dict[str, int]] = {}
        for split_name in self.ratios:
            mask = [sv == split_name for sv in split_values]
            split_counts_result[split_name] = sum(mask)
            split_vecs = vectors[mask]
            class_dist[split_name] = {
                cls: int(split_vecs[:, ci].sum())
                for ci, cls in enumerate(out_classes)
            }

        logger.info(
            "Split complete: %s",
            ", ".join(
                f"{k}={v}" for k, v in split_counts_result.items()
            ),
        )

        return SplitResult(
            split_counts=split_counts_result,
            class_distribution=class_dist,
            group_counts=group_counts,
        )
