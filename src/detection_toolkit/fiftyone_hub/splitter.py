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
