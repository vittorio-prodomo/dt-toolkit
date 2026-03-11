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
