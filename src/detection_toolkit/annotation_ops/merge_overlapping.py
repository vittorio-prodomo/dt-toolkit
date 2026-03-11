"""Merge overlapping same-class annotations within a FiftyOne dataset."""

import logging
from typing import List, Set

logger = logging.getLogger(__name__)


def _find_connected_components(geometries) -> List[Set[int]]:
    """Find connected components among geometries that overlap in area.

    Uses Union-Find for O(n² α(n)) performance.

    Args:
        geometries: List of Shapely geometries.

    Returns:
        List of sets, each set containing indices of a connected component.
    """
    n = len(geometries)
    if n == 0:
        return []

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if geometries[i].intersection(geometries[j]).area > 0:
                union(i, j)

    groups = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, set()).add(i)

    return list(groups.values())


def merge_overlapping(dataset, label_field=None):
    """Merge overlapping same-class annotations in-place.

    Args:
        dataset: FiftyOne Dataset or DatasetView.
        label_field: Label field name. Auto-resolved if None.

    Returns:
        The dataset (mutated in-place).
    """
    raise NotImplementedError
