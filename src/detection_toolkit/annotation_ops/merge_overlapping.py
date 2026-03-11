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


from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union


def _polyline_to_shapely(polyline):
    """Convert fo.Polyline exterior ring to a Shapely Polygon."""
    ring = polyline.points[0]
    coords = [(x, y) for x, y in ring]
    poly = Polygon(coords)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly


def _shapely_to_rings(poly):
    """Convert a Shapely Polygon to FiftyOne points (list of rings)."""
    rings = []
    exterior = list(poly.exterior.coords)
    rings.append([(x, y) for x, y in exterior])
    for interior in poly.interiors:
        rings.append([(x, y) for x, y in interior.coords])
    return rings


def _merge_polygons(polylines):
    """Merge a list of same-class fo.Polylines via Shapely unary_union.

    Args:
        polylines: List of fo.Polyline instances (same class).

    Returns:
        List of fo.Polyline instances (one per resulting polygon).
    """
    if len(polylines) == 1:
        return polylines

    shapely_polys = [_polyline_to_shapely(p) for p in polylines]
    merged = unary_union(shapely_polys)

    first = polylines[0]
    label = first.label
    # Collect custom attributes from first polyline
    extra_attrs = {}
    for attr in first.field_names:
        if attr not in ("id", "attributes", "tags", "label", "points", "closed", "filled", "index"):
            extra_attrs[attr] = first[attr]

    if isinstance(merged, Polygon):
        geoms = [merged]
    elif isinstance(merged, MultiPolygon):
        geoms = list(merged.geoms)
    else:
        logger.warning("Unexpected geometry type from unary_union: %s", type(merged))
        return polylines

    import fiftyone as fo

    result = []
    for geom in geoms:
        if geom.is_empty:
            continue
        pl = fo.Polyline(
            label=label,
            points=_shapely_to_rings(geom),
            closed=True,
            filled=True,
            **extra_attrs,
        )
        result.append(pl)

    return result
