"""Tests for annotation_ops.merge_overlapping."""

import pytest
from shapely.geometry import box


class TestFindConnectedComponents:
    """Tests for the Union-Find connected components logic."""

    def test_no_overlaps(self):
        """Disjoint geometries: each is its own component."""
        from detection_toolkit.annotation_ops.merge_overlapping import (
            _find_connected_components,
        )

        geoms = [box(0, 0, 1, 1), box(2, 2, 3, 3), box(4, 4, 5, 5)]
        components = _find_connected_components(geoms)
        assert len(components) == 3
        for comp in components:
            assert len(comp) == 1

    def test_two_overlapping(self):
        """Two overlapping boxes merge into one component."""
        from detection_toolkit.annotation_ops.merge_overlapping import (
            _find_connected_components,
        )

        geoms = [box(0, 0, 2, 2), box(1, 1, 3, 3)]
        components = _find_connected_components(geoms)
        assert len(components) == 1
        assert components[0] == {0, 1}

    def test_chain_reaction(self):
        """A overlaps B, B overlaps C, but A doesn't touch C — all merge."""
        from detection_toolkit.annotation_ops.merge_overlapping import (
            _find_connected_components,
        )

        geoms = [box(0, 0, 2, 2), box(1, 1, 3, 3), box(2, 2, 4, 4)]
        components = _find_connected_components(geoms)
        assert len(components) == 1
        assert components[0] == {0, 1, 2}

    def test_touching_not_merged(self):
        """Boxes that share an edge but no area are NOT merged."""
        from detection_toolkit.annotation_ops.merge_overlapping import (
            _find_connected_components,
        )

        geoms = [box(0, 0, 1, 1), box(1, 0, 2, 1)]
        components = _find_connected_components(geoms)
        assert len(components) == 2

    def test_single_geometry(self):
        """One geometry: one component."""
        from detection_toolkit.annotation_ops.merge_overlapping import (
            _find_connected_components,
        )

        geoms = [box(0, 0, 1, 1)]
        components = _find_connected_components(geoms)
        assert len(components) == 1
        assert components[0] == {0}

    def test_empty_list(self):
        """No geometries: no components."""
        from detection_toolkit.annotation_ops.merge_overlapping import (
            _find_connected_components,
        )

        components = _find_connected_components([])
        assert components == []


import fiftyone as fo


class TestMergePolygons:
    """Tests for Shapely-based polygon merging."""

    def _make_polyline(self, points_rings, label="crack"):
        """Create an fo.Polyline with given rings (relative coords)."""
        return fo.Polyline(
            label=label,
            points=points_rings,
            closed=True,
            filled=True,
        )

    def test_two_overlapping_squares(self):
        """Two overlapping squares merge into one polygon."""
        from detection_toolkit.annotation_ops.merge_overlapping import (
            _merge_polygons,
        )

        p1 = self._make_polyline([[(0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.5)]])
        p2 = self._make_polyline([[(0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)]])

        result = _merge_polygons([p1, p2])
        assert len(result) == 1
        merged = result[0]
        assert merged.label == "crack"
        assert merged.closed is True
        assert merged.filled is True
        assert len(merged.points) >= 1  # at least exterior ring

    def test_preserves_first_attributes(self):
        """Custom attributes come from the first polyline."""
        from detection_toolkit.annotation_ops.merge_overlapping import (
            _merge_polygons,
        )

        p1 = self._make_polyline([[(0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.5)]])
        p1["confidence"] = 0.9
        p2 = self._make_polyline([[(0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)]])
        p2["confidence"] = 0.3

        result = _merge_polygons([p1, p2])
        assert result[0]["confidence"] == 0.9

    def test_multipolygon_splits(self):
        """If unary_union yields MultiPolygon, each part is a separate Polyline."""
        from detection_toolkit.annotation_ops.merge_overlapping import (
            _merge_polygons,
        )

        # Two disjoint squares
        p1 = self._make_polyline([[(0.0, 0.0), (0.1, 0.0), (0.1, 0.1), (0.0, 0.1)]])
        p2 = self._make_polyline([[(0.9, 0.9), (1.0, 0.9), (1.0, 1.0), (0.9, 1.0)]])

        result = _merge_polygons([p1, p2])
        assert len(result) == 2  # MultiPolygon → 2 separate polylines

    def test_single_polyline_passthrough(self):
        """A single polyline is returned unchanged."""
        from detection_toolkit.annotation_ops.merge_overlapping import (
            _merge_polygons,
        )

        p = self._make_polyline([[(0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.5)]])
        result = _merge_polygons([p])
        assert len(result) == 1
        assert result[0].points == p.points
