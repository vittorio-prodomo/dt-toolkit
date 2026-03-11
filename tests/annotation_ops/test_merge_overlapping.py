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
