"""Tests for annotation_ops.merge_overlapping."""

import numpy as np
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


class TestMergeBboxes:
    """Tests for bounding box union merging."""

    def _make_detection(self, bbox, label="crack"):
        """Create an fo.Detection with given bbox [x, y, w, h]."""
        return fo.Detection(label=label, bounding_box=bbox)

    def test_two_overlapping(self):
        """Two overlapping boxes yield their union rectangle."""
        from detection_toolkit.annotation_ops.merge_overlapping import (
            _merge_bboxes,
        )

        d1 = self._make_detection([0.0, 0.0, 0.5, 0.5])
        d2 = self._make_detection([0.25, 0.25, 0.5, 0.5])

        result = _merge_bboxes([d1, d2])
        bb = result.bounding_box
        assert bb[0] == pytest.approx(0.0)   # x
        assert bb[1] == pytest.approx(0.0)   # y
        assert bb[2] == pytest.approx(0.75)  # w
        assert bb[3] == pytest.approx(0.75)  # h
        assert result.label == "crack"

    def test_preserves_first_attributes(self):
        """Custom attributes from the first detection are preserved."""
        from detection_toolkit.annotation_ops.merge_overlapping import (
            _merge_bboxes,
        )

        d1 = self._make_detection([0.0, 0.0, 0.5, 0.5])
        d1["confidence"] = 0.95
        d2 = self._make_detection([0.25, 0.25, 0.5, 0.5])
        d2["confidence"] = 0.1

        result = _merge_bboxes([d1, d2])
        assert result["confidence"] == 0.95

    def test_single_detection_passthrough(self):
        """A single detection is returned as-is."""
        from detection_toolkit.annotation_ops.merge_overlapping import (
            _merge_bboxes,
        )

        d = self._make_detection([0.1, 0.2, 0.3, 0.4])
        result = _merge_bboxes([d])
        assert result.bounding_box == [0.1, 0.2, 0.3, 0.4]


class TestMergeMasks:
    """Tests for instance mask merging."""

    def _make_masked_detection(self, bbox, mask, label="crack"):
        """Create fo.Detection with a boolean mask relative to bbox."""
        return fo.Detection(label=label, bounding_box=bbox, mask=mask)

    def test_two_overlapping_masks(self):
        """Two overlapping masked detections merge into one."""
        from detection_toolkit.annotation_ops.merge_overlapping import (
            _merge_masks,
        )

        # Image is 100x100
        # d1: bbox [0, 0, 0.5, 0.5] → pixels [0:50, 0:50], full mask
        mask1 = np.ones((50, 50), dtype=bool)
        d1 = self._make_masked_detection([0.0, 0.0, 0.5, 0.5], mask1)

        # d2: bbox [0.25, 0.25, 0.5, 0.5] → pixels [25:75, 25:75], full mask
        mask2 = np.ones((50, 50), dtype=bool)
        d2 = self._make_masked_detection([0.25, 0.25, 0.5, 0.5], mask2)

        result = _merge_masks([d1, d2], width=100, height=100)
        bb = result.bounding_box
        # Union bbox should cover [0, 0] to [0.75, 0.75]
        assert bb[0] == pytest.approx(0.0)
        assert bb[1] == pytest.approx(0.0)
        assert bb[2] == pytest.approx(0.75)
        assert bb[3] == pytest.approx(0.75)
        # Mask should exist and be the right shape
        assert result.mask is not None
        assert result.mask.shape == (75, 75)
        assert result.mask.any()

    def test_single_detection_passthrough(self):
        """A single masked detection is returned as-is."""
        from detection_toolkit.annotation_ops.merge_overlapping import (
            _merge_masks,
        )

        mask = np.ones((30, 40), dtype=bool)
        d = self._make_masked_detection([0.1, 0.2, 0.4, 0.3], mask)

        result = _merge_masks([d], width=100, height=100)
        assert result.bounding_box == [0.1, 0.2, 0.4, 0.3]
        np.testing.assert_array_equal(result.mask, mask)

    def test_preserves_first_attributes(self):
        """Custom attributes from the first detection are preserved."""
        from detection_toolkit.annotation_ops.merge_overlapping import (
            _merge_masks,
        )

        mask1 = np.ones((50, 50), dtype=bool)
        d1 = self._make_masked_detection([0.0, 0.0, 0.5, 0.5], mask1)
        d1["confidence"] = 0.8

        mask2 = np.ones((50, 50), dtype=bool)
        d2 = self._make_masked_detection([0.25, 0.25, 0.5, 0.5], mask2)
        d2["confidence"] = 0.2

        result = _merge_masks([d1, d2], width=100, height=100)
        assert result["confidence"] == 0.8


@pytest.mark.requires_mongod
class TestMergeOverlappingIntegration:
    """Integration tests using real FiftyOne datasets."""

    @pytest.fixture
    def polyline_dataset(self):
        """Dataset with overlapping Polyline annotations."""
        dataset = fo.Dataset()
        dataset.add_sample(
            fo.Sample(
                filepath="/tmp/test.jpg",
                segmentations=fo.Polylines(
                    polylines=[
                        fo.Polyline(
                            label="crack",
                            points=[[(0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.5)]],
                            closed=True,
                            filled=True,
                        ),
                        fo.Polyline(
                            label="crack",
                            points=[[(0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)]],
                            closed=True,
                            filled=True,
                        ),
                        fo.Polyline(
                            label="spall",
                            points=[[(0.3, 0.3), (0.6, 0.3), (0.6, 0.6), (0.3, 0.6)]],
                            closed=True,
                            filled=True,
                        ),
                    ]
                ),
            )
        )
        yield dataset
        fo.delete_dataset(dataset.name)

    @pytest.fixture
    def bbox_dataset(self):
        """Dataset with overlapping bbox-only Detections."""
        dataset = fo.Dataset()
        dataset.add_sample(
            fo.Sample(
                filepath="/tmp/test.jpg",
                ground_truth=fo.Detections(
                    detections=[
                        fo.Detection(label="crack", bounding_box=[0.0, 0.0, 0.5, 0.5]),
                        fo.Detection(label="crack", bounding_box=[0.25, 0.25, 0.5, 0.5]),
                        fo.Detection(label="spall", bounding_box=[0.0, 0.0, 0.3, 0.3]),
                    ]
                ),
            )
        )
        yield dataset
        fo.delete_dataset(dataset.name)

    def test_polylines_merged(self, polyline_dataset):
        """Two overlapping same-class polylines merge; different class untouched."""
        from detection_toolkit.annotation_ops import merge_overlapping

        merge_overlapping(polyline_dataset, label_field="segmentations")
        sample = polyline_dataset.first()
        polylines = sample.segmentations.polylines
        labels = [p.label for p in polylines]
        assert labels.count("crack") == 1  # merged
        assert labels.count("spall") == 1  # untouched

    def test_bboxes_merged(self, bbox_dataset):
        """Two overlapping same-class bboxes merge; different class untouched."""
        from detection_toolkit.annotation_ops import merge_overlapping

        merge_overlapping(bbox_dataset, label_field="ground_truth")
        sample = bbox_dataset.first()
        dets = sample.ground_truth.detections
        labels = [d.label for d in dets]
        assert labels.count("crack") == 1  # merged
        assert labels.count("spall") == 1  # untouched

    def test_chain_reaction_polylines(self):
        """A↔B↔C chain where A∩C=∅ still merges all three."""
        dataset = fo.Dataset()
        dataset.add_sample(
            fo.Sample(
                filepath="/tmp/test.jpg",
                segmentations=fo.Polylines(
                    polylines=[
                        fo.Polyline(
                            label="crack",
                            points=[[(0.0, 0.0), (0.3, 0.0), (0.3, 0.3), (0.0, 0.3)]],
                            closed=True, filled=True,
                        ),
                        fo.Polyline(
                            label="crack",
                            points=[[(0.2, 0.2), (0.5, 0.2), (0.5, 0.5), (0.2, 0.5)]],
                            closed=True, filled=True,
                        ),
                        fo.Polyline(
                            label="crack",
                            points=[[(0.4, 0.4), (0.7, 0.4), (0.7, 0.7), (0.4, 0.7)]],
                            closed=True, filled=True,
                        ),
                    ]
                ),
            )
        )
        try:
            from detection_toolkit.annotation_ops import merge_overlapping

            merge_overlapping(dataset, label_field="segmentations")
            sample = dataset.first()
            assert len(sample.segmentations.polylines) == 1
        finally:
            fo.delete_dataset(dataset.name)

    def test_masked_detections_merged(self):
        """Two overlapping masked detections merge correctly end-to-end."""
        dataset = fo.Dataset()
        import cv2

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        filepath = "/tmp/test_masked.jpg"
        cv2.imwrite(filepath, img)

        mask1 = np.ones((50, 50), dtype=bool)
        mask2 = np.ones((50, 50), dtype=bool)
        dataset.add_sample(
            fo.Sample(
                filepath=filepath,
                ground_truth=fo.Detections(
                    detections=[
                        fo.Detection(
                            label="crack",
                            bounding_box=[0.0, 0.0, 0.5, 0.5],
                            mask=mask1,
                        ),
                        fo.Detection(
                            label="crack",
                            bounding_box=[0.25, 0.25, 0.5, 0.5],
                            mask=mask2,
                        ),
                    ]
                ),
            )
        )
        try:
            from detection_toolkit.annotation_ops import merge_overlapping

            merge_overlapping(dataset, label_field="ground_truth")
            sample = dataset.first()
            dets = sample.ground_truth.detections
            assert len(dets) == 1
            assert dets[0].label == "crack"
            assert dets[0].mask is not None
            assert dets[0].bounding_box[2] == pytest.approx(0.75)
        finally:
            fo.delete_dataset(dataset.name)

    def test_no_annotations_skip(self):
        """Sample with no annotations is skipped without error."""
        dataset = fo.Dataset()
        dataset.add_sample(
            fo.Sample(
                filepath="/tmp/test.jpg",
                segmentations=fo.Polylines(polylines=[]),
            )
        )
        try:
            from detection_toolkit.annotation_ops import merge_overlapping

            merge_overlapping(dataset, label_field="segmentations")
            sample = dataset.first()
            assert len(sample.segmentations.polylines) == 0
        finally:
            fo.delete_dataset(dataset.name)

    def test_single_annotation_unchanged(self):
        """A single annotation passes through without modification."""
        dataset = fo.Dataset()
        dataset.add_sample(
            fo.Sample(
                filepath="/tmp/test.jpg",
                segmentations=fo.Polylines(
                    polylines=[
                        fo.Polyline(
                            label="crack",
                            points=[[(0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.5)]],
                            closed=True, filled=True,
                        ),
                    ]
                ),
            )
        )
        try:
            from detection_toolkit.annotation_ops import merge_overlapping

            merge_overlapping(dataset, label_field="segmentations")
            sample = dataset.first()
            assert len(sample.segmentations.polylines) == 1
        finally:
            fo.delete_dataset(dataset.name)
