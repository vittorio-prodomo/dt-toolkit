"""Tests for the dataset splitter algorithm and API."""

import numpy as np
import pytest


class TestGreedyAssign:
    """Tests for the core greedy bin-packing assignment."""

    def test_two_way_split_respects_ratio(self):
        """80/20 split of 100 items should produce ~80/~20."""
        from detection_toolkit.fiftyone_hub.splitter import _greedy_assign

        rng = np.random.RandomState(42)
        vectors = rng.randint(0, 6, size=(100, 3)).astype(float)
        ratios = {"train": 0.8, "val": 0.2}

        assignment = _greedy_assign(vectors, ratios, seed=42)

        assert set(assignment.keys()) == {"train", "val"}
        assert len(assignment["train"]) + len(assignment["val"]) == 100
        assert abs(len(assignment["train"]) - 80) <= 5
        assert abs(len(assignment["val"]) - 20) <= 5

    def test_three_way_split(self):
        """70/15/15 split of 100 items."""
        from detection_toolkit.fiftyone_hub.splitter import _greedy_assign

        rng = np.random.RandomState(42)
        vectors = rng.randint(0, 6, size=(100, 3)).astype(float)
        ratios = {"train": 0.7, "val": 0.15, "test": 0.15}

        assignment = _greedy_assign(vectors, ratios, seed=42)

        total = sum(len(v) for v in assignment.values())
        assert total == 100
        assert abs(len(assignment["train"]) - 70) <= 5

    def test_class_balance_maintained(self):
        """Per-class distribution should be roughly proportional across splits."""
        from detection_toolkit.fiftyone_hub.splitter import _greedy_assign

        vectors = np.zeros((200, 2))
        vectors[:, 0] = 5.0
        vectors[:150, 1] = 1.0

        ratios = {"train": 0.8, "val": 0.2}
        assignment = _greedy_assign(vectors, ratios, seed=42)

        train_rare = vectors[assignment["train"], 1].sum()
        val_rare = vectors[assignment["val"], 1].sum()
        total_rare = train_rare + val_rare
        train_ratio = train_rare / total_rare

        assert abs(train_ratio - 0.8) < 0.1

    def test_deterministic_with_same_seed(self):
        """Same seed should produce identical assignments."""
        from detection_toolkit.fiftyone_hub.splitter import _greedy_assign

        rng = np.random.RandomState(99)
        vectors = rng.randint(0, 6, size=(50, 3)).astype(float)
        ratios = {"train": 0.8, "val": 0.2}

        a1 = _greedy_assign(vectors, ratios, seed=7)
        a2 = _greedy_assign(vectors, ratios, seed=7)

        assert a1["train"] == a2["train"]
        assert a1["val"] == a2["val"]

    def test_different_seeds_differ(self):
        """Different seeds should (usually) produce different assignments."""
        from detection_toolkit.fiftyone_hub.splitter import _greedy_assign

        rng = np.random.RandomState(99)
        vectors = rng.randint(0, 6, size=(50, 3)).astype(float)
        ratios = {"train": 0.8, "val": 0.2}

        a1 = _greedy_assign(vectors, ratios, seed=1)
        a2 = _greedy_assign(vectors, ratios, seed=2)

        assert a1["train"] != a2["train"]

    def test_single_item(self):
        """A single item goes to the largest split."""
        from detection_toolkit.fiftyone_hub.splitter import _greedy_assign

        vectors = np.array([[1.0, 0.0]])
        ratios = {"train": 0.8, "val": 0.2}

        assignment = _greedy_assign(vectors, ratios, seed=42)

        assert len(assignment["train"]) == 1
        assert len(assignment["val"]) == 0

    def test_all_zero_vectors(self):
        """All-zero vectors (all unlabeled) should still split correctly."""
        from detection_toolkit.fiftyone_hub.splitter import _greedy_assign

        vectors = np.zeros((50, 3))
        ratios = {"train": 0.8, "val": 0.2}

        assignment = _greedy_assign(vectors, ratios, seed=42)

        total = len(assignment["train"]) + len(assignment["val"])
        assert total == 50


class TestAggregateGroups:
    """Tests for group-aware vector aggregation."""

    def test_groups_summed(self):
        """Group vectors are element-wise sums of member vectors."""
        from detection_toolkit.fiftyone_hub.splitter import _aggregate_groups

        ids = ["a", "b", "c", "d"]
        vectors = np.array([
            [1, 0],
            [0, 2],
            [3, 1],
            [1, 1],
        ], dtype=float)
        group_values = ["g1", "g1", "g2", "g2"]

        group_keys, group_vecs, group_members = _aggregate_groups(
            ids, vectors, group_values,
        )

        assert len(group_keys) == 2
        g1_idx = group_keys.index("g1")
        g2_idx = group_keys.index("g2")
        np.testing.assert_array_equal(group_vecs[g1_idx], [1, 2])
        np.testing.assert_array_equal(group_vecs[g2_idx], [4, 2])
        assert set(group_members[g1_idx]) == {"a", "b"}
        assert set(group_members[g2_idx]) == {"c", "d"}

    def test_single_member_groups(self):
        """Groups of size 1 pass through unchanged."""
        from detection_toolkit.fiftyone_hub.splitter import _aggregate_groups

        ids = ["a", "b"]
        vectors = np.array([[1, 2], [3, 4]], dtype=float)
        group_values = ["g1", "g2"]

        group_keys, group_vecs, group_members = _aggregate_groups(
            ids, vectors, group_values,
        )

        assert len(group_keys) == 2
        assert group_members[group_keys.index("g1")] == ["a"]
        assert group_members[group_keys.index("g2")] == ["b"]


from unittest.mock import MagicMock, patch


class TestExtractFeatureVectors:
    """Tests for feature vector extraction from FiftyOne labels."""

    def _make_mock_dataset(self, labels_list, label_type):
        """Helper: build a mock dataset returning bulk values()."""
        n = len(labels_list)
        sample_ids = [f"id_{i}" for i in range(n)]

        dataset = MagicMock()
        dataset.__len__ = MagicMock(return_value=n)

        def mock_values(fields):
            result = []
            for f in fields:
                if f == "id":
                    result.append(sample_ids)
                else:
                    result.append(labels_list)
            return result

        dataset.values = mock_values
        return dataset, sample_ids

    @patch("detection_toolkit.fiftyone_hub.splitter._detect_label_type")
    def test_detections_count_vector(self, mock_detect):
        """Detection labels produce instance count vectors."""
        from detection_toolkit.fiftyone_hub.splitter import _extract_feature_vectors

        mock_detect.return_value = ("detections", "ground_truth.detections")

        class_list = ["crack", "spall"]
        labels = [
            [
                MagicMock(label="crack"),
                MagicMock(label="crack"),
                MagicMock(label="spall"),
            ],
            [MagicMock(label="crack")],
            None,
        ]
        dataset, sample_ids = self._make_mock_dataset(labels, "detections")

        ids, vectors, classes = _extract_feature_vectors(
            dataset, "ground_truth", class_list,
        )

        assert ids == sample_ids
        assert classes == ["crack", "spall", "_negative"]
        assert vectors.shape == (3, 3)
        np.testing.assert_array_equal(vectors[0], [2, 1, 0])
        np.testing.assert_array_equal(vectors[1], [1, 0, 0])
        np.testing.assert_array_equal(vectors[2], [0, 0, 1])

    @patch("detection_toolkit.fiftyone_hub.splitter._detect_label_type")
    def test_classifications_one_hot(self, mock_detect):
        """Classification labels produce one-hot vectors."""
        from detection_toolkit.fiftyone_hub.splitter import _extract_feature_vectors

        mock_detect.return_value = ("classification", "predictions")

        class_list = ["cat", "dog"]
        labels = [
            MagicMock(label="cat"),
            MagicMock(label="dog"),
            None,
        ]
        dataset, sample_ids = self._make_mock_dataset(labels, "classification")

        ids, vectors, classes = _extract_feature_vectors(
            dataset, "predictions", class_list,
        )

        assert vectors.shape == (3, 3)
        np.testing.assert_array_equal(vectors[0], [1, 0, 0])
        np.testing.assert_array_equal(vectors[1], [0, 1, 0])
        np.testing.assert_array_equal(vectors[2], [0, 0, 1])

    @patch("detection_toolkit.fiftyone_hub.splitter._detect_label_type")
    def test_segmentation_binary_presence(self, mock_detect):
        """Segmentation labels produce binary presence vectors."""
        from detection_toolkit.fiftyone_hub.splitter import _extract_feature_vectors

        mock_detect.return_value = ("segmentation", "segmentations")

        class_list = ["road", "building"]
        mask_a = MagicMock()
        mask_a.mask = np.array([[0, 1], [1, 2]])  # has class 1 and 2
        mask_b = MagicMock()
        mask_b.mask = np.array([[0, 0], [0, 1]])  # has class 1 only
        labels = [mask_a, mask_b, None]
        dataset, sample_ids = self._make_mock_dataset(labels, "segmentation")

        ids, vectors, classes = _extract_feature_vectors(
            dataset, "segmentations", class_list,
        )

        assert vectors.shape == (3, 3)
        # mask_a: class indices 1 and 2 -> "road" (idx 0->pixel 1) and "building" (idx 1->pixel 2)
        np.testing.assert_array_equal(vectors[0], [1, 1, 0])
        # mask_b: class index 1 only -> "road"
        np.testing.assert_array_equal(vectors[1], [1, 0, 0])
        # None -> negative
        np.testing.assert_array_equal(vectors[2], [0, 0, 1])

    @patch("detection_toolkit.fiftyone_hub.splitter._detect_label_type")
    def test_all_unlabeled(self, mock_detect):
        """All-unlabeled dataset: every image gets the negative class."""
        from detection_toolkit.fiftyone_hub.splitter import _extract_feature_vectors

        mock_detect.return_value = ("detections", "ground_truth.detections")

        class_list = ["crack"]
        labels = [None, None, None]
        dataset, sample_ids = self._make_mock_dataset(labels, "detections")

        ids, vectors, classes = _extract_feature_vectors(
            dataset, "ground_truth", class_list,
        )

        assert vectors.shape == (3, 2)
        np.testing.assert_array_equal(vectors[:, -1], [1, 1, 1])
