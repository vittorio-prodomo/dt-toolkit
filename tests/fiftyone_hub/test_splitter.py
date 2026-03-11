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


class TestDatasetSplitter:
    """Tests for the DatasetSplitter public API."""

    def test_invalid_ratios_sum(self):
        """Ratios summing to > 1.0 raise ValueError."""
        from detection_toolkit.fiftyone_hub.splitter import DatasetSplitter

        with pytest.raises(ValueError, match="sum to"):
            DatasetSplitter(ratios={"train": 0.9, "val": 0.2})

    def test_invalid_ratios_negative(self):
        """Negative ratios raise ValueError."""
        from detection_toolkit.fiftyone_hub.splitter import DatasetSplitter

        with pytest.raises(ValueError, match="positive"):
            DatasetSplitter(ratios={"train": -0.1, "val": 0.2})

    def test_invalid_ratios_zero(self):
        """Zero ratios raise ValueError."""
        from detection_toolkit.fiftyone_hub.splitter import DatasetSplitter

        with pytest.raises(ValueError, match="positive"):
            DatasetSplitter(ratios={"train": 0.0, "val": 1.0})

    def test_empty_dataset_raises(self):
        """Splitting an empty dataset raises ValueError."""
        from detection_toolkit.fiftyone_hub.splitter import DatasetSplitter

        dataset = MagicMock()
        dataset.__len__ = MagicMock(return_value=0)

        splitter = DatasetSplitter(ratios={"train": 0.8, "val": 0.2})
        with pytest.raises(ValueError, match="empty"):
            splitter.split(dataset)

    def test_nonexistent_group_field_raises(self):
        """group_by field not in schema raises ValueError."""
        from detection_toolkit.fiftyone_hub.splitter import DatasetSplitter

        dataset = MagicMock()
        dataset.__len__ = MagicMock(return_value=10)
        dataset.has_sample_field = MagicMock(return_value=False)

        splitter = DatasetSplitter(
            ratios={"train": 0.8, "val": 0.2},
            group_by="bridge_id",
        )
        with pytest.raises(ValueError, match="bridge_id"):
            splitter.split(dataset)

    @patch("detection_toolkit.fiftyone_hub.splitter._extract_feature_vectors")
    @patch("detection_toolkit.fiftyone_hub.splitter._discover_classes")
    @patch("detection_toolkit.fiftyone_hub.splitter.resolve_label_field")
    def test_split_writes_field_and_returns_result(
        self, mock_resolve, mock_discover, mock_extract,
    ):
        """split() should call set_values and return a SplitResult."""
        from detection_toolkit.fiftyone_hub.splitter import DatasetSplitter

        mock_resolve.return_value = "ground_truth"
        mock_discover.return_value = ["crack", "spall"]

        ids = [f"id_{i}" for i in range(10)]
        vectors = np.zeros((10, 3))
        vectors[:7, 0] = 1.0   # 7 items with class 0
        vectors[7:, 1] = 1.0   # 3 items with class 1
        mock_extract.return_value = (ids, vectors, ["crack", "spall", "_negative"])

        dataset = MagicMock()
        dataset.__len__ = MagicMock(return_value=10)
        dataset.has_sample_field = MagicMock(return_value=False)

        splitter = DatasetSplitter(ratios={"train": 0.8, "val": 0.2})
        result = splitter.split(dataset)

        # Check set_values was called with the split field
        dataset.set_values.assert_called_once()
        call_args = dataset.set_values.call_args
        assert call_args[0][0] == "split"  # field name
        split_values = call_args[0][1]
        assert len(split_values) == 10
        assert set(split_values) <= {"train", "val"}

        # Check SplitResult
        assert sum(result.split_counts.values()) == 10
        assert "train" in result.class_distribution
        assert "val" in result.class_distribution
        assert result.group_counts is None

    @patch("detection_toolkit.fiftyone_hub.splitter._extract_feature_vectors")
    @patch("detection_toolkit.fiftyone_hub.splitter._discover_classes")
    @patch("detection_toolkit.fiftyone_hub.splitter.resolve_label_field")
    def test_split_with_group_by(
        self, mock_resolve, mock_discover, mock_extract,
    ):
        """split() with group_by aggregates before assigning."""
        from detection_toolkit.fiftyone_hub.splitter import DatasetSplitter

        mock_resolve.return_value = "ground_truth"
        mock_discover.return_value = ["a", "b"]

        ids = [f"id_{i}" for i in range(6)]
        vectors = np.array([
            [1, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 1],
        ], dtype=float)
        mock_extract.return_value = (ids, vectors, ["a", "b", "_negative"])

        dataset = MagicMock()
        dataset.__len__ = MagicMock(return_value=6)

        def has_field(f):
            return f in ("bridge_id", "ground_truth")

        dataset.has_sample_field = MagicMock(side_effect=has_field)
        # group_by field values
        dataset.values = MagicMock(return_value=["g1", "g1", "g2", "g2", "g3", "g3"])

        splitter = DatasetSplitter(
            ratios={"train": 0.67, "val": 0.33},
            group_by="bridge_id",
        )
        result = splitter.split(dataset)

        assert result.group_counts is not None
        assert sum(result.group_counts.values()) == 3  # 3 groups total
        assert sum(result.split_counts.values()) == 6  # 6 images total

    @patch("detection_toolkit.fiftyone_hub.splitter._extract_feature_vectors")
    @patch("detection_toolkit.fiftyone_hub.splitter._discover_classes")
    @patch("detection_toolkit.fiftyone_hub.splitter.resolve_label_field")
    def test_no_label_field_raises(self, mock_resolve, mock_discover, mock_extract):
        """When no suitable label field is found, ValueError is raised."""
        from detection_toolkit.fiftyone_hub.splitter import DatasetSplitter

        mock_resolve.side_effect = ValueError(
            "Could not resolve a label field."
        )

        dataset = MagicMock()
        dataset.__len__ = MagicMock(return_value=10)
        dataset.has_sample_field = MagicMock(return_value=False)

        splitter = DatasetSplitter(ratios={"train": 0.8, "val": 0.2})
        with pytest.raises(ValueError, match="Could not resolve"):
            splitter.split(dataset)


class TestCLIArgs:
    """Tests for dt-split argument parsing."""

    def test_default_args(self):
        from detection_toolkit.scripts.split import parse_args

        args = parse_args(["my_dataset"])
        assert args.name == "my_dataset"
        assert args.ratio == [0.8]
        assert args.label_field is None
        assert args.group_by is None
        assert args.seed == 42
        assert args.split_field == "split"
        assert args.yes is False

    def test_two_way_ratio(self):
        from detection_toolkit.scripts.split import parse_args

        args = parse_args(["ds", "--ratio", "0.75"])
        assert args.ratio == [0.75]

    def test_three_way_ratio(self):
        from detection_toolkit.scripts.split import parse_args

        args = parse_args(["ds", "--ratio", "0.7", "0.15"])
        assert args.ratio == [0.7, 0.15]

    def test_all_options(self):
        from detection_toolkit.scripts.split import parse_args

        args = parse_args([
            "ds",
            "--ratio", "0.7", "0.15",
            "--label-field", "segmentations",
            "--group-by", "bridge_id",
            "--seed", "99",
            "--split-field", "my_split",
            "--yes",
        ])
        assert args.label_field == "segmentations"
        assert args.group_by == "bridge_id"
        assert args.seed == 99
        assert args.split_field == "my_split"
        assert args.yes is True

    def test_ratio_to_ratios_dict_two_way(self):
        from detection_toolkit.scripts.split import _ratio_to_dict

        d = _ratio_to_dict([0.8])
        assert d == {"train": pytest.approx(0.8), "val": pytest.approx(0.2)}

    def test_ratio_to_ratios_dict_three_way(self):
        from detection_toolkit.scripts.split import _ratio_to_dict

        d = _ratio_to_dict([0.7, 0.15])
        assert d == {
            "train": pytest.approx(0.7),
            "val": pytest.approx(0.15),
            "test": pytest.approx(0.15),
        }

    def test_ratio_to_ratios_dict_invalid(self):
        from detection_toolkit.scripts.split import _ratio_to_dict

        with pytest.raises(ValueError):
            _ratio_to_dict([0.5, 0.3, 0.3])  # too many values


class TestEdgeCases:
    """Tests for edge cases and warnings."""

    @patch("detection_toolkit.fiftyone_hub.splitter._extract_feature_vectors")
    @patch("detection_toolkit.fiftyone_hub.splitter._discover_classes")
    @patch("detection_toolkit.fiftyone_hub.splitter.resolve_label_field")
    def test_existing_field_overwritten(
        self, mock_resolve, mock_discover, mock_extract,
    ):
        """When split_field already exists, split() overwrites it."""
        from detection_toolkit.fiftyone_hub.splitter import DatasetSplitter

        mock_resolve.return_value = "ground_truth"
        mock_discover.return_value = ["a"]

        ids = [f"id_{i}" for i in range(10)]
        vectors = np.ones((10, 2))
        mock_extract.return_value = (ids, vectors, ["a", "_negative"])

        dataset = MagicMock()
        dataset.__len__ = MagicMock(return_value=10)
        dataset.has_sample_field = MagicMock(return_value=True)

        splitter = DatasetSplitter(ratios={"train": 0.8, "val": 0.2})
        result = splitter.split(dataset)

        dataset.set_values.assert_called_once()
        assert sum(result.split_counts.values()) == 10

    @patch("detection_toolkit.fiftyone_hub.splitter._extract_feature_vectors")
    @patch("detection_toolkit.fiftyone_hub.splitter._discover_classes")
    @patch("detection_toolkit.fiftyone_hub.splitter.resolve_label_field")
    def test_large_group_still_assigned(
        self, mock_resolve, mock_discover, mock_extract,
    ):
        """A group larger than the smallest split target is still assigned."""
        from detection_toolkit.fiftyone_hub.splitter import DatasetSplitter

        mock_resolve.return_value = "ground_truth"
        mock_discover.return_value = ["a"]

        ids = [f"id_{i}" for i in range(10)]
        vectors = np.ones((10, 2))
        mock_extract.return_value = (ids, vectors, ["a", "_negative"])

        dataset = MagicMock()
        dataset.__len__ = MagicMock(return_value=10)

        def has_field(f):
            return True

        dataset.has_sample_field = MagicMock(side_effect=has_field)
        dataset.values = MagicMock(
            return_value=["big"] * 8 + ["small"] * 2,
        )

        splitter = DatasetSplitter(
            ratios={"train": 0.8, "val": 0.2},
            group_by="source",
        )
        result = splitter.split(dataset)

        assert sum(result.split_counts.values()) == 10
        assert sum(result.group_counts.values()) == 2

    @patch("detection_toolkit.fiftyone_hub.splitter._extract_feature_vectors")
    @patch("detection_toolkit.fiftyone_hub.splitter._discover_classes")
    @patch("detection_toolkit.fiftyone_hub.splitter.resolve_label_field")
    def test_single_image_dataset(
        self, mock_resolve, mock_discover, mock_extract,
    ):
        """Dataset with one image assigns it to the largest split."""
        from detection_toolkit.fiftyone_hub.splitter import DatasetSplitter

        mock_resolve.return_value = "ground_truth"
        mock_discover.return_value = ["a"]
        mock_extract.return_value = (
            ["id_0"],
            np.array([[1.0, 0.0]]),
            ["a", "_negative"],
        )

        dataset = MagicMock()
        dataset.__len__ = MagicMock(return_value=1)
        dataset.has_sample_field = MagicMock(return_value=False)

        splitter = DatasetSplitter(ratios={"train": 0.8, "val": 0.2})
        result = splitter.split(dataset)

        assert result.split_counts["train"] == 1
        assert result.split_counts["val"] == 0
