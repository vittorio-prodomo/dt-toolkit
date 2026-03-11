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
