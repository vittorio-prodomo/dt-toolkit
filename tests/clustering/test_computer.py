"""Tests for ClusterComputer."""

import numpy as np
import pytest


class FakeDataset:
    def __init__(self, fields):
        self._fields = dict(fields)
        self._stored = {}
    def values(self, field):
        return self._fields[field]
    def set_values(self, field, values):
        self._stored[field] = values
    def has_sample_field(self, field):
        return field in self._fields
    def __len__(self):
        return len(next(iter(self._fields.values())))


def test_cluster_hdbscan():
    """HDBSCAN assigns cluster labels."""
    from detection_toolkit.clustering import ClusterComputer
    rng = np.random.default_rng(42)
    cluster1 = rng.standard_normal((50, 8)) + 5
    cluster2 = rng.standard_normal((50, 8)) - 5
    embs = np.vstack([cluster1, cluster2]).tolist()
    ds = FakeDataset({"emb": embs})
    computer = ClusterComputer()
    labels = computer.cluster(
        ds, input_field="emb", output_field="cluster",
        method="hdbscan", min_cluster_size=10,
    )
    assert len(labels) == 100
    assert "cluster" in ds._stored
    assert len(ds._stored["cluster"]) == 100
    assert all(isinstance(l, str) for l in ds._stored["cluster"])


def test_cluster_kmeans():
    """K-Means assigns cluster labels."""
    from detection_toolkit.clustering import ClusterComputer
    rng = np.random.default_rng(42)
    embs = rng.standard_normal((60, 8)).tolist()
    ds = FakeDataset({"emb": embs})
    computer = ClusterComputer()
    labels = computer.cluster(
        ds, input_field="emb", output_field="cluster",
        method="kmeans", n_clusters=3,
    )
    assert len(labels) == 60
    unique = set(ds._stored["cluster"])
    assert len(unique) == 3


def test_cluster_dbscan():
    """DBSCAN assigns cluster labels."""
    from detection_toolkit.clustering import ClusterComputer
    rng = np.random.default_rng(42)
    cluster1 = rng.standard_normal((40, 4)) + 10
    cluster2 = rng.standard_normal((40, 4)) - 10
    embs = np.vstack([cluster1, cluster2]).tolist()
    ds = FakeDataset({"emb": embs})
    computer = ClusterComputer()
    labels = computer.cluster(
        ds, input_field="emb", output_field="cluster",
        method="dbscan", eps=2.0, min_samples=5,
    )
    assert len(labels) == 80
    assert "cluster" in ds._stored


def test_cluster_agglomerative():
    """Agglomerative assigns cluster labels."""
    from detection_toolkit.clustering import ClusterComputer
    rng = np.random.default_rng(42)
    embs = rng.standard_normal((50, 8)).tolist()
    ds = FakeDataset({"emb": embs})
    computer = ClusterComputer()
    labels = computer.cluster(
        ds, input_field="emb", output_field="cluster",
        method="agglomerative", n_clusters=4,
    )
    assert len(labels) == 50
    unique = set(ds._stored["cluster"])
    assert len(unique) == 4


def test_cluster_dry_run():
    """When output_field is None, no labels are saved."""
    from detection_toolkit.clustering import ClusterComputer
    rng = np.random.default_rng(42)
    embs = rng.standard_normal((40, 8)).tolist()
    ds = FakeDataset({"emb": embs})
    computer = ClusterComputer()
    labels = computer.cluster(
        ds, input_field="emb", output_field=None,
        method="kmeans", n_clusters=2,
    )
    assert len(labels) == 40
    assert len(ds._stored) == 0


def test_cluster_overwrite_protection():
    """Refuses to overwrite existing field."""
    from detection_toolkit.clustering import ClusterComputer
    embs = np.random.default_rng(42).standard_normal((30, 8)).tolist()
    ds = FakeDataset({"emb": embs, "existing": ["0"] * 30})
    computer = ClusterComputer()
    with pytest.raises(ValueError, match="already exists"):
        computer.cluster(ds, input_field="emb", output_field="existing",
                        method="kmeans", n_clusters=2)


def test_cluster_invalid_method():
    """Rejects unknown clustering method."""
    from detection_toolkit.clustering import ClusterComputer
    ds = FakeDataset({"emb": [[1, 2, 3]]})
    computer = ClusterComputer()
    with pytest.raises(ValueError, match="Unknown.*method"):
        computer.cluster(ds, input_field="emb", output_field="c", method="bogus")
