"""Tests for DimensionalityReducer."""

import numpy as np
import pytest


class FakeDataset:
    """Minimal mock of a FiftyOne dataset for testing."""

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


def test_reduce_pca_basic():
    """PCA reduces 64d embeddings to 8d, stores result in output field."""
    from detection_toolkit.reduction import DimensionalityReducer

    rng = np.random.default_rng(42)
    embs = rng.standard_normal((50, 64)).tolist()
    ds = FakeDataset({"embeddings": embs})

    reducer = DimensionalityReducer()
    result = reducer.reduce(
        ds,
        input_field="embeddings",
        output_field="reduced",
        method="pca",
        n_components=8,
    )

    # Result is a numpy array with shape (50, 8)
    assert isinstance(result, np.ndarray)
    assert result.shape == (50, 8)

    # Output was stored on the dataset
    assert "reduced" in ds._stored
    stored = ds._stored["reduced"]
    assert len(stored) == 50

    # Each stored value should be a list of length 8
    for row in stored:
        assert len(row) == 8


def test_reduce_handles_none_embeddings():
    """Samples with None embeddings get None in output; valid samples are reduced."""
    from detection_toolkit.reduction import DimensionalityReducer

    rng = np.random.default_rng(42)
    embs = rng.standard_normal((10, 32)).tolist()
    # Insert None at indices 2 and 7
    embs[2] = None
    embs[7] = None
    ds = FakeDataset({"embeddings": embs})

    reducer = DimensionalityReducer()
    result = reducer.reduce(
        ds,
        input_field="embeddings",
        output_field="reduced",
        method="pca",
        n_components=4,
    )

    assert result.shape == (10, 4)
    # None rows should become NaN
    assert np.all(np.isnan(result[2]))
    assert np.all(np.isnan(result[7]))
    # Valid rows should not be NaN
    assert not np.any(np.isnan(result[0]))
    assert not np.any(np.isnan(result[1]))

    # Stored values: None entries stay None
    stored = ds._stored["reduced"]
    assert stored[2] is None
    assert stored[7] is None
    # Valid entries are lists
    assert isinstance(stored[0], list)
    assert len(stored[0]) == 4


def test_reduce_overwrite_protection():
    """Raises ValueError when output field exists and overwrite=False."""
    from detection_toolkit.reduction import DimensionalityReducer

    rng = np.random.default_rng(42)
    embs = rng.standard_normal((10, 16)).tolist()
    ds = FakeDataset({"embeddings": embs, "reduced": embs})

    reducer = DimensionalityReducer()
    with pytest.raises(ValueError, match="already exists"):
        reducer.reduce(
            ds,
            input_field="embeddings",
            output_field="reduced",
            method="pca",
            n_components=4,
            overwrite=False,
        )


def test_reduce_overwrite_allowed():
    """Works when overwrite=True even if output field already exists."""
    from detection_toolkit.reduction import DimensionalityReducer

    rng = np.random.default_rng(42)
    embs = rng.standard_normal((10, 16)).tolist()
    ds = FakeDataset({"embeddings": embs, "reduced": embs})

    reducer = DimensionalityReducer()
    result = reducer.reduce(
        ds,
        input_field="embeddings",
        output_field="reduced",
        method="pca",
        n_components=4,
        overwrite=True,
    )

    assert result.shape == (10, 4)
    assert "reduced" in ds._stored


def test_reduce_invalid_method():
    """Raises ValueError for an unknown reduction method."""
    from detection_toolkit.reduction import DimensionalityReducer

    rng = np.random.default_rng(42)
    embs = rng.standard_normal((10, 16)).tolist()
    ds = FakeDataset({"embeddings": embs})

    reducer = DimensionalityReducer()
    with pytest.raises(ValueError, match="Unknown.*method"):
        reducer.reduce(
            ds,
            input_field="embeddings",
            output_field="reduced",
            method="bogus_method",
            n_components=4,
        )


def test_reduce_umap():
    """UMAP reduces embeddings."""
    pytest.importorskip("umap")
    from detection_toolkit.reduction import DimensionalityReducer

    rng = np.random.default_rng(42)
    embs = rng.standard_normal((80, 32)).tolist()
    ds = FakeDataset({"emb": embs})

    reducer = DimensionalityReducer()
    result = reducer.reduce(
        ds, input_field="emb", output_field="emb_umap",
        method="umap", n_components=4, n_neighbors=10,
    )
    assert result.shape == (80, 4)
    assert len(ds._stored["emb_umap"]) == 80


def test_reduce_tsne():
    """t-SNE reduces embeddings to 2d."""
    from detection_toolkit.reduction import DimensionalityReducer

    rng = np.random.default_rng(42)
    embs = rng.standard_normal((80, 32)).tolist()
    ds = FakeDataset({"emb": embs})

    reducer = DimensionalityReducer()
    result = reducer.reduce(
        ds, input_field="emb", output_field="emb_tsne",
        method="tsne", n_components=2, perplexity=10,
    )
    assert result.shape == (80, 2)
    assert len(ds._stored["emb_tsne"]) == 80


def test_reduce_pacmap():
    """PaCMAP reduces embeddings."""
    pytest.importorskip("pacmap")
    from detection_toolkit.reduction import DimensionalityReducer

    rng = np.random.default_rng(42)
    embs = rng.standard_normal((80, 32)).tolist()
    ds = FakeDataset({"emb": embs})

    reducer = DimensionalityReducer()
    result = reducer.reduce(
        ds, input_field="emb", output_field="emb_pacmap",
        method="pacmap", n_components=4,
    )
    assert result.shape == (80, 4)
    assert len(ds._stored["emb_pacmap"]) == 80
