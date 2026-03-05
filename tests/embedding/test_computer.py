"""Tests for EmbeddingComputer."""

import sys
import types

import numpy as np
import pytest


class FakeDataset:
    def __init__(self, fields=None, n_samples=10):
        self._fields = dict(fields or {})
        self._stored = {}
        self._n_samples = n_samples

    def values(self, field):
        return self._fields[field]

    def set_values(self, field, values):
        self._stored[field] = values

    def has_sample_field(self, field):
        return field in self._fields

    def compute_embeddings(self, model, embeddings_field=None, batch_size=None):
        """Mock compute_embeddings -- stores fake embeddings."""
        embs = np.random.default_rng(42).standard_normal(
            (self._n_samples, 16)
        ).tolist()
        if embeddings_field:
            self._fields[embeddings_field] = embs
            self._stored[embeddings_field] = embs
            return None
        return np.array(embs)

    def __len__(self):
        return self._n_samples


class FakeZooModel:
    has_embeddings = True


def test_compute_zoo_backend(monkeypatch):
    """Zoo backend calls compute_embeddings on the dataset."""
    import fiftyone.zoo

    from detection_toolkit.embedding import EmbeddingComputer

    def mock_load_zoo_model(name):
        return FakeZooModel()

    monkeypatch.setattr(fiftyone.zoo, "load_zoo_model", mock_load_zoo_model)

    ds = FakeDataset(n_samples=20)
    computer = EmbeddingComputer()
    computer.compute(
        ds,
        backend="zoo",
        model="dinov2-vitb14-torch",
        output_field="emb",
        batch_size=16,
    )

    assert "emb" in ds._stored


def test_compute_overwrite_protection():
    """Refuses to overwrite existing field unless overwrite=True."""
    from detection_toolkit.embedding import EmbeddingComputer

    ds = FakeDataset(fields={"existing": [None] * 10})
    computer = EmbeddingComputer()

    with pytest.raises(ValueError, match="already exists"):
        computer.compute(
            ds,
            backend="zoo",
            model="test",
            output_field="existing",
        )


def test_compute_overwrite_allowed(monkeypatch):
    """Overwrites existing field when overwrite=True."""
    import fiftyone.zoo

    from detection_toolkit.embedding import EmbeddingComputer

    def mock_load_zoo_model(name):
        return FakeZooModel()

    monkeypatch.setattr(fiftyone.zoo, "load_zoo_model", mock_load_zoo_model)

    ds = FakeDataset(fields={"existing": [None] * 10})
    computer = EmbeddingComputer()
    computer.compute(
        ds,
        backend="zoo",
        model="test",
        output_field="existing",
        overwrite=True,
    )
    assert "existing" in ds._stored


def test_compute_invalid_backend():
    """Rejects unknown backend."""
    from detection_toolkit.embedding import EmbeddingComputer

    ds = FakeDataset()
    computer = EmbeddingComputer()

    with pytest.raises(ValueError, match="Unknown.*backend"):
        computer.compute(ds, backend="bogus", model="x", output_field="e")
