"""Tests for weights.json export functionality."""

import json
from pathlib import Path

import numpy as np
import pytest


def test_write_weights_json(tmp_path):
    from detection_toolkit.fiftyone_hub.exporter import _write_weights_json

    filepaths = [
        "/data/images/train/img_001.jpg",
        "/data/images/train/img_002.jpg",
        "/data/images/train/img_003.jpg",
    ]
    scores = [0.3, 0.9, 0.6]
    output_dir = tmp_path / "images" / "train"
    output_dir.mkdir(parents=True)

    _write_weights_json(filepaths, scores, output_dir)

    weights_path = output_dir / "weights.json"
    assert weights_path.exists()

    weights = json.loads(weights_path.read_text())
    assert set(weights.keys()) == {"img_001.jpg", "img_002.jpg", "img_003.jpg"}

    # Normalized: min=0.3 -> 0.0, max=0.9 -> 1.0, mid=0.6 -> 0.5
    assert weights["img_001.jpg"] == pytest.approx(0.0)
    assert weights["img_002.jpg"] == pytest.approx(1.0)
    assert weights["img_003.jpg"] == pytest.approx(0.5)


def test_write_weights_json_constant_scores(tmp_path):
    """All-equal scores produce all 1.0 weights (avoid division by zero)."""
    from detection_toolkit.fiftyone_hub.exporter import _write_weights_json

    filepaths = ["/a/img1.jpg", "/a/img2.jpg"]
    scores = [0.5, 0.5]
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    _write_weights_json(filepaths, scores, output_dir)

    weights = json.loads((output_dir / "weights.json").read_text())
    assert weights["img1.jpg"] == pytest.approx(1.0)
    assert weights["img2.jpg"] == pytest.approx(1.0)
