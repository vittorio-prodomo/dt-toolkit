"""Tests for DatasetTiler.tile_fo() method.

These tests mock FiftyOne to avoid requiring a running MongoDB instance.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pyvips


def _create_test_image(path: Path, width: int, height: int):
    data = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    img = pyvips.Image.new_from_memory(data.data, width, height, 3, "uchar")
    path.parent.mkdir(parents=True, exist_ok=True)
    img.jpegsave(str(path), Q=90)


def _make_mock_dataset(tmp_path, num_samples=3, img_size=200):
    """Create a mock FiftyOne dataset with real images on disk."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    filepaths = []
    for i in range(num_samples):
        p = img_dir / f"img_{i:03d}.jpg"
        _create_test_image(p, img_size, img_size)
        filepaths.append(str(p))

    # Build mock dataset
    dataset = MagicMock()
    dataset.name = "test_dataset"
    dataset.classes = {"segmentations": ["crack", "spall"]}
    dataset.has_sample_field.return_value = True
    dataset.count.return_value = num_samples

    # values() returns bulk data
    def mock_values(fields, **kwargs):
        if isinstance(fields, list):
            result = []
            for f in fields:
                if f == "filepath":
                    result.append(filepaths)
                elif f == "metadata.width":
                    result.append([img_size] * num_samples)
                elif f == "metadata.height":
                    result.append([img_size] * num_samples)
                elif f == "split":
                    result.append(["train"] * num_samples)
                elif f.endswith(".detections") or f.endswith(".polylines"):
                    labels = []
                    for j in range(num_samples):
                        if j < num_samples - 1:  # last sample has no labels
                            labels.append({
                                "polylines": [{
                                    "label": "crack",
                                    "points": [[(0.1, 0.1), (0.5, 0.1), (0.5, 0.5), (0.1, 0.5)]],
                                }],
                            })
                        else:
                            labels.append(None)
                    result.append(labels)
                else:
                    result.append([None] * num_samples)
            return result
        return []

    dataset.values = mock_values

    # distinct("split") returns split names
    dataset.distinct = MagicMock(return_value=["train"])

    return dataset, filepaths


def test_tile_fo_yolo(tmp_path):
    """tile_fo() produces YOLO output with data.yaml and label files."""
    from detection_toolkit.tiling import TilingConfig, DatasetTiler

    dataset, _ = _make_mock_dataset(tmp_path)
    output_dir = tmp_path / "tiled"

    config = TilingConfig(tile_size=128, overlap_percent=0.0, edge_behavior="crop")
    tiler = DatasetTiler(config, verbose=False)

    with patch("detection_toolkit.fiftyone_hub.utils.resolve_label_field", return_value="segmentations"), \
         patch("detection_toolkit.fiftyone_hub.utils.detect_container_type", return_value="polylines"):
        result = tiler.tile_fo(
            dataset,
            output_dir,
            output_format="yolo",
            reimport=False,
        )

    # Check YOLO output structure
    assert (output_dir / "data.yaml").exists()
    assert (output_dir / "images" / "train").exists()
    assert (output_dir / "labels" / "train").exists()

    tile_images = list((output_dir / "images" / "train").glob("*.jpg"))
    tile_labels = list((output_dir / "labels" / "train").glob("*.txt"))
    assert len(tile_images) >= 1
    assert len(tile_labels) >= 1


def test_tile_fo_coco(tmp_path):
    """tile_fo() produces COCO output with annotations JSON."""
    from detection_toolkit.tiling import TilingConfig, DatasetTiler

    dataset, _ = _make_mock_dataset(tmp_path)
    output_dir = tmp_path / "tiled"

    config = TilingConfig(tile_size=128, overlap_percent=0.0, edge_behavior="crop")
    tiler = DatasetTiler(config, verbose=False)

    with patch("detection_toolkit.fiftyone_hub.utils.resolve_label_field", return_value="segmentations"), \
         patch("detection_toolkit.fiftyone_hub.utils.detect_container_type", return_value="polylines"):
        result = tiler.tile_fo(
            dataset,
            output_dir,
            output_format="coco",
            reimport=False,
        )

    # Check COCO output structure
    ann_dir = output_dir / "annotations"
    assert ann_dir.exists()
    ann_files = list(ann_dir.glob("instances_*.json"))
    assert len(ann_files) >= 1

    coco_data = json.loads(ann_files[0].read_text())
    assert "images" in coco_data
    assert "annotations" in coco_data
    assert "categories" in coco_data


def test_tile_fo_worker_pool_does_not_fork(tmp_path):
    """Regression: tiling workers must not be forked from the parent.

    libvips runs its work on a GLib thread pool. Once the parent has done any
    pyvips work, the pool's idle threads are registered in GLib's bookkeeping;
    a forked child inherits that bookkeeping but not the threads, so its first
    pyvips call hands work to a thread that does not exist and blocks forever
    in ``vips_threadpool_run``. (A fork after a pymongo/FiftyOne client exists
    is unsafe for the same reason.) This test does pyvips work in the parent
    (``_make_mock_dataset`` writes JPEGs) and asserts the pool's start method.
    """
    from concurrent.futures import ProcessPoolExecutor

    from detection_toolkit.tiling import TilingConfig, DatasetTiler

    dataset, _ = _make_mock_dataset(tmp_path)
    config = TilingConfig(
        tile_size=128, overlap_percent=0.0, edge_behavior="crop", max_processes=2,
    )
    tiler = DatasetTiler(config, verbose=False)

    start_methods = []

    def recording_pool(*args, **kwargs):
        ctx = kwargs.get("mp_context")
        start_methods.append(ctx.get_start_method() if ctx else None)
        return ProcessPoolExecutor(*args, **kwargs)

    with patch("detection_toolkit.fiftyone_hub.utils.resolve_label_field", return_value="segmentations"), \
         patch("detection_toolkit.fiftyone_hub.utils.detect_container_type", return_value="polylines"), \
         patch("detection_toolkit.tiling.tiler.ProcessPoolExecutor", side_effect=recording_pool):
        tiler.tile_fo(dataset, tmp_path / "tiled", output_format="coco", reimport=False)

    assert start_methods, "expected tile_fo to use a process pool"
    assert all(m not in (None, "fork") for m in start_methods), start_methods
