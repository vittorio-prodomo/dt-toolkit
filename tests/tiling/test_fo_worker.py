"""Tests for the FiftyOne tile worker function."""

from pathlib import Path

import numpy as np
import pyvips


def _create_test_image(path: Path, width: int, height: int, bands: int = 3):
    """Write a simple test JPEG image."""
    data = np.random.randint(0, 255, (height, width, bands), dtype=np.uint8)
    img = pyvips.Image.new_from_memory(data.data, width, height, bands, "uchar")
    path.parent.mkdir(parents=True, exist_ok=True)
    img.jpegsave(str(path), Q=90)


def test_tile_single_fo_sample_yolo_bbox(tmp_path):
    """Worker tiles an image with bbox annotations in YOLO format."""
    from detection_toolkit.tiling.config import TilingConfig
    from detection_toolkit.tiling.parallel import tile_single_fo_sample

    # Create a 200x200 test image
    img_path = tmp_path / "source" / "img_001.jpg"
    img_path.parent.mkdir(parents=True)
    _create_test_image(img_path, 200, 200)

    # Annotation: bbox covering most of the image
    annotations = [
        {"class_id": 0, "type": "bbox", "bbox": [10, 10, 80, 80]},
    ]

    output_dir = tmp_path / "output"
    cfg = TilingConfig(tile_size=128, overlap_percent=0.0, edge_behavior="crop")

    args = (
        str(img_path),
        annotations,
        str(output_dir),
        "train",
        "img_001",
        2,
        ["crack"],
        cfg.to_dict(),
        "yolo",
    )

    pos, neg, coco_entries = tile_single_fo_sample(args)

    # 200x200 with tile_size=128, crop -> 1 tile (partial second tile cropped)
    assert pos >= 0
    assert neg >= 0
    assert pos + neg >= 1
    assert coco_entries == []  # YOLO mode

    # Check output files exist
    img_out = output_dir / "images" / "train"
    lbl_out = output_dir / "labels" / "train"
    assert img_out.exists()
    assert lbl_out.exists()

    tile_images = list(img_out.glob("*.jpg"))
    tile_labels = list(lbl_out.glob("*.txt"))
    assert len(tile_images) >= 1
    assert len(tile_labels) >= 1


def test_tile_single_fo_sample_yolo_polygon(tmp_path):
    """Worker tiles an image with polygon annotations in YOLO format."""
    from detection_toolkit.tiling.config import TilingConfig
    from detection_toolkit.tiling.parallel import tile_single_fo_sample

    img_path = tmp_path / "source" / "img_002.jpg"
    img_path.parent.mkdir(parents=True)
    _create_test_image(img_path, 200, 200)

    annotations = [
        {
            "class_id": 0,
            "type": "polygon",
            "polygons": [[(20, 20), (80, 20), (80, 80), (20, 80)]],
        },
    ]

    output_dir = tmp_path / "output"
    cfg = TilingConfig(tile_size=128, overlap_percent=0.0, edge_behavior="crop")

    args = (
        str(img_path),
        annotations,
        str(output_dir),
        "train",
        "img_002",
        2,
        ["crack"],
        cfg.to_dict(),
        "yolo",
    )

    pos, neg, coco_entries = tile_single_fo_sample(args)
    assert pos + neg >= 1


def test_tile_single_fo_sample_coco(tmp_path):
    """Worker tiles an image and returns COCO annotation entries."""
    from detection_toolkit.tiling.config import TilingConfig
    from detection_toolkit.tiling.parallel import tile_single_fo_sample

    img_path = tmp_path / "source" / "img_003.jpg"
    img_path.parent.mkdir(parents=True)
    _create_test_image(img_path, 200, 200)

    annotations = [
        {"class_id": 0, "type": "bbox", "bbox": [10, 10, 80, 80]},
    ]

    output_dir = tmp_path / "output"
    cfg = TilingConfig(tile_size=128, overlap_percent=0.0, edge_behavior="crop")

    args = (
        str(img_path),
        annotations,
        str(output_dir),
        "train",
        "img_003",
        2,
        ["crack"],
        cfg.to_dict(),
        "coco",
    )

    pos, neg, coco_entries = tile_single_fo_sample(args)
    assert pos + neg >= 1
    # COCO mode returns entries with image_entry + annotation dicts
    if pos > 0:
        assert len(coco_entries) >= 1
        entry = coco_entries[0]
        assert "image" in entry
        assert "annotations" in entry


def test_tile_single_fo_sample_no_annotations(tmp_path):
    """Worker handles image with no annotations (all negative tiles)."""
    from detection_toolkit.tiling.config import TilingConfig
    from detection_toolkit.tiling.parallel import tile_single_fo_sample

    img_path = tmp_path / "source" / "img_004.jpg"
    img_path.parent.mkdir(parents=True)
    _create_test_image(img_path, 200, 200)

    output_dir = tmp_path / "output"
    cfg = TilingConfig(tile_size=128, overlap_percent=0.0, edge_behavior="crop")

    args = (
        str(img_path),
        [],  # no annotations
        str(output_dir),
        "train",
        "img_004",
        2,
        ["crack"],
        cfg.to_dict(),
        "yolo",
    )

    pos, neg, coco_entries = tile_single_fo_sample(args)
    assert pos == 0
    assert neg >= 1
