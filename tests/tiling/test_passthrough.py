"""Tests for passthrough threshold logic."""

from detection_toolkit.tiling.config import TilingConfig


def test_should_passthrough_below_tile_size():
    """Images smaller than tile_size always pass through."""
    from detection_toolkit.tiling.parallel import _should_passthrough

    cfg = TilingConfig(tile_size=416)
    assert _should_passthrough(400, 300, cfg) is True


def test_should_passthrough_equal_to_tile_size():
    """Image with max dim == tile_size passes through."""
    from detection_toolkit.tiling.parallel import _should_passthrough

    cfg = TilingConfig(tile_size=416)
    assert _should_passthrough(416, 300, cfg) is True


def test_should_not_passthrough_above_tile_size():
    """Images larger than tile_size are tiled normally."""
    from detection_toolkit.tiling.parallel import _should_passthrough

    cfg = TilingConfig(tile_size=416)
    assert _should_passthrough(800, 600, cfg) is False


def test_should_passthrough_with_skip_below():
    """--skip-below extends the threshold above tile_size."""
    from detection_toolkit.tiling.parallel import _should_passthrough

    cfg = TilingConfig(tile_size=416, skip_below=640)
    # 500px > tile_size but <= skip_below -> passthrough
    assert _should_passthrough(500, 400, cfg) is True
    # 640px == skip_below -> passthrough
    assert _should_passthrough(640, 480, cfg) is True
    # 641px > skip_below -> tile
    assert _should_passthrough(641, 480, cfg) is False


def test_should_passthrough_height_dominant():
    """Threshold applies to max(w, h), not just width."""
    from detection_toolkit.tiling.parallel import _should_passthrough

    cfg = TilingConfig(tile_size=416, skip_below=640)
    # Width is small but height exceeds threshold
    assert _should_passthrough(200, 800, cfg) is False
    # Height is the max dim but still below threshold
    assert _should_passthrough(200, 600, cfg) is True


def test_passthrough_image_copies_file(tmp_path):
    """_passthrough_image does a file copy, not re-encode."""
    from detection_toolkit.tiling.parallel import _passthrough_image

    # Create a fake image file with known content
    src = tmp_path / "src" / "image.jpg"
    src.parent.mkdir()
    src.write_bytes(b"FAKE_JPEG_CONTENT_12345")

    dst = tmp_path / "dst" / "image.jpg"
    dst.parent.mkdir()

    _passthrough_image(src, dst)

    assert dst.exists()
    assert dst.read_bytes() == b"FAKE_JPEG_CONTENT_12345"


def _create_yolo_fixture(tmp_path, img_w=200, img_h=150):
    """Create a minimal YOLO image + label for testing.

    Returns (img_path, label_path, output_path).
    """
    import pyvips

    img = pyvips.Image.black(img_w, img_h, bands=3)
    img_path = tmp_path / "images" / "train" / "small.jpg"
    img_path.parent.mkdir(parents=True)
    img.jpegsave(str(img_path), Q=90)

    # YOLO label: one bbox annotation (class 0, centered, 50% of image)
    label_path = tmp_path / "labels" / "train" / "small.txt"
    label_path.parent.mkdir(parents=True)
    label_path.write_text("0 0.5 0.5 0.5 0.5\n")

    output_path = tmp_path / "output"
    output_path.mkdir()
    (output_path / "images" / "train").mkdir(parents=True)
    (output_path / "labels" / "train").mkdir(parents=True)

    return img_path, label_path, output_path


def test_yolo_worker_passthrough_copies_image(tmp_path):
    """YOLO worker passes through small images without re-encoding."""
    from detection_toolkit.tiling.parallel import tile_single_yolo_image

    img_path, label_path, output_path = _create_yolo_fixture(tmp_path, 200, 150)
    original_bytes = img_path.read_bytes()

    cfg = TilingConfig(tile_size=416)  # 200x150 < 416 -> passthrough
    args = (
        str(img_path),
        str(label_path),
        str(output_path),
        "train",
        3,  # pad_width
        cfg.to_dict(),
    )
    pos, neg = tile_single_yolo_image(args)

    assert pos == 1
    assert neg == 0

    # Image should be bit-identical copy (no re-encode)
    out_img = output_path / "images" / "train" / "small.jpg"
    assert out_img.exists()
    assert out_img.read_bytes() == original_bytes

    # Label should be bit-identical copy
    out_lbl = output_path / "labels" / "train" / "small.txt"
    assert out_lbl.exists()
    assert out_lbl.read_text() == "0 0.5 0.5 0.5 0.5\n"


def test_yolo_worker_passthrough_no_label(tmp_path):
    """YOLO worker handles passthrough of images without label files."""
    from detection_toolkit.tiling.parallel import tile_single_yolo_image

    img_path, _, output_path = _create_yolo_fixture(tmp_path, 200, 150)

    cfg = TilingConfig(tile_size=416)
    args = (
        str(img_path),
        None,  # no label file
        str(output_path),
        "train",
        3,
        cfg.to_dict(),
    )
    pos, neg = tile_single_yolo_image(args)

    assert pos == 0
    assert neg == 1

    # Image should exist
    out_img = output_path / "images" / "train" / "small.jpg"
    assert out_img.exists()

    # Empty label for negative
    out_lbl = output_path / "labels" / "train" / "small.txt"
    assert out_lbl.exists()
    assert out_lbl.read_text() == ""


def test_unlabeled_worker_passthrough(tmp_path):
    """Unlabeled worker passes through small images."""
    from detection_toolkit.tiling.parallel import tile_single_image

    import pyvips
    img_path = tmp_path / "src" / "photo.jpg"
    img_path.parent.mkdir()
    img = pyvips.Image.black(300, 200, bands=3)
    img.jpegsave(str(img_path), Q=90)
    original_bytes = img_path.read_bytes()

    output_dir = tmp_path / "dst"
    output_dir.mkdir()

    cfg = TilingConfig(tile_size=416)  # 300x200 < 416 -> passthrough
    args = (
        str(img_path),
        str(output_dir),
        "photo",   # tile_prefix
        3,         # pad_width
        cfg.to_dict(),
    )
    count = tile_single_image(args)

    assert count == 1
    out_img = output_dir / "photo.jpg"
    assert out_img.exists()
    assert out_img.read_bytes() == original_bytes


def test_fo_worker_passthrough_yolo(tmp_path):
    """FiftyOne worker passes through small images in YOLO mode."""
    from detection_toolkit.tiling.parallel import tile_single_fo_sample

    import pyvips
    img_path = tmp_path / "src" / "small.jpg"
    img_path.parent.mkdir()
    img = pyvips.Image.black(200, 150, bands=3)
    img.jpegsave(str(img_path), Q=90)
    original_bytes = img_path.read_bytes()

    output_dir = tmp_path / "output"

    # One polygon annotation: class 0, simple square
    annotations = [{
        "class_id": 0,
        "type": "polygon",
        "polygons": [[(10, 10), (50, 10), (50, 50), (10, 50)]],
    }]

    cfg = TilingConfig(tile_size=416)  # 200x150 < 416 -> passthrough
    args = (
        str(img_path),
        annotations,
        str(output_dir),
        "train",
        "small",     # tile_prefix
        3,           # pad_width
        ["defect"],  # class_list
        cfg.to_dict(),
        "yolo",      # output_format
    )
    pos, neg, coco_entries = tile_single_fo_sample(args)

    assert pos == 1
    assert neg == 0
    assert coco_entries == []

    # Image should be bit-identical
    out_img = output_dir / "images" / "train" / "small.jpg"
    assert out_img.exists()
    assert out_img.read_bytes() == original_bytes

    # Label file should exist with the annotation
    out_lbl = output_dir / "labels" / "train" / "small.txt"
    assert out_lbl.exists()
    assert out_lbl.read_text().strip() != ""


def test_fo_worker_passthrough_coco(tmp_path):
    """FiftyOne worker passes through small images in COCO mode."""
    from detection_toolkit.tiling.parallel import tile_single_fo_sample

    import pyvips
    img_path = tmp_path / "src" / "small.jpg"
    img_path.parent.mkdir()
    img = pyvips.Image.black(200, 150, bands=3)
    img.jpegsave(str(img_path), Q=90)

    output_dir = tmp_path / "output"

    annotations = [{
        "class_id": 0,
        "type": "polygon",
        "polygons": [[(10, 10), (50, 10), (50, 50), (10, 50)]],
    }]

    cfg = TilingConfig(tile_size=416)
    args = (
        str(img_path),
        annotations,
        str(output_dir),
        "train",
        "small",
        3,
        ["defect"],
        cfg.to_dict(),
        "coco",
    )
    pos, neg, coco_entries = tile_single_fo_sample(args)

    assert pos == 1
    assert neg == 0
    assert len(coco_entries) == 1

    entry = coco_entries[0]
    assert entry["image"]["width"] == 200
    assert entry["image"]["height"] == 150
    assert entry["image"]["file_name"] == "small.jpg"
    assert len(entry["annotations"]) == 1
