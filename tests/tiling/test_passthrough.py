"""Tests for passthrough threshold logic."""

import shutil
from pathlib import Path

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
