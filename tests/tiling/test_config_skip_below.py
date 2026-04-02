"""Tests for skip_below field in TilingConfig."""

from detection_toolkit.tiling.config import TilingConfig


def test_skip_below_default_is_none():
    cfg = TilingConfig(tile_size=416)
    assert cfg.skip_below is None


def test_skip_below_accepts_positive_int():
    cfg = TilingConfig(tile_size=416, skip_below=640)
    assert cfg.skip_below == 640


def test_skip_below_zero_raises():
    import pytest
    with pytest.raises(ValueError, match="skip_below must be > 0"):
        TilingConfig(tile_size=416, skip_below=0)


def test_skip_below_negative_raises():
    import pytest
    with pytest.raises(ValueError, match="skip_below must be > 0"):
        TilingConfig(tile_size=416, skip_below=-100)


def test_skip_below_roundtrips_through_dict():
    cfg = TilingConfig(tile_size=416, skip_below=640)
    d = cfg.to_dict()
    assert d["skip_below"] == 640
    cfg2 = TilingConfig.from_dict(d)
    assert cfg2.skip_below == 640


def test_skip_below_none_roundtrips_through_dict():
    cfg = TilingConfig(tile_size=416)
    d = cfg.to_dict()
    assert d["skip_below"] is None
    cfg2 = TilingConfig.from_dict(d)
    assert cfg2.skip_below is None


def test_skip_below_roundtrips_through_yaml(tmp_path):
    cfg = TilingConfig(tile_size=416, skip_below=640)
    yaml_path = tmp_path / "config.yaml"
    cfg.to_yaml(yaml_path)
    cfg2 = TilingConfig.from_yaml(yaml_path)
    assert cfg2.skip_below == 640
