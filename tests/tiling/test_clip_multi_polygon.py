"""Tests for multi-polygon handling during tile clipping."""

import pytest


class TestTilingConfigDisjointStrategy:
    """TilingConfig accepts and validates disjoint_strategy."""

    def test_default_is_bridge(self):
        from detection_toolkit.tiling.config import TilingConfig

        cfg = TilingConfig(tile_size=640)
        assert cfg.disjoint_strategy == "bridge"

    def test_accepts_split(self):
        from detection_toolkit.tiling.config import TilingConfig

        cfg = TilingConfig(tile_size=640, disjoint_strategy="split")
        assert cfg.disjoint_strategy == "split"

    def test_rejects_invalid(self):
        from detection_toolkit.tiling.config import TilingConfig

        with pytest.raises(ValueError, match="disjoint_strategy"):
            TilingConfig(tile_size=640, disjoint_strategy="invalid")

    def test_round_trips_through_dict(self):
        from detection_toolkit.tiling.config import TilingConfig

        cfg = TilingConfig(tile_size=640, disjoint_strategy="split")
        restored = TilingConfig.from_dict(cfg.to_dict())
        assert restored.disjoint_strategy == "split"
