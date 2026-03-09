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


# ── C-shape test polygon ───────────────────────────────────────────
# Opens to the right.  When clipped to the right-half tile
# (100, 0, 100, 200), the top and bottom arms become two disjoint
# sub-polygons.
C_SHAPE = [
    (10, 10), (150, 10), (150, 40), (60, 40),
    (60, 160), (150, 160), (150, 190), (10, 190),
]

# Right-half tile
TILE_X, TILE_Y, TILE_W, TILE_H = 100, 0, 100, 200


class TestClipAnnotationsMultiPolygon:
    """_clip_annotations preserves all sub-polygons."""

    def _clip(self, polygon, tx=TILE_X, ty=TILE_Y, tw=TILE_W, th=TILE_H):
        from detection_toolkit.tiling.config import TilingConfig
        from detection_toolkit.tiling.parallel import _clip_annotations

        ann = [{"class_id": 0, "type": "polygon", "polygon": polygon}]
        cfg = TilingConfig(
            tile_size=max(tw, th),
            min_area_ratio=0.0,
            max_area_ratio=1.0,
            densify_factor=0.0,
            smoothing_tolerance=0.0,
        )
        return _clip_annotations(ann, tx, ty, tw, th, cfg)

    def test_concave_produces_multiple_polygons(self):
        result = self._clip(C_SHAPE)
        assert len(result) == 1, "One source annotation → one output annotation"
        ann = result[0]
        assert ann["type"] == "polygon"
        assert "polygons" in ann, "Must use 'polygons' (plural) key"
        assert len(ann["polygons"]) == 2, "C-shape clipped to right tile → 2 sub-polygons"

    def test_convex_produces_single_polygon(self):
        """A simple rectangle that fits in the tile → single sub-polygon."""
        square = [(110, 50), (180, 50), (180, 150), (110, 150)]
        result = self._clip(square)
        assert len(result) == 1
        assert len(result[0]["polygons"]) == 1

    def test_area_filter_drops_small_subpolygons(self):
        """Sub-polygons below min_area_ratio are dropped individually."""
        from detection_toolkit.tiling.config import TilingConfig
        from detection_toolkit.tiling.parallel import _clip_annotations

        ann = [{"class_id": 0, "type": "polygon", "polygon": C_SHAPE}]
        # Set min_area_ratio high enough to drop the small arms
        cfg = TilingConfig(
            tile_size=200,
            min_area_ratio=0.15,
            max_area_ratio=1.0,
            densify_factor=0.0,
            smoothing_tolerance=0.0,
        )
        result = _clip_annotations(ann, TILE_X, TILE_Y, TILE_W, TILE_H, cfg)
        # Both arms are small (~1500 / 20000 = 0.075), so both should be dropped
        assert len(result) == 0

    def test_coordinates_are_tile_local(self):
        result = self._clip(C_SHAPE)
        for poly in result[0]["polygons"]:
            for x, y in poly:
                assert 0 <= x <= TILE_W, f"x={x} out of tile bounds"
                assert 0 <= y <= TILE_H, f"y={y} out of tile bounds"


class TestWriteYoloLabelMultiPolygon:
    """_write_yolo_label handles multi-polygon annotations."""

    def _make_multi_polygon_ann(self):
        """Two small disjoint rectangles in tile-local coords."""
        return {
            "class_id": 0,
            "type": "polygon",
            "polygons": [
                [(10, 10), (40, 10), (40, 30), (10, 30)],
                [(10, 70), (40, 70), (40, 90), (10, 90)],
            ],
        }

    def test_bridge_produces_single_line(self, tmp_path):
        from detection_toolkit.tiling.parallel import _write_yolo_label

        label_path = tmp_path / "tile.txt"
        ann = self._make_multi_polygon_ann()
        _write_yolo_label(label_path, [ann], 100, 100, disjoint_strategy="bridge")

        lines = [l for l in label_path.read_text().strip().split("\n") if l]
        assert len(lines) == 1, "bridge → single YOLO line"

    def test_split_produces_two_lines(self, tmp_path):
        from detection_toolkit.tiling.parallel import _write_yolo_label

        label_path = tmp_path / "tile.txt"
        ann = self._make_multi_polygon_ann()
        _write_yolo_label(label_path, [ann], 100, 100, disjoint_strategy="split")

        lines = [l for l in label_path.read_text().strip().split("\n") if l]
        assert len(lines) == 2, "split → one line per sub-polygon"
        # Both lines should have the same class_id
        assert all(l.startswith("0 ") for l in lines)

    def test_single_polygon_unchanged(self, tmp_path):
        from detection_toolkit.tiling.parallel import _write_yolo_label

        label_path = tmp_path / "tile.txt"
        ann = {
            "class_id": 1,
            "type": "polygon",
            "polygons": [[(10, 10), (90, 10), (90, 90), (10, 90)]],
        }
        _write_yolo_label(label_path, [ann], 100, 100, disjoint_strategy="bridge")

        lines = [l for l in label_path.read_text().strip().split("\n") if l]
        assert len(lines) == 1
        assert lines[0].startswith("1 ")
