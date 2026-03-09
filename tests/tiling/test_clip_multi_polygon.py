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

        ann = [{"class_id": 0, "type": "polygon", "polygons": [polygon]}]
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

        ann = [{"class_id": 0, "type": "polygon", "polygons": [C_SHAPE]}]
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


class TestToCOCOAnnotationsMultiPolygon:
    """_to_coco_annotations groups sub-polygons in segmentation list."""

    def test_multi_polygon_in_segmentation(self):
        from detection_toolkit.tiling.parallel import _to_coco_annotations

        ann = {
            "class_id": 0,
            "type": "polygon",
            "polygons": [
                [(10, 10), (40, 10), (40, 30), (10, 30)],
                [(10, 70), (40, 70), (40, 90), (10, 90)],
            ],
        }
        result = _to_coco_annotations([ann], 100, 100, ["crack"])
        assert len(result) == 1, "Multi-polygon → one COCO annotation"
        seg = result[0]["segmentation"]
        assert len(seg) == 2, "segmentation has two polygon lists"
        assert result[0]["category_id"] == 1  # COCO is 1-based

    def test_single_polygon_in_segmentation(self):
        from detection_toolkit.tiling.parallel import _to_coco_annotations

        ann = {
            "class_id": 0,
            "type": "polygon",
            "polygons": [[(10, 10), (90, 10), (90, 90), (10, 90)]],
        }
        result = _to_coco_annotations([ann], 100, 100, ["crack"])
        assert len(result) == 1
        assert len(result[0]["segmentation"]) == 1

    def test_bbox_covers_all_subpolygons(self):
        from detection_toolkit.tiling.parallel import _to_coco_annotations

        ann = {
            "class_id": 0,
            "type": "polygon",
            "polygons": [
                [(10, 10), (40, 10), (40, 30), (10, 30)],
                [(10, 70), (40, 70), (40, 90), (10, 90)],
            ],
        }
        result = _to_coco_annotations([ann], 100, 100, ["crack"])
        bbox = result[0]["bbox"]
        # bbox should span from (10,10) to (40,90) → [10, 10, 30, 80]
        assert bbox[0] == 10 and bbox[1] == 10
        assert bbox[2] == 30 and bbox[3] == 80

    def test_area_sums_all_subpolygons(self):
        from detection_toolkit.tiling.parallel import _to_coco_annotations

        ann = {
            "class_id": 0,
            "type": "polygon",
            "polygons": [
                [(0, 0), (10, 0), (10, 10), (0, 10)],    # area = 100
                [(20, 0), (30, 0), (30, 10), (20, 10)],   # area = 100
            ],
        }
        result = _to_coco_annotations([ann], 100, 100, ["crack"])
        assert result[0]["area"] == pytest.approx(200.0)


class TestClipCOCOAnnotationsMultiPolygon:
    """_clip_coco_annotations groups sub-polygons under one annotation."""

    def _make_tiler(self):
        from detection_toolkit.tiling.config import TilingConfig
        from detection_toolkit.tiling.tiler import DatasetTiler

        cfg = TilingConfig(
            tile_size=200,
            min_area_ratio=0.0,
            max_area_ratio=1.0,
            densify_factor=0.0,
            smoothing_tolerance=0.0,
        )
        return DatasetTiler(cfg, verbose=False)

    def test_concave_polygon_stays_one_annotation(self):
        tiler = self._make_tiler()

        # C-shape as flat COCO polygon
        flat_c = [
            10, 10, 150, 10, 150, 40, 60, 40,
            60, 160, 150, 160, 150, 190, 10, 190,
        ]
        ann = {
            "category_id": 1,
            "segmentation": [flat_c],
            "bbox": [10, 10, 140, 180],
            "area": 100,
            "iscrowd": 0,
        }

        result = tiler._clip_coco_annotations(
            [ann], tx=100, ty=0, tw=100, th=200, img_w=200, img_h=200,
        )
        assert len(result) == 1, "One source annotation → one output annotation"
        seg = result[0]["segmentation"]
        assert isinstance(seg, list)
        assert len(seg) == 2, "Two sub-polygons in segmentation list"
        assert result[0]["category_id"] == 1

    def test_already_multi_polygon_stays_one_annotation(self):
        """COCO annotation that already has two polygons in segmentation."""
        tiler = self._make_tiler()

        # Two separate horizontal bars fully inside the tile
        poly1 = [110, 10, 180, 10, 180, 40, 110, 40]
        poly2 = [110, 160, 180, 160, 180, 190, 110, 190]
        ann = {
            "category_id": 2,
            "segmentation": [poly1, poly2],
            "bbox": [110, 10, 70, 180],
            "area": 200,
            "iscrowd": 0,
        }

        result = tiler._clip_coco_annotations(
            [ann], tx=100, ty=0, tw=100, th=200, img_w=200, img_h=200,
        )
        assert len(result) == 1, "Still one annotation"
        # Each source polygon clips cleanly → 2 sub-polygons
        assert len(result[0]["segmentation"]) >= 2


class TestCLIDisjointStrategy:
    """CLI --disjoint-strategy argument."""

    def test_default_is_bridge(self):
        from detection_toolkit.scripts.tile import parse_args

        args = parse_args(["some_input", "some_output"])
        assert args.disjoint_strategy == "bridge"

    def test_accepts_split(self):
        from detection_toolkit.scripts.tile import parse_args

        args = parse_args([
            "some_input", "some_output", "--disjoint-strategy", "split",
        ])
        assert args.disjoint_strategy == "split"

    def test_rejects_invalid(self):
        from detection_toolkit.scripts.tile import parse_args

        with pytest.raises(SystemExit):
            parse_args([
                "some_input", "some_output", "--disjoint-strategy", "nope",
            ])
