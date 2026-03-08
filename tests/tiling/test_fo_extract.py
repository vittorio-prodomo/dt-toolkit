"""Tests for FiftyOne annotation extraction."""

import pytest


def test_extract_detections_bbox():
    """Detections with bbox-only are extracted as bbox dicts."""
    from detection_toolkit.tiling.fo_extract import extract_annotations

    # Simulate FiftyOne bulk values() output for Detections field.
    # Each element is a fo.Detections dict-like with .detections list.
    # We pass the raw list of Detection-like dicts as returned by values().
    sample_labels = {
        "detections": [
            {"label": "crack", "bounding_box": [0.1, 0.2, 0.3, 0.4]},
            {"label": "spall", "bounding_box": [0.5, 0.5, 0.2, 0.3]},
        ]
    }
    class_map = {"crack": 0, "spall": 1}

    result = extract_annotations(
        sample_labels,
        container_type="detections",
        class_map=class_map,
        img_w=1000,
        img_h=800,
    )

    assert len(result) == 2
    assert result[0]["class_id"] == 0
    assert result[0]["type"] == "bbox"
    # bounding_box [0.1, 0.2, 0.3, 0.4] -> abs [100, 160, 300, 320]
    assert result[0]["bbox"] == pytest.approx([100.0, 160.0, 300.0, 320.0])

    assert result[1]["class_id"] == 1
    assert result[1]["bbox"] == pytest.approx([500.0, 400.0, 200.0, 240.0])


def test_extract_polylines():
    """Polylines are extracted as polygon dicts with absolute coords."""
    from detection_toolkit.tiling.fo_extract import extract_annotations

    sample_labels = {
        "polylines": [
            {
                "label": "crack",
                "points": [[(0.1, 0.2), (0.3, 0.4), (0.5, 0.2)]],
            },
        ]
    }
    class_map = {"crack": 0}

    result = extract_annotations(
        sample_labels,
        container_type="polylines",
        class_map=class_map,
        img_w=1000,
        img_h=500,
    )

    assert len(result) == 1
    assert result[0]["class_id"] == 0
    assert result[0]["type"] == "polygon"
    expected_points = [(100.0, 100.0), (300.0, 200.0), (500.0, 100.0)]
    assert result[0]["polygon"] == pytest.approx(expected_points)


def test_extract_empty_labels():
    """None or empty label containers return empty list."""
    from detection_toolkit.tiling.fo_extract import extract_annotations

    result = extract_annotations(
        None,
        container_type="detections",
        class_map={"a": 0},
        img_w=100,
        img_h=100,
    )
    assert result == []


def test_extract_unknown_class_skipped():
    """Annotations with labels not in class_map are skipped."""
    from detection_toolkit.tiling.fo_extract import extract_annotations

    sample_labels = {
        "detections": [
            {"label": "unknown_class", "bounding_box": [0.1, 0.2, 0.3, 0.4]},
        ]
    }
    class_map = {"crack": 0}

    result = extract_annotations(
        sample_labels,
        container_type="detections",
        class_map=class_map,
        img_w=100,
        img_h=100,
    )
    assert result == []


def test_build_class_map():
    """Build class name -> ID mapping from FiftyOne dataset classes dict."""
    from detection_toolkit.tiling.fo_extract import build_class_map

    # FiftyOne stores classes as {field_name: [class_list]}
    classes_dict = {"segmentations": ["crack", "spall", "rebar"]}
    result = build_class_map(classes_dict)

    assert result == {"crack": 0, "spall": 1, "rebar": 2}


def test_build_class_map_empty():
    """Empty or missing classes dict returns empty map."""
    from detection_toolkit.tiling.fo_extract import build_class_map

    assert build_class_map({}) == {}
    assert build_class_map(None) == {}
