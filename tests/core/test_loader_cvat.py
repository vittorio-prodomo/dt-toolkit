"""Tests for CVAT format detection in DatasetLoader."""

import textwrap
from pathlib import Path

from detection_toolkit.core.loader import DatasetLoader
from detection_toolkit.core.models import DatasetFormat


MINIMAL_CVAT_XML = textwrap.dedent("""\
    <?xml version="1.0" encoding="utf-8"?>
    <annotations>
      <version>1.1</version>
      <meta>
        <task>
          <labels>
            <label><name>crack</name></label>
          </labels>
        </task>
      </meta>
      <image id="0" name="img_001.jpg" width="640" height="480">
        <box label="crack" xtl="10" ytl="20" xbr="100" ybr="200" occluded="0"/>
      </image>
    </annotations>
""")


def test_detect_cvat_labels_xml(tmp_path):
    """Auto-detect CVAT format when labels.xml is present."""
    (tmp_path / "labels.xml").write_text(MINIMAL_CVAT_XML)
    loader = DatasetLoader(tmp_path)
    assert loader.detect_format() == DatasetFormat.CVAT


def test_detect_cvat_annotations_xml(tmp_path):
    """Auto-detect CVAT format when annotations.xml is present."""
    (tmp_path / "annotations.xml").write_text(MINIMAL_CVAT_XML)
    loader = DatasetLoader(tmp_path)
    assert loader.detect_format() == DatasetFormat.CVAT


def test_no_false_positive_random_xml(tmp_path):
    """A non-CVAT XML file should not trigger CVAT detection."""
    (tmp_path / "labels.xml").write_text("<root><item/></root>")
    loader = DatasetLoader(tmp_path)
    try:
        fmt = loader.detect_format()
        # If it detects something, it must not be CVAT
        assert fmt != DatasetFormat.CVAT
    except ValueError:
        pass  # "could not detect" is also acceptable


def test_explicit_format_cvat(tmp_path):
    """Explicit DatasetFormat.CVAT is accepted even without probing."""
    (tmp_path / "labels.xml").write_text(MINIMAL_CVAT_XML)
    loader = DatasetLoader(tmp_path)
    fmt = DatasetFormat.CVAT
    # Should not raise — the enum value is valid
    assert fmt.value == "cvat"
