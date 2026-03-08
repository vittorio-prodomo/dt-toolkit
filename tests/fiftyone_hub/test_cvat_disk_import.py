"""Tests for CVAT Image disk import into FiftyOne."""

import textwrap
from pathlib import Path


def _make_cvat_dataset(tmp_path, subset=None):
    """Create a minimal CVAT Image dataset on disk.

    Returns (dataset_dir, labels_path, image_path).
    """
    dataset_dir = tmp_path / "cvat_ds"
    dataset_dir.mkdir()

    subset_attr = f' subset="{subset}"' if subset else ""
    img_name = "img_001.jpg"
    xml = textwrap.dedent(f"""\
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
          <image id="0" name="{img_name}" width="4" height="4"{subset_attr}>
            <box label="crack" xtl="0" ytl="0" xbr="2" ybr="2" occluded="0"/>
          </image>
        </annotations>
    """)
    (dataset_dir / "labels.xml").write_text(xml)

    # Create image in data/ (or data/{subset}/ if subset)
    if subset:
        img_dir = dataset_dir / "data" / subset
    else:
        img_dir = dataset_dir / "data"
    img_dir.mkdir(parents=True)

    from PIL import Image

    img = Image.new("RGB", (4, 4), color="red")
    img_path = img_dir / img_name
    img.save(img_path)

    return dataset_dir, dataset_dir / "labels.xml", img_path


def test_load_cvat_image_basic(tmp_path):
    """Import a CVAT Image dataset from disk."""
    dataset_dir, _, _ = _make_cvat_dataset(tmp_path)

    from detection_toolkit.fiftyone_hub.loader import FiftyOneDatasetLoader

    loader = FiftyOneDatasetLoader(verbose=False)
    dataset = loader.load(
        dataset_dir,
        "test_cvat_basic",
        persistent=False,
        overwrite=True,
    )
    try:
        assert len(dataset) == 1
        sample = dataset.first()
        assert sample["detections"] is not None
        assert len(sample["detections"].detections) == 1
        assert sample["detections"].detections[0].label == "crack"
    finally:
        dataset.delete()


def test_load_cvat_image_with_subset(tmp_path):
    """Import a CVAT Image dataset with subset -> split field."""
    dataset_dir, _, _ = _make_cvat_dataset(tmp_path, subset="train")

    from detection_toolkit.fiftyone_hub.loader import FiftyOneDatasetLoader

    loader = FiftyOneDatasetLoader(verbose=False)
    dataset = loader.load(
        dataset_dir,
        "test_cvat_subset",
        persistent=False,
        overwrite=True,
    )
    try:
        assert len(dataset) == 1
        sample = dataset.first()
        assert sample["split"] == "train"
    finally:
        dataset.delete()


def test_load_cvat_with_data_path(tmp_path):
    """Import CVAT Image dataset with external data_path."""
    xml_dir = tmp_path / "xml_only"
    xml_dir.mkdir()
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    xml = textwrap.dedent("""\
        <?xml version="1.0" encoding="utf-8"?>
        <annotations>
          <version>1.1</version>
          <meta><task><labels>
            <label><name>crack</name></label>
          </labels></task></meta>
          <image id="0" name="img.jpg" width="4" height="4">
            <box label="crack" xtl="0" ytl="0" xbr="2" ybr="2" occluded="0"/>
          </image>
        </annotations>
    """)
    (xml_dir / "labels.xml").write_text(xml)

    from PIL import Image

    img = Image.new("RGB", (4, 4), color="blue")
    img.save(img_dir / "img.jpg")

    from detection_toolkit.fiftyone_hub.loader import FiftyOneDatasetLoader

    loader = FiftyOneDatasetLoader(verbose=False)
    dataset = loader.load(
        xml_dir,
        "test_cvat_data_path",
        persistent=False,
        overwrite=True,
        data_path=str(img_dir),
    )
    try:
        assert len(dataset) == 1
    finally:
        dataset.delete()


def test_load_cvat_no_subset_no_split(tmp_path):
    """Without subset attributes, no split field is added."""
    dataset_dir, _, _ = _make_cvat_dataset(tmp_path)

    from detection_toolkit.fiftyone_hub.loader import FiftyOneDatasetLoader

    loader = FiftyOneDatasetLoader(verbose=False)
    dataset = loader.load(
        dataset_dir,
        "test_cvat_no_split",
        persistent=False,
        overwrite=True,
    )
    try:
        assert not dataset.has_sample_field("split")
    finally:
        dataset.delete()
