"""Tests for FiftyOneDatasetExporter media overlap detection."""

from pathlib import Path
from unittest.mock import MagicMock

from detection_toolkit.fiftyone_hub.exporter import FiftyOneDatasetExporter


class TestShouldSkipMedia:
    """Tests for _should_skip_media overlap detection."""

    def _make_split_view(self, filepaths):
        """Build a mock split view returning the given filepaths."""
        view = MagicMock()
        view.values = MagicMock(return_value=filepaths)
        return view

    def test_all_overlap_returns_true(self, tmp_path):
        """When all filepaths are inside target dir, returns True."""
        target = tmp_path / "images" / "train"
        target.mkdir(parents=True)
        # Create real files so resolve() works
        for name in ("a.jpg", "b.jpg", "c.jpg"):
            (target / name).touch()

        filepaths = [str(target / name) for name in ("a.jpg", "b.jpg", "c.jpg")]
        view = self._make_split_view(filepaths)

        exporter = FiftyOneDatasetExporter()
        assert exporter._should_skip_media(view, target) is True

    def test_no_overlap_returns_false(self, tmp_path):
        """When no filepaths are inside target dir, returns False."""
        source = tmp_path / "source"
        source.mkdir()
        target = tmp_path / "images" / "train"
        target.mkdir(parents=True)

        for name in ("a.jpg", "b.jpg"):
            (source / name).touch()

        filepaths = [str(source / name) for name in ("a.jpg", "b.jpg")]
        view = self._make_split_view(filepaths)

        exporter = FiftyOneDatasetExporter()
        assert exporter._should_skip_media(view, target) is False

    def test_partial_overlap_returns_false(self, tmp_path):
        """When some but not all filepaths overlap, returns False."""
        source = tmp_path / "source"
        source.mkdir()
        target = tmp_path / "images" / "train"
        target.mkdir(parents=True)

        (target / "a.jpg").touch()
        (source / "b.jpg").touch()

        filepaths = [str(target / "a.jpg"), str(source / "b.jpg")]
        view = self._make_split_view(filepaths)

        exporter = FiftyOneDatasetExporter()
        assert exporter._should_skip_media(view, target) is False

    def test_empty_filepaths_returns_false(self):
        """Empty filepath list returns False."""
        view = self._make_split_view([])

        exporter = FiftyOneDatasetExporter()
        assert exporter._should_skip_media(view, Path("/nonexistent")) is False

    def test_symlinks_resolved(self, tmp_path):
        """Symlinks pointing into target dir are detected as overlap."""
        target = tmp_path / "images" / "train"
        target.mkdir(parents=True)
        (target / "real.jpg").touch()

        link_dir = tmp_path / "links"
        link_dir.mkdir()
        link = link_dir / "link.jpg"
        link.symlink_to(target / "real.jpg")

        filepaths = [str(link)]
        view = self._make_split_view(filepaths)

        exporter = FiftyOneDatasetExporter()
        assert exporter._should_skip_media(view, target) is True

    def test_path_prefix_false_positive(self, tmp_path):
        """A dir named 'train' must not match 'training'."""
        train_dir = tmp_path / "images" / "train"
        train_dir.mkdir(parents=True)
        training_dir = tmp_path / "images" / "training"
        training_dir.mkdir(parents=True)
        (training_dir / "a.jpg").touch()

        filepaths = [str(training_dir / "a.jpg")]
        view = self._make_split_view(filepaths)

        exporter = FiftyOneDatasetExporter()
        assert exporter._should_skip_media(view, train_dir) is False

    def test_target_dir_does_not_exist(self, tmp_path):
        """When target dir doesn't exist yet, returns False."""
        source = tmp_path / "source"
        source.mkdir()
        (source / "a.jpg").touch()

        filepaths = [str(source / "a.jpg")]
        view = self._make_split_view(filepaths)

        exporter = FiftyOneDatasetExporter()
        assert exporter._should_skip_media(view, tmp_path / "nonexistent") is False
