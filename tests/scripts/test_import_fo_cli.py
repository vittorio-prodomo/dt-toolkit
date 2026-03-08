"""Tests for dt-import-fo CLI argument parsing."""


def test_disk_format_cvat():
    """--format cvat is accepted for disk subcommand."""
    from detection_toolkit.scripts.import_fo import parse_args

    args = parse_args(["disk", "/tmp/ds", "my_ds", "--format", "cvat"])
    assert args.format == "cvat"
    assert args.subcommand == "disk"


def test_disk_data_path():
    """--data-path is accepted for disk subcommand."""
    from detection_toolkit.scripts.import_fo import parse_args

    args = parse_args(["disk", "/tmp/ds", "my_ds", "--data-path", "/mnt/share"])
    assert args.data_path == "/mnt/share"


def test_disk_data_path_default_none():
    """--data-path defaults to None."""
    from detection_toolkit.scripts.import_fo import parse_args

    args = parse_args(["disk", "/tmp/ds", "my_ds"])
    assert args.data_path is None


def test_disk_format_choices():
    """--format only accepts yolo, coco, cvat."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "detection_toolkit.scripts.import_fo",
         "disk", "/tmp/ds", "my_ds", "--format", "invalid"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "invalid choice" in result.stderr
