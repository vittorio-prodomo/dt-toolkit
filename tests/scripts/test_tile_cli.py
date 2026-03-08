"""Tests for dt-tile CLI argument parsing."""

import pytest


def test_parse_fo_flag():
    """--fo flag sets fo attribute."""
    from detection_toolkit.scripts.tile import parse_args

    args = parse_args(["--fo", "my_dataset", "/tmp/output", "-t", "640"])
    assert args.fo == "my_dataset"
    assert args.output == "/tmp/output"
    assert args.tile_size == 640


def test_parse_fo_with_to():
    """--fo with --to sets output format."""
    from detection_toolkit.scripts.tile import parse_args

    args = parse_args(["--fo", "ds", "/tmp/out", "--to", "coco"])
    assert args.to == "coco"


def test_parse_fo_with_name():
    """--fo with --name sets dataset name."""
    from detection_toolkit.scripts.tile import parse_args

    args = parse_args(["--fo", "ds", "/tmp/out", "--name", "custom_name"])
    assert args.name == "custom_name"


def test_parse_fo_no_reimport():
    """--fo with --no-reimport disables reimport."""
    from detection_toolkit.scripts.tile import parse_args

    args = parse_args(["--fo", "ds", "/tmp/out", "--no-reimport"])
    assert args.no_reimport is True


def test_parse_disk_mode():
    """Positional input arg still works for disk mode."""
    from detection_toolkit.scripts.tile import parse_args

    args = parse_args(["/data/dataset", "/tmp/out", "-t", "512"])
    assert args.input == "/data/dataset"
    assert args.fo is None


def test_parse_fo_defaults():
    """--fo mode has sensible defaults."""
    from detection_toolkit.scripts.tile import parse_args

    args = parse_args(["--fo", "ds", "/tmp/out"])
    assert args.to == "yolo"
    assert args.name is None
    assert args.no_reimport is False
