#!/usr/bin/env python3
"""Ingest an on-disk dataset and print its metadata.

Detects the format (YOLO / COCO) and directory variant automatically,
then displays split layout, class names, and image counts.

Usage::

    python -m detection_toolkit.scripts.ingest datasets/my_dataset
    python -m detection_toolkit.scripts.ingest datasets/my_dataset --format yolo
"""

import argparse
import logging
import sys
from pathlib import Path

from detection_toolkit.cli import make_parser


def parse_args(argv=None):
    p = make_parser("Ingest a dataset and display its metadata.")
    p.add_argument("dataset", type=Path, help="Path to dataset root directory.")
    p.add_argument(
        "--format", "-f",
        choices=["yolo", "coco"],
        default=None,
        help="Force format (auto-detected if omitted).",
    )
    p.add_argument("--quiet", "-q", action="store_true", help="Suppress progress output.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(level=level, format="%(name)s - %(levelname)s - %(message)s")

    from detection_toolkit.core.loader import DatasetLoader
    from detection_toolkit.core.models import DatasetFormat

    fmt = DatasetFormat(args.format) if args.format else None
    loader = DatasetLoader(args.dataset, verbose=not args.quiet)
    info = loader.load_dataset(format=fmt)

    print(f"Format:       {info.format.value}")
    print(f"Task type:    {info.task_type.value}")
    print(f"Root:         {info.root_dir}")
    print(f"Classes ({info.num_classes}):")
    for cid in sorted(info.class_names):
        print(f"  {cid}: {info.class_names[cid]}")
    print(f"Total images: {info.total_images}")
    print(f"Splits:")
    for name, split in info.splits.items():
        lbl = f", {split.num_labels} labels" if split.label_paths else ""
        coco = f", coco_json={split.coco_ann_path.name}" if split.coco_ann_path else ""
        print(f"  {name}: {split.num_images} images{lbl}{coco}")


if __name__ == "__main__":
    main()
