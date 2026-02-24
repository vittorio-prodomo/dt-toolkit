#!/usr/bin/env python3
"""Export a FiftyOne dataset to YOLO or COCO format on disk.

Usage::

    python -m detection_toolkit.scripts.export_fo my_dataset_name output/exported --to yolo
    python -m detection_toolkit.scripts.export_fo my_dataset_name output/exported --to coco \\
        --splits train val --min-area-ratio 0.05
"""

import argparse
import logging
import sys
from pathlib import Path

from detection_toolkit.cli import make_parser


def parse_args(argv=None):
    p = make_parser("Export a FiftyOne dataset to YOLO or COCO format.")
    p.add_argument("name", type=str, help="FiftyOne dataset name.")
    p.add_argument("output", type=Path, help="Output directory.")
    p.add_argument(
        "--to", "-t",
        required=True,
        choices=["yolo", "coco"],
        help="Target format.",
    )
    p.add_argument(
        "--export-media", "-e",
        choices=["symlink", "copy", "move"],
        default="symlink",
        help="How to handle image files (default: symlink).",
    )
    p.add_argument("--splits", "-s", nargs="+", default=None,
                   metavar="S",
                   help="Export only these splits (default: all).")
    p.add_argument("--label-field", "-l", default=None,
                   metavar="FIELD",
                   help="FiftyOne label field to export.")
    p.add_argument("--min-area-ratio", "-m", type=float, default=0.0,
                   metavar="F",
                   help="Min annotation area ratio (default: 0.0).")
    p.add_argument("--max-area-ratio", "-M", type=float, default=1.0,
                   metavar="F",
                   help="Max annotation area ratio (default: 1.0).")
    p.add_argument("--negative-ratio", "-n", type=float, default=None,
                   metavar="F",
                   help="Negative sample ratio.")
    p.add_argument("--exclude-tags", "-x", nargs="+", default=None,
                   metavar="TAG",
                   help="Exclude samples with these tags.")
    p.add_argument("--quiet", "-q", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(level=level, format="%(name)s - %(levelname)s - %(message)s")

    import fiftyone as fo

    from detection_toolkit.fiftyone_hub import FiftyOneDatasetExporter
    from detection_toolkit.core.models import DatasetFormat

    dataset = fo.load_dataset(args.name)
    fmt = DatasetFormat(args.to)

    exporter = FiftyOneDatasetExporter(verbose=not args.quiet)
    exporter.export(
        dataset=dataset,
        output_dir=args.output,
        format=fmt,
        export_media=args.export_media,
        splits=args.splits,
        label_field=args.label_field,
        min_area_ratio=args.min_area_ratio,
        max_area_ratio=args.max_area_ratio,
        negative_ratio=args.negative_ratio,
        exclude_tags=args.exclude_tags,
    )

    print(f"FiftyOne export complete:")
    print(f"  Dataset:  {dataset.name}")
    print(f"  Format:   {fmt.value}")
    print(f"  Output:   {args.output}")


if __name__ == "__main__":
    main()
