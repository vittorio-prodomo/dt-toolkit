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
    p.add_argument("--weights-field", "-w", default=None,
                   metavar="FIELD",
                   help="Score field for loss weighting (writes weights.json per split).")
    p.add_argument("--merge-overlapping", action="store_true",
                   help="Merge overlapping same-class annotations before export.")
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

    logger = logging.getLogger(__name__)

    exporter = FiftyOneDatasetExporter()

    if args.merge_overlapping:
        from detection_toolkit.annotation_ops import merge_overlapping

        tmp_name = f"_tmp_merge_{dataset.name}"
        export_target = dataset.clone(tmp_name)
        logger.info("Cloned dataset for merge: %s", tmp_name)
        try:
            merge_overlapping(export_target, label_field=args.label_field)
            exporter.export(
                dataset=export_target,
                output_dir=args.output,
                format=fmt,
                export_media=args.export_media,
                splits=args.splits,
                label_field=args.label_field,
                min_area_ratio=args.min_area_ratio,
                max_area_ratio=args.max_area_ratio,
                negative_ratio=args.negative_ratio,
                exclude_tags=args.exclude_tags,
                weights_field=args.weights_field,
            )
        finally:
            fo.delete_dataset(tmp_name)
            logger.info("Deleted temporary dataset: %s", tmp_name)
    else:
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
            weights_field=args.weights_field,
        )

    print(f"FiftyOne export complete:")
    print(f"  Dataset:  {dataset.name}")
    print(f"  Format:   {fmt.value}")
    print(f"  Output:   {args.output}")


if __name__ == "__main__":
    main()
