#!/usr/bin/env python3
"""Convert datasets between YOLO and COCO formats.

Usage::

    # COCO → YOLO
    python -m detection_toolkit.scripts.convert datasets/coco_ds output/yolo_ds --to yolo

    # YOLO → COCO
    python -m detection_toolkit.scripts.convert datasets/yolo_ds output/coco_ds --to coco
"""

import argparse
import logging
import sys
from pathlib import Path

from detection_toolkit.cli import make_parser


def parse_args(argv=None):
    p = make_parser("Convert a dataset between YOLO and COCO formats.")
    p.add_argument("input", type=Path, help="Input dataset directory.")
    p.add_argument("output", type=Path, help="Output dataset directory.")
    p.add_argument(
        "--to", "-t",
        required=True,
        choices=["yolo", "coco"],
        help="Target format.",
    )
    p.add_argument(
        "--contour-approx", "-A",
        type=float,
        default=0.001,
        metavar="F",
        help="Contour approximation factor for COCO→YOLO (default: 0.001).",
    )
    p.add_argument(
        "--hole-strategy", "-H",
        choices=["fill", "bridge"],
        default="bridge",
        help=(
            "How to handle holes in RLE masks (COCO→YOLO only). "
            "'fill' ignores holes; 'bridge' preserves them via "
            "inverse bridges (default: bridge)."
        ),
    )
    p.add_argument(
        "--disjoint-strategy", "-D",
        choices=["split", "bridge"],
        default="bridge",
        help=(
            "How to handle disjoint regions (COCO→YOLO only). "
            "'split' writes each as a separate annotation; "
            "'bridge' connects them with MST bridges (default: bridge)."
        ),
    )
    p.add_argument(
        "--keep-zero-indexing", "-z",
        action="store_true",
        help="Keep 0-based category IDs for YOLO→COCO (default: 1-based).",
    )
    p.add_argument("--quiet", "-q", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(level=level, format="%(name)s - %(levelname)s - %(message)s")

    if args.to == "yolo":
        from detection_toolkit.formats import COCOToYOLOConverter

        converter = COCOToYOLOConverter(verbose=not args.quiet)
        info = converter.convert(
            args.input, args.output,
            contour_approx_factor=args.contour_approx,
            hole_strategy=args.hole_strategy,
            disjoint_strategy=args.disjoint_strategy,
        )
    else:
        from detection_toolkit.formats import YOLOToCOCOConverter

        converter = YOLOToCOCOConverter(verbose=not args.quiet)
        info = converter.convert(
            args.input, args.output,
            keep_zero_indexing=args.keep_zero_indexing,
        )

    print(f"Conversion complete: {info.format.value}")
    print(f"  Classes: {info.num_classes}")
    print(f"  Images:  {info.total_images}")
    print(f"  Output:  {args.output}")


if __name__ == "__main__":
    main()
