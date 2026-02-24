#!/usr/bin/env python3
"""Filter or merge classes within a dataset.

Usage::

    # Keep only specific classes
    python -m detection_toolkit.scripts.filter datasets/ds output/filtered \\
        --action keep --classes crack spall

    # Remove specific classes
    python -m detection_toolkit.scripts.filter datasets/ds output/filtered \\
        --action remove --classes efflorescence surface_defect

    # Merge variant class names into canonical names
    python -m detection_toolkit.scripts.filter datasets/ds output/merged \\
        --merge crack=mild_crack,splitting_crack,crack_map \\
        --merge spall=mechanical_spall,spall
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List

from detection_toolkit.cli import make_parser


def parse_args(argv=None):
    p = make_parser("Filter or merge classes in a dataset.")
    p.add_argument("input", type=Path, help="Input dataset directory.")
    p.add_argument("output", type=Path, help="Output dataset directory.")

    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--action", "-a",
        choices=["keep", "remove"],
        help="Filter action: keep or remove the listed classes.",
    )
    group.add_argument(
        "--merge", "-m",
        action="append",
        metavar="CANONICAL=var1,var2,...",
        help="Merge classes: canonical_name=variant1,variant2,...",
    )

    p.add_argument(
        "--classes", "-c",
        nargs="+",
        metavar="CLS",
        help="Class names for --action keep/remove.",
    )
    p.add_argument(
        "--format", "-f",
        choices=["yolo", "coco"],
        default=None,
        help="Force format (auto-detected if omitted).",
    )
    p.add_argument("--quiet", "-q", action="store_true")
    return p.parse_args(argv)


def _parse_merge_specs(specs: List[str]) -> Dict[str, List[str]]:
    """Parse ``canonical=var1,var2`` strings into a config dict."""
    config: Dict[str, List[str]] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(
                f"Invalid merge spec: {spec!r}. Expected canonical=var1,var2,..."
            )
        canonical, variants_str = spec.split("=", 1)
        variants = [v.strip() for v in variants_str.split(",") if v.strip()]
        config[canonical.strip()] = variants
    return config


def main(argv=None):
    args = parse_args(argv)

    level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(level=level, format="%(name)s - %(levelname)s - %(message)s")

    from detection_toolkit.core.models import DatasetFormat

    fmt = DatasetFormat(args.format) if args.format else None

    if args.merge:
        # Class merging mode
        from detection_toolkit.classes import ClassMerger

        config = _parse_merge_specs(args.merge)
        merger = ClassMerger(config, verbose=not args.quiet)
        result = merger.merge(args.input, args.output, format=fmt)
        print(f"Class merge complete:")
    else:
        # Class filter mode
        if not args.classes:
            print("Error: --classes is required with --action", file=sys.stderr)
            sys.exit(1)

        from detection_toolkit.classes import ClassFilter, ClassFilterConfig

        cfg = ClassFilterConfig(action=args.action, classes=args.classes)
        filt = ClassFilter(cfg, verbose=not args.quiet)
        result = filt.filter(args.input, args.output, format=fmt)
        print(f"Class filter ({args.action}) complete:")

    print(f"  Classes: {result.num_classes} -> {result.class_list}")
    print(f"  Images:  {result.total_images}")
    print(f"  Output:  {args.output}")


if __name__ == "__main__":
    main()
