#!/usr/bin/env python3
"""Merge multiple datasets into one with class remapping.

Usage::

    # Merge two YOLO datasets with a mapping config
    python -m detection_toolkit.scripts.merge output/merged \\
        --datasets datasets/ds_a datasets/ds_b \\
        --config merge_config.yaml

    # Merge with inline mapping
    python -m detection_toolkit.scripts.merge output/merged \\
        --datasets datasets/ds_a datasets/ds_b \\
        --map crack=mild_crack,splitting_crack \\
        --map spall=mechanical_spall,spall
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List

from detection_toolkit.cli import make_parser


def parse_args(argv=None):
    p = make_parser("Merge multiple datasets with class remapping.")
    p.add_argument("output", type=Path, help="Output dataset directory.")
    p.add_argument(
        "--datasets", "-d",
        nargs="+",
        type=Path,
        required=True,
        metavar="DIR",
        help="Input dataset directories.",
    )
    p.add_argument(
        "--config", "-c",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to a MergeConfig YAML file.",
    )
    p.add_argument(
        "--map", "-m",
        action="append",
        metavar="CANONICAL=var1,var2,...",
        help="Inline label mapping (repeatable).",
    )
    p.add_argument(
        "--missing-splits", "-M",
        choices=["include_all", "common_only", "error"],
        default="include_all",
    )
    p.add_argument(
        "--image-conflicts", "-I",
        choices=["prefix_dataset", "error"],
        default="prefix_dataset",
    )
    p.add_argument(
        "--unmapped-classes", "-u",
        choices=["include", "skip", "error"],
        default="include",
    )
    p.add_argument(
        "--output-splits", "-O",
        choices=["preserve_splits", "merge_to_train"],
        default="preserve_splits",
    )
    p.add_argument(
        "--format", "-f",
        choices=["yolo", "coco"],
        default=None,
        help="Force format (auto-detected if omitted).",
    )
    p.add_argument(
        "--cross-format", "-x",
        action="store_true",
        help="Allow merging datasets of different formats.",
    )
    p.add_argument("--quiet", "-q", action="store_true")
    return p.parse_args(argv)


def _parse_inline_mapping(specs: List[str]) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid mapping: {spec!r}")
        canonical, variants_str = spec.split("=", 1)
        variants = [v.strip() for v in variants_str.split(",") if v.strip()]
        mapping[canonical.strip()] = variants
    return mapping


def main(argv=None):
    args = parse_args(argv)

    level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(level=level, format="%(name)s - %(levelname)s - %(message)s")

    from detection_toolkit.merge import MergeConfig, DatasetMerger
    from detection_toolkit.core.models import DatasetFormat

    if args.config:
        config = MergeConfig.from_yaml(args.config)
    else:
        label_mapping = _parse_inline_mapping(args.map) if args.map else {}
        config = MergeConfig(
            label_mapping=label_mapping,
            missing_splits=args.missing_splits,
            image_conflicts=args.image_conflicts,
            unmapped_classes=args.unmapped_classes,
            output_splits=args.output_splits,
        )

    merger = DatasetMerger(config, verbose=not args.quiet)

    if args.cross_format:
        out_fmt = DatasetFormat(args.format) if args.format else DatasetFormat.COCO
        result = merger.merge_cross_format(args.datasets, args.output, out_fmt)
    else:
        fmt = DatasetFormat(args.format) if args.format else None
        result = merger.merge(args.datasets, args.output, format=fmt)

    print(f"Merge complete:")
    print(f"  Classes: {result.num_classes} -> {result.class_list}")
    print(f"  Images:  {result.total_images}")
    print(f"  Splits:  {result.split_names}")
    print(f"  Output:  {args.output}")


if __name__ == "__main__":
    main()
