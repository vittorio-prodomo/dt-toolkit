#!/usr/bin/env python3
"""Split a FiftyOne dataset into stratified train/val/test sets.

Usage::

    dt-split my_dataset --ratio 0.8
    dt-split my_dataset --ratio 0.7 0.15 --group-by bridge_id
"""

import argparse
import logging
import sys

from detection_toolkit.cli import make_parser


def _ratio_to_dict(ratio_values: list) -> dict:
    """Convert CLI ratio values to a split name -> proportion dict.

    One value:  train/val (remainder is val).
    Two values: train/val/test (remainder is test).
    """
    if len(ratio_values) == 1:
        train = ratio_values[0]
        if train <= 0 or train >= 1:
            raise ValueError(f"Ratio must be between 0 and 1, got {train}")
        return {"train": train, "val": round(1.0 - train, 10)}

    if len(ratio_values) == 2:
        train, val = ratio_values
        test = round(1.0 - train - val, 10)
        if test < 0:
            raise ValueError(
                f"Ratios sum to {train + val:.4f} which exceeds 1.0"
            )
        return {"train": train, "val": val, "test": test}

    raise ValueError("--ratio accepts 1 or 2 values")


def _print_summary(result):
    """Print a split distribution table."""
    splits = list(result.split_counts.keys())

    # Collect all class names, display _negative as "negative"
    all_classes = []
    for s in splits:
        for cls in result.class_distribution[s]:
            if cls not in all_classes:
                all_classes.append(cls)

    header = f"{'':>14s}" + "".join(f"{s:>10s}" for s in splits)
    print(f"\nSplit distribution:\n{header}")
    print(
        f"{'images':>14s}"
        + "".join(f"{result.split_counts[s]:>10d}" for s in splits)
    )
    for cls in all_classes:
        display_name = cls.lstrip("_") if cls.startswith("_") else cls
        row = f"{display_name:>14s}"
        for s in splits:
            row += f"{result.class_distribution[s].get(cls, 0):>10d}"
        print(row)
    print()


def parse_args(argv=None):
    p = make_parser(
        "Split a FiftyOne dataset into stratified train/val/test sets."
    )
    p.add_argument(
        "name", type=str, metavar="DATASET_NAME",
        help="FiftyOne dataset name.",
    )
    p.add_argument(
        "--ratio", "-r",
        type=float,
        nargs="+",
        default=[0.8],
        metavar="F",
        help=(
            "Split proportions. One value = train/val "
            "(e.g. 0.8 means 80%% train, 20%% val). "
            "Two values = train/val/test "
            "(e.g. 0.7 0.15 means 70/15/15). Default: 0.8."
        ),
    )
    p.add_argument(
        "--label-field", "-l",
        default=None,
        metavar="FIELD",
        help="Label field for stratification. Auto-resolved if omitted.",
    )
    p.add_argument(
        "--group-by", "-g",
        default=None,
        metavar="FIELD",
        help="Sample field for group-aware splitting (e.g. bridge_id).",
    )
    p.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    p.add_argument(
        "--split-field",
        default="split",
        metavar="FIELD",
        help="Name of the sample field to write (default: split).",
    )
    p.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip overwrite confirmation prompt.",
    )
    p.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress informational output.",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(
        level=level, format="%(name)s - %(levelname)s - %(message)s",
    )

    try:
        ratios = _ratio_to_dict(args.ratio)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    import fiftyone as fo

    from detection_toolkit.fiftyone_hub.splitter import DatasetSplitter

    dataset = fo.load_dataset(args.name)

    # Check for existing split field
    if dataset.has_sample_field(args.split_field) and not args.yes:
        existing = dataset.count_values(args.split_field)
        summary = ", ".join(f"{k}: {v}" for k, v in sorted(existing.items()))
        answer = input(
            f"Field '{args.split_field}' already exists "
            f"with values {{{summary}}}. Overwrite? [y/N] "
        )
        if answer.lower() not in ("y", "yes"):
            print("Aborted.")
            sys.exit(0)

    splitter = DatasetSplitter(
        ratios=ratios,
        label_field=args.label_field,
        group_by=args.group_by,
        seed=args.seed,
        split_field=args.split_field,
    )

    result = splitter.split(dataset)
    _print_summary(result)


if __name__ == "__main__":
    main()
