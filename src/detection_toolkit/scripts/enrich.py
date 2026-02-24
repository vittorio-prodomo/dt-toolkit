#!/usr/bin/env python3
"""Enrich a FiftyOne dataset with computed label attributes.

Usage::

    python -m detection_toolkit.scripts.enrich my_dataset
    python -m detection_toolkit.scripts.enrich my_dataset -l detections -l segmentations
"""

import argparse
import logging
import sys

from detection_toolkit.cli import make_parser


def parse_args(argv=None):
    p = make_parser("Enrich a FiftyOne dataset with computed label attributes.")
    p.add_argument("name", type=str, help="FiftyOne dataset name.")
    p.add_argument(
        "--label-field", "-l",
        action="append",
        default=None,
        metavar="FIELD",
        help="Label field(s) to enrich (repeatable; default: auto-detect).",
    )
    p.add_argument(
        "--batch-size", "-b",
        type=int,
        default=1000,
        metavar="N",
        help="Batch size for mask processing (default: 1000).",
    )
    verbosity = p.add_mutually_exclusive_group()
    verbosity.add_argument("--quiet", "-q", action="store_true",
                           help="Suppress informational output.")
    verbosity.add_argument("--verbose", "-v", action="store_true",
                           help="Show detailed debug output.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.quiet:
        level = logging.WARNING
    elif args.verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logging.basicConfig(
        level=level, format="%(name)s - %(levelname)s - %(message)s"
    )

    import fiftyone as fo

    try:
        dataset = fo.load_dataset(args.name)
    except ValueError:
        existing = fo.list_datasets()
        print(f"Error: Dataset '{args.name}' not found.", file=sys.stderr)
        if existing:
            print(f"Available datasets: {existing}", file=sys.stderr)
        sys.exit(1)

    from detection_toolkit.enrich import DatasetEnricher

    enricher = DatasetEnricher(verbose=not args.quiet)
    enriched = enricher.enrich(
        dataset,
        label_fields=args.label_field,
        batch_size=args.batch_size,
        interactive=True,
    )

    if enriched:
        print("Enrichment complete.")
    else:
        print("Enrichment aborted.")


if __name__ == "__main__":
    main()
