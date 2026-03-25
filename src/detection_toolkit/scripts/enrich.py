#!/usr/bin/env python3
"""Enrich a FiftyOne dataset with computed attributes.

Sub-commands:

    dt-enrich areas   — compute area_px / area_rel for label annotations
    dt-enrich gps     — extract GPS lat/lon from EXIF and create map visualization
    dt-enrich all     — run all applicable enrichments

Usage::

    dt-enrich areas my_dataset
    dt-enrich gps my_dataset
    dt-enrich all my_dataset
    dt-enrich areas my_dataset -l detections --batch-size 500
    dt-enrich gps my_dataset --brain-key gps_coords
"""

import argparse
import logging
import sys

from detection_toolkit.cli import make_parser, formatter_class


def _load_dataset(name: str):
    """Load a FiftyOne dataset by name, or exit with an error."""
    import fiftyone as fo

    try:
        return fo.load_dataset(name)
    except ValueError:
        existing = fo.list_datasets()
        print(f"Error: Dataset '{name}' not found.", file=sys.stderr)
        if existing:
            print(f"Available datasets: {existing}", file=sys.stderr)
        sys.exit(1)


def _setup_logging(args):
    if getattr(args, "quiet", False):
        level = logging.WARNING
    elif getattr(args, "verbose", False):
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(
        level=level, format="%(name)s - %(levelname)s - %(message)s"
    )


def _add_common_args(parser):
    """Add dataset name and verbosity flags shared by all sub-commands."""
    parser.add_argument("name", type=str, help="FiftyOne dataset name.")
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument("--quiet", "-q", action="store_true",
                           help="Suppress informational output.")
    verbosity.add_argument("--verbose", "-v", action="store_true",
                           help="Show detailed debug output.")


# ------------------------------------------------------------------
# Sub-command: areas
# ------------------------------------------------------------------

def cmd_areas(args):
    """Compute area_px and area_rel for label annotations."""
    _setup_logging(args)
    dataset = _load_dataset(args.name)

    from detection_toolkit.enrich import DatasetEnricher

    enricher = DatasetEnricher(verbose=not args.quiet)
    enriched = enricher.enrich(
        dataset,
        label_fields=args.label_field,
        batch_size=args.batch_size,
        interactive=True,
    )
    if enriched:
        print("Area enrichment complete.")
    else:
        print("Area enrichment skipped.")


# ------------------------------------------------------------------
# Sub-command: gps
# ------------------------------------------------------------------

def cmd_gps(args):
    """Extract GPS lat/lon from EXIF and create a map visualization."""
    _setup_logging(args)
    dataset = _load_dataset(args.name)

    from detection_toolkit.enrich.gps import enrich_gps

    enriched = enrich_gps(
        dataset,
        brain_key=args.brain_key,
        verbose=not getattr(args, "quiet", False),
    )
    if enriched:
        print("GPS enrichment complete.")
    else:
        print("No GPS data found — GPS enrichment skipped.")


# ------------------------------------------------------------------
# Sub-command: all
# ------------------------------------------------------------------

def cmd_all(args):
    """Run all applicable enrichments."""
    _setup_logging(args)
    dataset = _load_dataset(args.name)
    verbose = not getattr(args, "quiet", False)

    # Areas
    from detection_toolkit.enrich import DatasetEnricher

    enricher = DatasetEnricher(verbose=verbose)
    areas_done = enricher.enrich(
        dataset,
        label_fields=None,
        batch_size=args.batch_size,
        interactive=False,
    )
    if areas_done:
        print("Area enrichment complete.")
    else:
        print("Area enrichment skipped (no label fields or already done).")

    # GPS
    from detection_toolkit.enrich.gps import enrich_gps

    gps_done = enrich_gps(
        dataset,
        brain_key=args.brain_key,
        verbose=verbose,
    )
    if gps_done:
        print("GPS enrichment complete.")
    else:
        print("No GPS data found — GPS enrichment skipped.")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main(argv=None):
    p = make_parser(
        "Enrich a FiftyOne dataset with computed attributes.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # areas
    p_areas = sub.add_parser(
        "areas",
        help="Compute area_px / area_rel for label annotations.",
        formatter_class=formatter_class(),
    )
    _add_common_args(p_areas)
    p_areas.add_argument(
        "--label-field", "-l", action="append", default=None,
        metavar="FIELD",
        help="Label field(s) to enrich (repeatable; default: auto-detect).",
    )
    p_areas.add_argument(
        "--batch-size", "-b", type=int, default=1000, metavar="N",
        help="Batch size for mask processing (default: 1000).",
    )
    p_areas.set_defaults(func=cmd_areas)

    # gps
    p_gps = sub.add_parser(
        "gps",
        help="Extract GPS lat/lon from EXIF and create map visualization.",
        formatter_class=formatter_class(),
    )
    _add_common_args(p_gps)
    p_gps.add_argument(
        "--brain-key", "-k", default="gps_location",
        help="Brain key for the visualization (default: gps_location).",
    )
    p_gps.set_defaults(func=cmd_gps)

    # all
    p_all = sub.add_parser(
        "all",
        help="Run all applicable enrichments.",
        formatter_class=formatter_class(),
    )
    _add_common_args(p_all)
    p_all.add_argument(
        "--batch-size", "-b", type=int, default=1000, metavar="N",
        help="Batch size for mask area processing (default: 1000).",
    )
    p_all.add_argument(
        "--brain-key", "-k", default="gps_location",
        help="Brain key for GPS visualization (default: gps_location).",
    )
    p_all.set_defaults(func=cmd_all)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
