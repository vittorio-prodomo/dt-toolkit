#!/usr/bin/env python3
"""Import datasets into FiftyOne from disk or CVAT.

Usage::

    python -m detection_toolkit.scripts.import_fo disk <path> <name> [opts]
    python -m detection_toolkit.scripts.import_fo cvat <name> [opts]
"""

import argparse
import logging
import sys
from pathlib import Path

from detection_toolkit.cli import make_parser, formatter_class


def _add_shared_opts(p):
    """Add options shared by both subcommands."""
    p.add_argument("--no-persistent", "-N", action="store_true",
                   help="Create a non-persistent (session-only) dataset.")
    p.add_argument(
        "--create-thumbnails", "-T", action="store_true",
        help="Generate thumbnails for faster grid browsing.",
    )
    p.add_argument(
        "--thumbnail-size", "-Z", type=int, default=256,
        metavar="PX",
        help="Thumbnail height in pixels (default: 256).",
    )
    p.add_argument(
        "--enrich", "-E", action="store_true",
        help="Compute additional label attributes (area) after import.",
    )
    verbosity = p.add_mutually_exclusive_group()
    verbosity.add_argument("--quiet", "-q", action="store_true",
                           help="Suppress informational output.")
    verbosity.add_argument("--verbose", "-v", action="store_true",
                           help="Show detailed debug output (e.g. per-label "
                                "info for skipped annotations).")


def parse_args(argv=None):
    p = make_parser("Import a dataset into FiftyOne.", raw_text=True)
    sub = p.add_subparsers(dest="subcommand", title="subcommands")

    # ---- disk (existing behaviour) ------------------------------------
    disk = sub.add_parser(
        "disk",
        help="Import an on-disk YOLO/COCO dataset.",
        description="Import an on-disk YOLO/COCO dataset into FiftyOne.",
        formatter_class=formatter_class(raw_text=True),
    )
    disk.add_argument("dataset", type=Path,
                      help="Path to dataset root directory.")
    disk.add_argument("name", type=str, help="FiftyOne dataset name.")
    disk.add_argument(
        "--format", "-f",
        choices=["yolo", "coco"],
        default=None,
        help="Force format (auto-detected if omitted).",
    )
    disk.add_argument("--shuffle", "-S", action="store_true",
                      help="Shuffle samples after loading.")
    disk.add_argument(
        "--include-all-data", "-a", action="store_true",
        help="Include images without labels (negatives). YOLO only.",
    )
    disk.add_argument(
        "--use-polylines", "-p", action="store_true",
        help="Import segmentations as polylines instead of dense masks. "
             "COCO only.",
    )
    disk.add_argument("--overwrite", "-O", action="store_true",
                      help="Delete existing FiftyOne dataset with this name.")
    _add_shared_opts(disk)

    # ---- cvat (new) ---------------------------------------------------
    cvat = sub.add_parser(
        "cvat",
        help="Import annotations (and optionally media) from a CVAT server.",
        description="Import annotations from CVAT into a FiftyOne dataset.",
        formatter_class=formatter_class(raw_text=True),
    )
    cvat.add_argument("name", type=str, help="FiftyOne dataset name.")

    # CVAT source (mutually exclusive)
    src = cvat.add_mutually_exclusive_group(required=True)
    src.add_argument("--project-name", "-P", type=str,
                     metavar="NAME",
                     help="CVAT project name.")
    src.add_argument("--project-id", "-I", type=int,
                     metavar="ID",
                     help="CVAT project ID.")
    src.add_argument("--task-ids", "-t", type=int, nargs="+",
                     metavar="ID",
                     help="One or more CVAT task IDs.")
    src.add_argument("--job-ids", "-j", type=int, nargs="+",
                     metavar="ID",
                     help="One or more CVAT job IDs.")

    # Data mapping / media
    cvat.add_argument(
        "--data-path", "-d", type=str, default=None,
        metavar="PATH",
        help="Root of the connected file-share that CVAT sees, NOT the\n"
             "leaf directory where images happen to live.  CVAT stores\n"
             "frame filenames as paths relative to this root (e.g.\n"
             "'renamed/00001.jpg'), so the path must match the share\n"
             "mount point so that relative lookups resolve correctly.\n"
             "Can also be a JSON manifest or an existing dataset name.",
    )
    cvat.add_argument("--download-media", "-D", action="store_true",
                      help="Download media files from CVAT.")
    cvat.add_argument("--num-workers", "-w", type=int, default=None,
                      metavar="N",
                      help="Threads for media download.")

    # Label control
    cvat.add_argument(
        "--label-types", "-l", nargs="+", default=None,
        metavar="TYPE",
        help="Label types to import (e.g. detections segmentations).",
    )
    cvat.add_argument("--no-insert-new", "-n", action="store_true",
                      help="Do not create samples for media not already "
                           "in the dataset.")
    cvat.add_argument("--occluded-attr", "-o", type=str, default=None,
                      metavar="ATTR",
                      help="Attribute name to store occlusion info.")
    cvat.add_argument("--group-id-attr", "-g", type=str, default=None,
                      metavar="ATTR",
                      help="Attribute name to store group IDs.")

    # CVAT connection
    cvat.add_argument("--url", "-U", type=str, default=None,
                      metavar="URL",
                      help="CVAT server URL.")
    cvat.add_argument("--username", "-u", type=str, default=None,
                      metavar="USER",
                      help="CVAT username.")
    cvat.add_argument("--password", "-p", type=str, default=None,
                      metavar="PASS",
                      help="CVAT password.")
    cvat.add_argument("--organization", "-O", type=str, default=None,
                      metavar="ORG",
                      help="CVAT organization (workspace) name.")

    # cvat-specific dataset handling (mutually exclusive)
    mode = cvat.add_mutually_exclusive_group()
    mode.add_argument("--overwrite", "-W", action="store_true",
                      help="Delete existing dataset entirely and recreate.")
    mode.add_argument("--sync", "-S", action="store_true",
                      help="Keep samples, clear labels, re-import all from CVAT.")
    _add_shared_opts(cvat)

    args = p.parse_args(argv)

    if args.subcommand is None:
        p.print_help()
        sys.exit(0)

    return args


def main(argv=None):
    args = parse_args(argv)

    if args.quiet:
        level = logging.WARNING
    else:
        level = logging.INFO

    logging.basicConfig(level=level,
                        format="%(name)s - %(levelname)s - %(message)s")

    from detection_toolkit.fiftyone_hub import FiftyOneDatasetLoader

    # FiftyOne configures its root logger at INFO on import.  Only
    # lower the specific loggers we care about to avoid pymongo noise.
    if args.verbose:
        logging.getLogger("fiftyone.utils.annotations").setLevel(
            logging.DEBUG
        )

    loader = FiftyOneDatasetLoader(verbose=not args.quiet)

    if args.subcommand == "disk":
        from detection_toolkit.core.models import DatasetFormat

        fmt = DatasetFormat(args.format) if args.format else None
        dataset = loader.load(
            dataset_path=args.dataset,
            dataset_name=args.name,
            format=fmt,
            persistent=not args.no_persistent,
            overwrite=args.overwrite,
            shuffle=args.shuffle,
            include_all_data=args.include_all_data,
            use_polylines=args.use_polylines,
            create_thumbs=args.create_thumbnails,
            thumbnail_size=args.thumbnail_size,
        )

    elif args.subcommand == "cvat":
        dataset = loader.load_cvat(
            dataset_name=args.name,
            project_name=args.project_name,
            project_id=args.project_id,
            task_ids=args.task_ids,
            job_ids=args.job_ids,
            data_path=args.data_path,
            label_types=args.label_types,
            insert_new=not args.no_insert_new,
            download_media=args.download_media,
            num_workers=args.num_workers,
            occluded_attr=args.occluded_attr,
            group_id_attr=args.group_id_attr,
            persistent=not args.no_persistent,
            overwrite=args.overwrite,
            sync=args.sync,
            create_thumbs=args.create_thumbnails,
            thumbnail_size=args.thumbnail_size,
            cvat_url=args.url,
            cvat_username=args.username,
            cvat_password=args.password,
            cvat_organization=args.organization,
        )

    if args.enrich:
        from detection_toolkit.enrich import DatasetEnricher
        enricher = DatasetEnricher(verbose=not args.quiet)
        enricher.enrich(dataset, interactive=False)

    print("FiftyOne import complete:")
    print(f"  Dataset:  {dataset.name}")
    print(f"  Samples:  {len(dataset)}")
    print(f"  Media:    {dataset.media_type}")
    if args.subcommand == "disk":
        splits = dataset.distinct("split")
        if splits:
            print(f"  Splits:   {splits}")


if __name__ == "__main__":
    main()
