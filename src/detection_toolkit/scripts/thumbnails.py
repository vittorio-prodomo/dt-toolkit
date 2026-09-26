#!/usr/bin/env python3
"""Generate thumbnails for an existing FiftyOne dataset.

Usage::

    dt-thumbnails my_dataset
    dt-thumbnails my_dataset --size 512
    dt-thumbnails my_dataset --output-dir /path/to/thumbs
    dt-thumbnails my_dataset --check        # read-only health audit

Re-running is cheap: only missing or out-of-date thumbnails are written.
"""

import logging
import sys

from detection_toolkit.cli import make_parser


def parse_args(argv=None):
    p = make_parser("Generate thumbnails for an existing FiftyOne dataset.")
    p.add_argument("dataset", help="FiftyOne dataset name.")
    p.add_argument(
        "--size", "-s", type=int, default=256,
        help="Thumbnail height in pixels (default: 256).",
    )
    p.add_argument(
        "--output-dir", "-o", default=None,
        help="Directory where the .{name}_thumbnails/ folder is created "
             "(default: parent of the dataset's common filepath ancestor).",
    )
    p.add_argument(
        "--format", "-f", dest="fmt", choices=("webp", "jpg", "keep"), default="webp",
        help="Thumbnail format (default: webp; 'keep' = the source's own format). "
             "The extension is appended: f.png -> f.png.webp.",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Regenerate every thumbnail, even up-to-date ones.",
    )
    p.add_argument(
        "--num-workers", "-j", type=int, default=None,
        help="Worker processes (default: half the CPUs, max 16).",
    )
    p.add_argument(
        "--check", action="store_true",
        help="Only audit the existing thumbnails (unset / relative / missing / "
             "shared by different images); writes nothing, exit 1 if unhealthy.",
    )

    g = p.add_mutually_exclusive_group()
    g.add_argument("--quiet", "-q", action="store_true",
                   help="Suppress informational output.")
    g.add_argument("--verbose", "-v", action="store_true",
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
    logging.basicConfig(level=level,
                        format="%(name)s - %(levelname)s - %(message)s")

    import fiftyone as fo

    try:
        dataset = fo.load_dataset(args.dataset)
    except ValueError:
        existing = fo.list_datasets()
        print(f"Error: Dataset '{args.dataset}' not found.", file=sys.stderr)
        if existing:
            print(f"Available datasets: {existing}", file=sys.stderr)
        sys.exit(1)

    from detection_toolkit.fiftyone_hub.thumbnails import (
        FIELD, audit_thumbnails, create_thumbnails,
    )

    if args.check:
        thumbs = (dataset.values(FIELD) if dataset.has_sample_field(FIELD)
                  else [None] * len(dataset))
        report = audit_thumbnails(dataset.values("filepath"), thumbs)
        report["grid_media_field"] = dataset.app_config.grid_media_field
        for key, value in report.items():
            print(f"{key:28s} {value}")
        sys.exit(0 if report["ok"] else 1)

    summary = create_thumbnails(dataset, output_dir=args.output_dir, size=args.size,
                                force=args.force, num_workers=args.num_workers,
                                fmt=args.fmt)
    print(f"Thumbnails ready for '{args.dataset}' ({len(dataset)} samples, "
          f"size={args.size}px, {args.fmt}): {summary['generated']} written, "
          f"{summary['reused']} up to date, {summary['failed']} failed "
          f"-> {summary['root']}")


if __name__ == "__main__":
    main()
