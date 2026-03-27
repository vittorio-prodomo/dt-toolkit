#!/usr/bin/env python3
"""Generate thumbnails for an existing FiftyOne dataset.

Usage::

    dt-thumbnails my_dataset
    dt-thumbnails my_dataset --size 512
    dt-thumbnails my_dataset --output-dir /path/to/thumbs
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

    from detection_toolkit.fiftyone_hub.utils import create_thumbnails

    create_thumbnails(dataset, output_dir=args.output_dir, size=args.size)
    print(f"Thumbnails ready for '{args.dataset}' ({len(dataset)} samples, "
          f"size={args.size}px)")


if __name__ == "__main__":
    main()
