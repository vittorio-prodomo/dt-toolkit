#!/usr/bin/env python3
"""Tile a dataset into fixed-size patches.

Usage::

    python -m detection_toolkit.scripts.tile datasets/my_ds output/tiled \\
        --tile-size 640 --overlap 0.3

    python -m detection_toolkit.scripts.tile datasets/my_ds output/tiled \\
        --config tiling_config.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

from detection_toolkit.cli import make_parser


def parse_args(argv=None):
    p = make_parser("Tile a dataset into fixed-size patches.")
    p.add_argument("input", type=Path, help="Input dataset directory.")
    p.add_argument("output", type=Path, help="Output tiled dataset directory.")

    p.add_argument(
        "--config", "-c",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to a TilingConfig YAML file (overrides other args).",
    )

    # Core tiling params
    p.add_argument("--tile-size", "-t", type=int, default=640, metavar="PX",
                   help="Tile width and height in pixels (default: 640).")
    p.add_argument("--overlap", "-o", type=float, default=0.0, metavar="F",
                   help="Overlap fraction between adjacent tiles, 0.0-1.0 (default: 0.0).")
    p.add_argument("--min-area-ratio", "-m", type=float, default=0.1, metavar="F",
                   help="Discard annotations whose clipped area is below this fraction "
                        "of their original area (default: 0.1).")
    p.add_argument("--max-area-ratio", "-M", type=float, default=1.0, metavar="F",
                   help="Discard annotations whose clipped area exceeds this fraction "
                        "of their original area (default: 1.0).")

    # Edge handling
    p.add_argument("--edge-behavior", "-e", choices=["crop", "padding"], default="crop",
                   help="How to handle tiles at image borders: 'crop' discards "
                        "partial tiles, 'padding' pads them (default: crop).")
    p.add_argument("--padding-color", "-C", type=int, nargs=3, default=[153, 153, 153],
                   metavar=("R", "G", "B"),
                   help="RGB fill color when edge-behavior is 'padding' "
                        "(default: 153 153 153).")

    # Polygon processing
    p.add_argument("--densify-factor", "-d", type=float, default=0.5, metavar="F",
                   help="Add intermediate polygon vertices every tile_size*factor "
                        "pixels to improve clipping accuracy (default: 0.5).")
    p.add_argument("--smoothing-tolerance", "-s", type=float, default=0.1, metavar="F",
                   help="Simplify clipped polygons with this tolerance in pixels "
                        "to reduce point count (default: 0.1).")

    # Negative sampling
    p.add_argument("--negative-ratio", "-n", type=float, default=0.1, metavar="F",
                   help="Ratio of negative (empty) to positive tiles to keep; "
                        "-1 keeps all negatives (default: 0.1).")
    p.add_argument("--random-seed", "-r", type=int, default=42, metavar="N",
                   help="Random seed for negative tile sampling (default: 42).")
    p.add_argument("--fraction", "-F", type=float, default=None, metavar="F",
                   help="Process only this fraction of source images, useful "
                        "for quick testing (default: all).")

    # Image compression
    p.add_argument("--compression", "-Q", type=int, default=90, metavar="N",
                   help="JPEG quality 0-100 for saved tiles (default: 90).")

    # Parallelism
    p.add_argument("--max-processes", "-P", type=int, default=None, metavar="N",
                   help="Max parallel worker processes (default: CPU count).")

    # Format
    p.add_argument("--format", "-f", choices=["yolo", "coco"], default=None,
                   help="Force dataset format (auto-detected if omitted).")

    p.add_argument("--quiet", "-q", action="store_true",
                   help="Suppress progress output.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(level=level, format="%(name)s - %(levelname)s - %(message)s")

    from detection_toolkit.tiling import TilingConfig, DatasetTiler
    from detection_toolkit.core.models import DatasetFormat

    if args.config:
        config = TilingConfig.from_yaml(args.config)
    else:
        config = TilingConfig(
            tile_size=args.tile_size,
            overlap_percent=args.overlap,
            edge_behavior=args.edge_behavior,
            padding_value=tuple(args.padding_color),
            min_area_ratio=args.min_area_ratio,
            max_area_ratio=args.max_area_ratio,
            densify_factor=args.densify_factor,
            smoothing_tolerance=args.smoothing_tolerance,
            negative_sample_ratio=args.negative_ratio,
            compression=args.compression,
            max_processes=args.max_processes,
            random_seed=args.random_seed,
            fraction=args.fraction,
        )

    fmt = DatasetFormat(args.format) if args.format else None
    tiler = DatasetTiler(config, verbose=not args.quiet)
    result = tiler.tile(args.input, args.output, format=fmt)

    print(f"Tiling complete:")
    print(f"  Tile size:  {config.tile_size}px, overlap={config.overlap_percent:.0%}")
    print(f"  Classes:    {result.num_classes}")
    print(f"  Tiles:      {result.total_images}")
    for name, split in result.splits.items():
        print(f"    {name}: {split.num_images} tiles")
    print(f"  Output:     {args.output}")


if __name__ == "__main__":
    main()
