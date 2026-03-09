#!/usr/bin/env python3
"""Tile a dataset into fixed-size patches.

Usage::

    # Disk mode — tile a local YOLO/COCO dataset
    python -m detection_toolkit.scripts.tile datasets/my_ds output/tiled \\
        --tile-size 640 --overlap 0.3

    python -m detection_toolkit.scripts.tile datasets/my_ds output/tiled \\
        --config tiling_config.yaml

    # FiftyOne mode — tile a FiftyOne dataset
    python -m detection_toolkit.scripts.tile --fo my_dataset output/tiled \\
        --tile-size 640 --to coco
"""

import logging
import sys
from pathlib import Path

from detection_toolkit.cli import make_parser


def parse_args(argv=None):
    p = make_parser("Tile a dataset into fixed-size patches.")

    # input is optional: required for disk mode, absent for --fo mode
    p.add_argument("input", nargs="?", default=None,
                   help="Input dataset directory (disk mode). "
                        "Omit when using --fo.")
    p.add_argument("output", help="Output tiled dataset directory.")

    # FiftyOne input mode
    fo_group = p.add_argument_group("FiftyOne input")
    fo_group.add_argument(
        "--fo", type=str, default=None, metavar="DATASET",
        help="Use a FiftyOne dataset as input instead of a disk directory. "
             "Pass the dataset name (or a saved view name).",
    )
    fo_group.add_argument(
        "--to", choices=["yolo", "coco"], default="yolo",
        help="Output format when using --fo mode (default: yolo).",
    )
    fo_group.add_argument(
        "--name", type=str, default=None, metavar="NAME",
        help="Name for the reimported tiled dataset in FiftyOne. "
             "Defaults to '{source}_tiled_{tile_size}'.",
    )
    fo_group.add_argument(
        "--no-reimport", action="store_true",
        help="Do not reimport tiled results back into FiftyOne.",
    )

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
    p.add_argument("--disjoint-strategy", choices=["bridge", "split"],
                   default="bridge",
                   help="How to handle annotations that become multiple disjoint "
                        "polygons after tile clipping (e.g. concave shapes split "
                        "by a tile boundary). 'bridge' connects sub-polygons via "
                        "zero-width bridges into a single polygon (default). "
                        "'split' emits each sub-polygon as a separate instance. "
                        "Only affects YOLO output; COCO natively supports "
                        "multi-polygon annotations.")

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

    # Output structure
    p.add_argument("--flatten", action="store_true",
                   help="Flatten output directory structure. Only applies to "
                        "unlabeled image datasets.")

    # Format
    p.add_argument("--format", "-f", choices=["yolo", "coco"], default=None,
                   help="Force dataset format (auto-detected if omitted).")

    p.add_argument("--quiet", "-q", action="store_true",
                   help="Suppress progress output.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # Validate: exactly one of input or --fo must be set
    if args.fo and args.input:
        print("error: cannot use both positional input and --fo", file=sys.stderr)
        sys.exit(2)
    if not args.fo and not args.input:
        print("error: provide either a positional input directory or --fo DATASET",
              file=sys.stderr)
        sys.exit(2)

    level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(level=level, format="%(name)s - %(levelname)s - %(message)s")
    log = logging.getLogger("dt-tile")

    from detection_toolkit.tiling import TilingConfig, DatasetTiler

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
            flatten=args.flatten,
            disjoint_strategy=args.disjoint_strategy,
        )

    output_path = Path(args.output)
    tiler = DatasetTiler(config, verbose=not args.quiet)

    if args.fo:
        # FiftyOne mode
        # Warn on incompatible flags
        if args.flatten:
            log.warning("--flatten is ignored in --fo mode")
        if args.format:
            log.warning("--format is ignored in --fo mode (use --to instead)")
        if args.fraction is not None:
            log.warning("--fraction is ignored in --fo mode "
                        "(pre-filter your dataset view instead)")

        import fiftyone as fo

        dataset = fo.load_dataset(args.fo)
        result = tiler.tile_fo(
            dataset,
            output_path,
            output_format=args.to,
            reimport=not args.no_reimport,
            dataset_name=args.name,
        )
    else:
        # Disk mode
        from detection_toolkit.core.models import DatasetFormat

        fmt = DatasetFormat(args.format) if args.format else None
        result = tiler.tile(Path(args.input), output_path, format=fmt)

    print("Tiling complete:")
    print(f"  Tile size:  {config.tile_size}px, overlap={config.overlap_percent:.0%}")
    if result is not None:
        print(f"  Classes:    {result.num_classes}")
        print(f"  Tiles:      {result.total_images}")
        for name, split in result.splits.items():
            print(f"    {name}: {split.num_images} tiles")
    print(f"  Output:     {output_path}")


if __name__ == "__main__":
    main()
