#!/usr/bin/env python3
"""Create a FiftyOne visualization from embeddings or pre-reduced points.

Usage::

    dt-visualize my_dataset --points clip_umap --brain-key clip_viz
    dt-visualize my_dataset --embeddings clip_embedding --brain-key clip_viz --method umap
    dt-visualize my_dataset --model clip-vit-base32-torch --brain-key clip_viz
"""

import logging
import sys

from detection_toolkit.cli import make_parser


def _add_verbosity(p):
    g = p.add_mutually_exclusive_group()
    g.add_argument("--quiet", "-q", action="store_true",
                   help="Suppress informational output.")
    g.add_argument("--verbose", "-v", action="store_true",
                   help="Show detailed debug output.")


def parse_args(argv=None):
    p = make_parser(
        "Create a FiftyOne visualization from embeddings or pre-reduced points."
    )

    p.add_argument("dataset", help="FiftyOne dataset name.")
    p.add_argument("--brain-key", "-k", required=True,
                   help="Brain key to store the visualization under.")

    # Input source (mutually exclusive)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--points", "-p", default=None, metavar="FIELD",
        help="Field containing pre-reduced 2D/3D coordinates (uses method=manual).",
    )
    source.add_argument(
        "--embeddings", "-e", default=None, metavar="FIELD",
        help="Field containing high-dimensional embeddings (reduced by FiftyOne).",
    )
    source.add_argument(
        "--model", default=None, metavar="MODEL",
        help="FiftyOne zoo model name or HuggingFace model ID. "
             "Computes embeddings and reduces in one step.",
    )

    p.add_argument("--num-dims", "-n", type=int, default=2,
                   choices=[2, 3],
                   help="Number of visualization dimensions (default: 2).")
    p.add_argument("--method", "-m", default="umap",
                   choices=["umap", "tsne", "pca"],
                   help="Reduction method when using --embeddings (default: umap).")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Batch size for model inference (only with --model).")
    p.add_argument("--seed", type=int, default=51,
                   help="Random seed (default: 51).")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite brain key if it already exists.")

    _add_verbosity(p)

    return p.parse_args(argv)


def _load_dataset(name):
    import fiftyone as fo

    try:
        return fo.load_dataset(name)
    except ValueError:
        existing = fo.list_datasets()
        print(f"Error: Dataset '{name}' not found.", file=sys.stderr)
        if existing:
            print(f"Available datasets: {existing}", file=sys.stderr)
        sys.exit(1)


def main(argv=None):
    args = parse_args(argv)

    if args.quiet:
        level = logging.WARNING
    elif args.verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(level=level, format="%(name)s - %(levelname)s - %(message)s")

    import numpy as np

    import fiftyone.brain as fob

    dataset = _load_dataset(args.dataset)

    # Check for existing brain key
    if args.brain_key in dataset.list_brain_runs() and not args.overwrite:
        print(
            f"Error: Brain key '{args.brain_key}' already exists. "
            f"Use --overwrite to replace it.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.brain_key in dataset.list_brain_runs() and args.overwrite:
        dataset.delete_brain_run(args.brain_key)

    if args.points:
        # Pre-reduced coordinates — use method='manual'
        raw = dataset.values(args.points)
        # Replace None entries with NaN so numpy can build a homogeneous array
        ndims = args.num_dims
        points = np.array([v if v is not None else [np.nan] * ndims for v in raw])
        fob.compute_visualization(
            dataset,
            points=points,
            method="manual",
            brain_key=args.brain_key,
            num_dims=args.num_dims,
            seed=args.seed,
        )
        print(f"Visualization created: brain_key={args.brain_key}")
        print(f"  source: {args.points} (pre-reduced, method=manual)")
    elif args.embeddings:
        # High-dimensional embeddings — let FiftyOne reduce
        fob.compute_visualization(
            dataset,
            embeddings=args.embeddings,
            method=args.method,
            brain_key=args.brain_key,
            num_dims=args.num_dims,
            seed=args.seed,
        )
        print(f"Visualization created: brain_key={args.brain_key}")
        print(f"  source: {args.embeddings} (method={args.method})")
    else:
        # Model — FiftyOne handles embed + reduce + visualize
        kwargs = {}
        if args.batch_size is not None:
            kwargs["batch_size"] = args.batch_size
        fob.compute_visualization(
            dataset,
            model=args.model,
            method=args.method,
            brain_key=args.brain_key,
            num_dims=args.num_dims,
            seed=args.seed,
            **kwargs,
        )
        print(f"Visualization created: brain_key={args.brain_key}")
        print(f"  source: {args.model} (model, method={args.method})")

    print(f"  samples: {len(dataset)}")
    print(f"  dims: {args.num_dims}")


if __name__ == "__main__":
    main()
