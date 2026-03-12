#!/usr/bin/env python3
"""Dimensionality reduction on FiftyOne dataset embeddings.

Usage::

    dt-reduce my_dataset --input-field clip_embedding --output-field clip_umap
    dt-reduce my_dataset -i clip_embedding -o clip_pca --method pca --n-components 8
"""

import logging
import sys

import numpy as np

from detection_toolkit.cli import make_parser


def _add_verbosity(p):
    g = p.add_mutually_exclusive_group()
    g.add_argument("--quiet", "-q", action="store_true",
                   help="Suppress informational output.")
    g.add_argument("--verbose", "-v", action="store_true",
                   help="Show detailed debug output.")


def parse_args(argv=None):
    p = make_parser("Dimensionality reduction on FiftyOne dataset embeddings.")

    p.add_argument("dataset", help="FiftyOne dataset name.")
    p.add_argument("--input-field", "-i", required=True,
                   help="Source embedding field.")
    p.add_argument("--output-field", "-o", required=True,
                   help="Field to store reduced embeddings.")
    p.add_argument("--method", "-m", default="umap",
                   choices=["umap", "tsne", "pca", "pacmap"],
                   help="Reduction method (default: umap).")
    p.add_argument("--n-components", "-n", type=int, default=32,
                   help="Number of output dimensions (default: 32).")
    p.add_argument("--random-state", type=int, default=42,
                   help="Random seed (default: 42).")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite output field if it exists.")

    # Method-specific arguments
    p.add_argument("--n-neighbors", type=int, default=15,
                   help="Number of neighbors (UMAP/PaCMAP, default: 15).")
    p.add_argument("--min-dist", type=float, default=0.1,
                   help="Minimum distance (UMAP, default: 0.1).")
    p.add_argument("--metric", default="cosine",
                   help="Distance metric (UMAP/t-SNE, default: cosine).")
    p.add_argument("--perplexity", type=float, default=30.0,
                   help="Perplexity (t-SNE, default: 30.0).")

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


def _collect_method_kwargs(args):
    """Map CLI args to method-specific keyword arguments."""
    method = args.method
    if method == "umap":
        return {
            "n_neighbors": args.n_neighbors,
            "min_dist": args.min_dist,
            "metric": args.metric,
        }
    elif method == "tsne":
        return {
            "perplexity": args.perplexity,
            "metric": args.metric,
        }
    elif method == "pacmap":
        return {
            "n_neighbors": args.n_neighbors,
        }
    elif method == "pca":
        return {}
    return {}


def main(argv=None):
    args = parse_args(argv)

    if args.quiet:
        level = logging.WARNING
    elif args.verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(level=level, format="%(name)s - %(levelname)s - %(message)s")

    from detection_toolkit.reduction import DimensionalityReducer

    dataset = _load_dataset(args.dataset)
    kwargs = _collect_method_kwargs(args)

    reducer = DimensionalityReducer()

    result = reducer.reduce(
        dataset,
        input_field=args.input_field,
        output_field=args.output_field,
        method=args.method,
        n_components=args.n_components,
        random_state=args.random_state,
        overwrite=args.overwrite,
        **kwargs,
    )

    n_samples = result.shape[0]
    valid = int(np.isfinite(result).all(axis=1).sum())

    print(f"Reduction complete ({args.method}): {n_samples} samples")
    print(f"  input:  {args.input_field}")
    print(f"  output: {args.output_field} ({args.n_components}d)")
    print(f"  valid:  {valid}")


if __name__ == "__main__":
    main()
