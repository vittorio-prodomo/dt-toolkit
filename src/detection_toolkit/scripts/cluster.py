#!/usr/bin/env python3
"""Clustering on FiftyOne dataset embeddings.

Usage::

    dt-cluster my_dataset --input-field clip_embedding --output-field cluster_id
    dt-cluster my_dataset -i clip_embedding -o cluster_id --method kmeans --n-clusters 10
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
    p = make_parser("Clustering on FiftyOne dataset embeddings.")

    p.add_argument("dataset", help="FiftyOne dataset name.")
    p.add_argument("--input-field", "-i", required=True,
                   help="Source embedding field.")
    p.add_argument("--output-field", "-o", default=None,
                   help="Field to store cluster labels (omit for dry run).")
    p.add_argument("--method", "-m", default="hdbscan",
                   choices=["hdbscan", "kmeans", "dbscan", "agglomerative"],
                   help="Clustering method (default: hdbscan).")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite output field if it exists.")

    # Method-specific arguments
    p.add_argument("--min-cluster-size", type=int, default=15,
                   help="Minimum cluster size (HDBSCAN, default: 15).")
    p.add_argument("--min-samples", type=int, default=None,
                   help="Minimum samples (HDBSCAN/DBSCAN).")
    p.add_argument("--cluster-selection-method", default="eom",
                   choices=["eom", "leaf"],
                   help="Cluster selection method (HDBSCAN, default: eom).")
    p.add_argument("--n-clusters", type=int, default=None,
                   help="Number of clusters (K-Means/Agglomerative).")
    p.add_argument("--eps", type=float, default=0.5,
                   help="Epsilon radius (DBSCAN, default: 0.5).")
    p.add_argument("--metric", default="euclidean",
                   help="Distance metric (HDBSCAN/DBSCAN/Agglomerative, default: euclidean).")
    p.add_argument("--linkage", default="ward",
                   choices=["ward", "complete", "average", "single"],
                   help="Linkage criterion (Agglomerative, default: ward).")

    _add_verbosity(p)

    return p.parse_args(argv)


def _collect_method_kwargs(args):
    """Map CLI args to method-specific keyword arguments."""
    method = args.method
    if method == "hdbscan":
        kwargs = {
            "min_cluster_size": args.min_cluster_size,
            "cluster_selection_method": args.cluster_selection_method,
            "metric": args.metric,
        }
        if args.min_samples is not None:
            kwargs["min_samples"] = args.min_samples
        return kwargs
    elif method == "kmeans":
        kwargs = {}
        if args.n_clusters is not None:
            kwargs["n_clusters"] = args.n_clusters
        return kwargs
    elif method == "dbscan":
        kwargs = {
            "eps": args.eps,
            "metric": args.metric,
        }
        if args.min_samples is not None:
            kwargs["min_samples"] = args.min_samples
        return kwargs
    elif method == "agglomerative":
        kwargs = {
            "linkage": args.linkage,
        }
        if args.n_clusters is not None:
            kwargs["n_clusters"] = args.n_clusters
        if args.linkage != "ward":
            kwargs["metric"] = args.metric
        return kwargs
    return {}


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

    from detection_toolkit.clustering import ClusterComputer

    dataset = _load_dataset(args.dataset)
    kwargs = _collect_method_kwargs(args)

    computer = ClusterComputer()
    labels = computer.cluster(
        dataset,
        input_field=args.input_field,
        output_field=args.output_field,
        method=args.method,
        overwrite=args.overwrite,
        **kwargs,
    )

    n_samples = len(labels)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())

    print(f"Clustering complete ({args.method}): {n_samples} samples")
    print(f"  clusters: {n_clusters}")
    print(f"  noise:    {n_noise}")
    if args.output_field is not None:
        print(f"  saved to: {args.output_field}")
    else:
        print("  (dry run — results not saved)")


if __name__ == "__main__":
    main()
