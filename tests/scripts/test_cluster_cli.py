"""Tests for dt-cluster CLI argument parsing."""


def test_cluster_parse_defaults():
    from detection_toolkit.scripts.cluster import parse_args
    args = parse_args(["my_ds", "--input-field", "emb"])
    assert args.dataset == "my_ds"
    assert args.input_field == "emb"
    assert args.output_field is None
    assert args.method == "hdbscan"
    assert args.overwrite is False


def test_cluster_parse_hdbscan():
    from detection_toolkit.scripts.cluster import parse_args
    args = parse_args([
        "ds", "--input-field", "emb", "--output-field", "cl",
        "--method", "hdbscan", "--min-cluster-size", "20",
        "--min-samples", "5", "--cluster-selection-method", "leaf",
    ])
    assert args.method == "hdbscan"
    assert args.min_cluster_size == 20
    assert args.min_samples == 5
    assert args.cluster_selection_method == "leaf"


def test_cluster_parse_kmeans():
    from detection_toolkit.scripts.cluster import parse_args
    args = parse_args([
        "ds", "--input-field", "emb", "--output-field", "cl",
        "--method", "kmeans", "--n-clusters", "10",
    ])
    assert args.method == "kmeans"
    assert args.n_clusters == 10


def test_cluster_parse_dbscan():
    from detection_toolkit.scripts.cluster import parse_args
    args = parse_args([
        "ds", "--input-field", "emb", "--output-field", "cl",
        "--method", "dbscan", "--eps", "0.3", "--min-samples", "10",
    ])
    assert args.method == "dbscan"
    assert args.eps == 0.3
    assert args.min_samples == 10


def test_cluster_parse_agglomerative():
    from detection_toolkit.scripts.cluster import parse_args
    args = parse_args([
        "ds", "--input-field", "emb", "--output-field", "cl",
        "--method", "agglomerative", "--n-clusters", "5",
        "--linkage", "complete", "--metric", "cosine",
    ])
    assert args.method == "agglomerative"
    assert args.n_clusters == 5
    assert args.linkage == "complete"
    assert args.metric == "cosine"
