"""Tests for dt-reduce CLI argument parsing."""

import pytest


def test_reduce_parse_defaults():
    from detection_toolkit.scripts.reduce import parse_args
    args = parse_args(["my_ds", "--input-field", "emb", "--output-field", "emb_pca"])
    assert args.dataset == "my_ds"
    assert args.input_field == "emb"
    assert args.output_field == "emb_pca"
    assert args.method == "umap"
    assert args.n_components == 32
    assert args.random_state == 42
    assert args.overwrite is False


def test_reduce_parse_custom():
    from detection_toolkit.scripts.reduce import parse_args
    args = parse_args([
        "ds", "--input-field", "emb", "--output-field", "out",
        "--method", "pca", "--n-components", "8", "--overwrite",
    ])
    assert args.method == "pca"
    assert args.n_components == 8
    assert args.overwrite is True


def test_reduce_parse_method_kwargs():
    from detection_toolkit.scripts.reduce import parse_args
    args = parse_args([
        "ds", "--input-field", "emb", "--output-field", "out",
        "--method", "umap", "--n-neighbors", "20", "--min-dist", "0.05",
        "--metric", "cosine",
    ])
    assert args.n_neighbors == 20
    assert args.min_dist == 0.05
    assert args.metric == "cosine"


def test_reduce_parse_tsne_kwargs():
    from detection_toolkit.scripts.reduce import parse_args
    args = parse_args([
        "ds", "--input-field", "emb", "--output-field", "out",
        "--method", "tsne", "--perplexity", "50",
    ])
    assert args.perplexity == 50
