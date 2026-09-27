"""Tests for dt-embed CLI argument parsing."""

import pytest


def test_embed_parse_zoo():
    from detection_toolkit.scripts.embed import parse_args
    args = parse_args([
        "my_ds", "--backend", "zoo",
        "--model", "dinov2-vitb14-torch",
        "--output-field", "dinov2_emb",
    ])
    assert args.dataset == "my_ds"
    assert args.backend == "zoo"
    assert args.model == "dinov2-vitb14-torch"
    assert args.output_field == "dinov2_emb"
    assert args.batch_size == 32
    assert args.overwrite is False


def test_embed_parse_huggingface():
    from detection_toolkit.scripts.embed import parse_args
    args = parse_args([
        "ds", "--backend", "huggingface",
        "--model", "facebook/dinov3-vits16-pretrain-lvd1689m",
        "--output-field", "dinov3_emb",
        "--batch-size", "16", "--overwrite",
    ])
    assert args.backend == "huggingface"
    assert args.model == "facebook/dinov3-vits16-pretrain-lvd1689m"
    assert args.batch_size == 16
    assert args.overwrite is True


def test_embed_parse_api():
    from detection_toolkit.scripts.embed import parse_args
    args = parse_args([
        "ds", "--backend", "api",
        "--model", "Qwen/Qwen3-VL-Embedding-8B",
        "--output-field", "qwen_emb",
        "--api-url", "http://localhost:8001/v1",
        "--instruction", "Represent the image.",
    ])
    assert args.backend == "api"
    assert args.api_url == "http://localhost:8001/v1"
    assert args.instruction == "Represent the image."


def test_embed_parse_defaults():
    from detection_toolkit.scripts.embed import parse_args
    args = parse_args([
        "ds", "--backend", "zoo",
        "--model", "clip-vit-base32-torch",
        "--output-field", "clip_emb",
    ])
    assert args.api_url is None
    assert args.instruction is None
    assert args.batch_size == 32


def test_embed_rejects_a_bad_gpu_list_before_touching_the_dataset(capsys):
    import pytest
    from detection_toolkit.scripts.embed import main

    with pytest.raises(SystemExit) as exc:
        main(["ds", "--backend", "zoo", "--model", "m", "-o", "f", "--gpu", "0,0"])
    assert exc.value.code == 2
    assert "twice" in capsys.readouterr().err
