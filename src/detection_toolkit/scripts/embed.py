#!/usr/bin/env python3
"""Compute embeddings on FiftyOne datasets.

Usage::

    dt-embed my_dataset --backend zoo --model dinov2-vitb14-torch --output-field dinov2_emb
    dt-embed my_dataset --backend huggingface --model facebook/dinov3-vits16-pretrain-lvd1689m --output-field dinov3_emb
    dt-embed my_dataset --backend api --model Qwen/Qwen3-VL-Embedding-8B --output-field qwen_emb --api-url http://localhost:8001/v1
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
    p = make_parser("Compute embeddings on FiftyOne datasets.")
    p.add_argument("dataset", help="FiftyOne dataset name.")
    p.add_argument("--backend", "-b", required=True,
                   choices=["zoo", "huggingface", "api"],
                   help="Embedding backend.")
    p.add_argument("--model", "-m", required=True,
                   help="Model name or path (depends on backend).")
    p.add_argument("--output-field", "-o", required=True,
                   help="Field to store embeddings.")
    p.add_argument("--batch-size", type=int, default=32,
                   help="Batch size for inference (default: 32).")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite output field if it exists.")
    p.add_argument("--api-url", default=None,
                   help="API endpoint URL (for 'api' backend).")
    p.add_argument("--instruction", default=None,
                   help="Instruction prompt for VLMs (for 'api' backend).")
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

    from detection_toolkit.embedding import EmbeddingComputer

    dataset = _load_dataset(args.dataset)
    computer = EmbeddingComputer()
    computer.compute(
        dataset,
        backend=args.backend,
        model=args.model,
        output_field=args.output_field,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
        api_url=args.api_url,
        instruction=args.instruction,
    )

    print(f"Embeddings computed: backend={args.backend}, model={args.model}")
    print(f"  field: {args.output_field}")
    print(f"  samples: {len(dataset)}")


if __name__ == "__main__":
    main()
