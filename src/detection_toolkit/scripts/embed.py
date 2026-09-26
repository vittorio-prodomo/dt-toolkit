#!/usr/bin/env python3
"""Compute embeddings on FiftyOne datasets.

Usage::

    dt-embed my_dataset --backend zoo --model dinov2-vitb14-torch --output-field dinov2_emb
    dt-embed my_dataset --backend huggingface --model facebook/dinov3-vits16-pretrain-lvd1689m --output-field dinov3_emb
    dt-embed my_dataset --backend api --model Qwen/Qwen3-VL-Embedding-8B --output-field qwen_emb --api-url http://localhost:8001/v1

    # Multi-GPU: split dataset across GPUs
    dt-embed my_dataset --backend huggingface --model facebook/dinov3-vits16-pretrain-lvd1689m --output-field dinov3_emb --gpu 0,1
"""

import logging
import os
import sys
import traceback

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
    p.add_argument("--gpu", default=None, metavar="IDs",
                   help="GPU device ID(s) to use. Single (e.g. '0') or "
                        "comma-separated for multi-GPU (e.g. '0,1'). "
                        "Multi-GPU splits the dataset across GPUs.")
    _add_verbosity(p)
    return p.parse_args(argv)


def _load_dataset(name):
    import fiftyone as fo
    try:
        return fo.load_dataset(name)
    except ValueError:
        existing = fo.list_datasets()
        print("Error: Dataset '{}' not found.".format(name), file=sys.stderr)
        if existing:
            print("Available datasets: {}".format(existing), file=sys.stderr)
        sys.exit(1)


def _run_single(args):
    """Run embedding computation on a single GPU (original path)."""
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

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
    return len(dataset)


def _worker(gpu_id, dataset_name, sample_ids, backend, model_name,
            output_field, batch_size, counter, error_queue,
            api_url, instruction):
    """Worker process: compute embeddings for a shard on a specific GPU.

    Updates ``counter`` incrementally (per batch) so the main process can
    show a live progress bar.  Errors are put on ``error_queue`` instead of
    crashing silently.
    """
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        os.environ["FIFTYONE_SHOW_PROGRESS_BARS"] = "false"

        import numpy as np
        import torch
        from PIL import Image

        import fiftyone as fo

        dataset = fo.load_dataset(dataset_name)
        view = dataset.select(sample_ids)

        # --- Load model -------------------------------------------------------
        if backend == "huggingface":
            from transformers import AutoImageProcessor, AutoModel
            processor = AutoImageProcessor.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            model = model.to("cuda")
            model.requires_grad_(False)

        elif backend == "zoo":
            import fiftyone.zoo as foz
            model = foz.load_zoo_model(model_name)
            view.compute_embeddings(
                model, embeddings_field=output_field, batch_size=batch_size,
                progress=False,
            )
            with counter.get_lock():
                counter.value += len(sample_ids)
            return  # zoo backend handles everything internally

        elif backend == "api":
            from detection_toolkit.embedding.backends.api import compute as api_compute
            api_compute(
                view, model_name, output_field,
                api_url=api_url, batch_size=batch_size, instruction=instruction,
            )
            with counter.get_lock():
                counter.value += len(sample_ids)
            return

        # --- Manual batched inference (huggingface) ---------------------------
        sample_list = list(view)
        for batch_start in range(0, len(sample_list), batch_size):
            batch_samples = sample_list[batch_start:batch_start + batch_size]
            images = []
            valid_samples = []
            for sample in batch_samples:
                try:
                    img = Image.open(sample.filepath).convert("RGB")
                    images.append(img)
                    valid_samples.append(sample)
                except Exception:
                    pass

            if not images:
                with counter.get_lock():
                    counter.value += len(batch_samples)
                continue

            inputs = processor(images=images, return_tensors="pt")
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            # Use CLS token embedding.
            embeddings = outputs.last_hidden_state[:, 0].cpu().numpy()

            for sample, emb in zip(valid_samples, embeddings):
                sample[output_field] = emb.tolist()
                sample.save()

            with counter.get_lock():
                counter.value += len(batch_samples)

    except Exception:
        error_queue.put((gpu_id, traceback.format_exc()))


def _run_multi(args, gpu_ids):
    """Run embedding computation across multiple GPUs."""
    import multiprocessing as mp
    import signal
    import time

    from tqdm import tqdm

    dataset = _load_dataset(args.dataset)

    if not args.overwrite and dataset.has_sample_field(args.output_field):
        print("Error: Field '{}' already exists. "
              "Use --overwrite to replace it.".format(args.output_field),
              file=sys.stderr)
        sys.exit(1)

    all_ids = dataset.values("id")
    n = len(all_ids)
    n_gpus = len(gpu_ids)

    # Shard sample IDs into contiguous chunks for better cache locality.
    chunk_size = (n + n_gpus - 1) // n_gpus
    shards = [all_ids[i * chunk_size:(i + 1) * chunk_size] for i in range(n_gpus)]

    print("Multi-GPU: splitting {} samples across {} GPUs ({})".format(
        n, n_gpus,
        ", ".join("GPU {}: {}".format(g, len(shards[i]))
                  for i, g in enumerate(gpu_ids))))

    ctx = mp.get_context("spawn")
    counter = ctx.Value("i", 0)
    error_queue = ctx.Queue()

    processes = []
    for rank, gpu_id in enumerate(gpu_ids):
        p = ctx.Process(
            target=_worker,
            args=(gpu_id, args.dataset, shards[rank],
                  args.backend, args.model, args.output_field,
                  args.batch_size, counter, error_queue,
                  args.api_url, args.instruction),
        )
        p.start()
        processes.append(p)

    def _kill_workers(procs):
        for p in procs:
            if p.is_alive():
                p.kill()
        for p in procs:
            p.join(timeout=5)

    # Handle Ctrl+C gracefully.
    original_sigint = signal.getsignal(signal.SIGINT)

    def _sigint_handler(signum, frame):
        print("\nInterrupted -- killing workers ...", file=sys.stderr)
        _kill_workers(processes)
        signal.signal(signal.SIGINT, original_sigint)
        sys.exit(1)

    signal.signal(signal.SIGINT, _sigint_handler)

    try:
        with tqdm(total=n, desc="Computing embeddings", unit="samples") as pbar:
            while any(p.is_alive() for p in processes):
                with counter.get_lock():
                    done = counter.value
                delta = done - pbar.n
                if delta > 0:
                    pbar.update(delta)

                # Check for worker errors.
                if not error_queue.empty():
                    gpu_id, tb = error_queue.get_nowait()
                    print("\nError in worker GPU {}:\n{}".format(gpu_id, tb),
                          file=sys.stderr)
                    _kill_workers(processes)
                    sys.exit(1)

                time.sleep(0.3)

            # Final update.
            with counter.get_lock():
                done = counter.value
            delta = done - pbar.n
            if delta > 0:
                pbar.update(delta)

        # Check for any remaining errors after workers exit.
        if not error_queue.empty():
            gpu_id, tb = error_queue.get_nowait()
            print("\nError in worker GPU {}:\n{}".format(gpu_id, tb),
                  file=sys.stderr)
            sys.exit(1)

        for p in processes:
            p.join()

        # Check exit codes.
        for i, p in enumerate(processes):
            if p.exitcode != 0:
                print("Error: Worker for GPU {} exited with code {}".format(
                    gpu_ids[i], p.exitcode), file=sys.stderr)
                sys.exit(1)

    except KeyboardInterrupt:
        _kill_workers(processes)
        sys.exit(1)
    finally:
        signal.signal(signal.SIGINT, original_sigint)

    return n


def main(argv=None):
    args = parse_args(argv)

    if args.quiet:
        level = logging.WARNING
    elif args.verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(level=level, format="%(name)s - %(levelname)s - %(message)s")

    # Parse GPU list.
    if args.gpu and "," in args.gpu:
        gpu_ids = [g.strip() for g in args.gpu.split(",")]
        n_samples = _run_multi(args, gpu_ids)
    else:
        n_samples = _run_single(args)

    print("Embeddings computed: backend={}, model={}".format(
        args.backend, args.model))
    print("  field: {}".format(args.output_field))
    print("  samples: {}".format(n_samples))


if __name__ == "__main__":
    main()
