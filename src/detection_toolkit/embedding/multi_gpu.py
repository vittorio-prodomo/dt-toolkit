"""Compute embeddings on several GPUs at once.

The dataset's samples are split into one contiguous shard per GPU; each shard
runs in its own spawned process that loads the model through the same backend
code as the single-GPU path (``backends.<name>.load_model``) and embeds its
whole shard with ``backends.<name>.embed`` — FiftyOne's ``compute_embeddings``
on a view. A multi-GPU run therefore writes exactly the vectors a single-GPU
run would.

It only pays off when the MODEL is the bottleneck. Measured 2026-09-26: 4K
drone JPEGs with dinov3-vits16 are bound by image decoding on the CPU, and a
second GPU brings no speed-up. The image-loader processes are therefore split
between the GPUs (same total as one GPU) so multi-GPU is never slower from
oversubscription.
"""

import importlib
import logging
import os
import time
import traceback

from .computer import _sanitize_field_name

logger = logging.getLogger(__name__)

LOCAL_BACKENDS = ("zoo", "huggingface")


def parse_gpus(spec):
    """'0,1' -> ['0', '1']. Rejects empty items, non-integers and repeats."""
    gpus = [g.strip() for g in str(spec).split(",")]
    if not all(g.isdigit() for g in gpus):
        raise ValueError(f"invalid GPU list {spec!r}: expected e.g. '0' or '0,1'")
    if len(set(gpus)) != len(gpus):
        raise ValueError(f"GPU listed twice in {spec!r}")
    return gpus


def shard_ids(ids, n):
    """Split ``ids`` into at most ``n`` contiguous, balanced, non-empty shards."""
    n = max(1, min(n, len(ids)))
    size, extra = divmod(len(ids), n)
    shards, start = [], 0
    for i in range(n):
        end = start + size + (1 if i < extra else 0)
        shards.append(list(ids[start:end]))
        start = end
    return [s for s in shards if s]


def loader_workers_per_gpu(n_gpus, cpu_count=None):
    """Image-loader processes per GPU worker: FiftyOne's single-run default
    (half the CPUs) split between the GPUs, so the total stays the same."""
    cpus = cpu_count if cpu_count is not None else (os.cpu_count() or 2)
    return max(1, (cpus // 2) // max(1, n_gpus))


def get_backend(name):
    return importlib.import_module(f"detection_toolkit.embedding.backends.{name}")


def _worker(gpu, dataset_name, ids, backend, model_name, output_field,
            batch_size, num_workers, errors):
    """One GPU: load the model once, embed the whole shard in one call."""
    try:
        import fiftyone as fo

        dataset = fo.load_dataset(dataset_name)
        impl = get_backend(backend)
        model = impl.load_model(model_name)
        impl.embed(dataset.select(ids), model, output_field,
                   batch_size=batch_size, num_workers=num_workers)
    except BaseException:
        errors.put((gpu, traceback.format_exc()))


def compute_multi_gpu(dataset, backend, model, output_field, gpus,
                      batch_size=None, overwrite=False, poll_seconds=2.0):
    """Embed ``dataset`` on ``gpus`` (list of device ids) in parallel."""
    import multiprocessing as mp

    if backend not in LOCAL_BACKENDS:
        raise ValueError(
            f"multi-GPU needs a local model backend {LOCAL_BACKENDS}; "
            f"'{backend}' sends images to a remote api and uses no local GPU")
    import fiftyone as fo
    from tqdm import tqdm

    output_field = _sanitize_field_name(output_field)
    if dataset.has_sample_field(output_field):
        if not overwrite:
            raise ValueError(
                f"Field '{output_field}' already exists. Use overwrite=True to replace it.")
        # cleared first so the progress count below only sees this run's vectors
        dataset.clear_sample_field(output_field)
    else:
        # declared once, here: workers adding a new field concurrently would
        # race on the dataset document's field list
        dataset.add_sample_field(output_field, fo.VectorField)

    shards = shard_ids(dataset.values("id"), len(gpus))
    per_gpu = loader_workers_per_gpu(len(shards))
    logger.info("Multi-GPU: %s; %d image loaders per GPU", ", ".join(
        f"GPU {g}: {len(s)}" for g, s in zip(gpus, shards)), per_gpu)

    ctx = mp.get_context("spawn")
    errors = ctx.Queue()
    procs = []
    threads = str(max(1, (os.cpu_count() or 1) // len(shards)))
    child_env = {"FIFTYONE_SHOW_PROGRESS_BARS": "false",
                 # each worker's own torch/OpenMP threads get its share of the CPU
                 "OMP_NUM_THREADS": threads, "MKL_NUM_THREADS": threads}
    saved = {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", *child_env)}
    try:
        for gpu, shard in zip(gpus, shards):
            # a spawned child inherits the environment at start(): set its GPU
            # before any of its imports can initialise CUDA
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu
            os.environ.update(child_env)
            p = ctx.Process(target=_worker, args=(
                gpu, dataset.name, shard, backend, model, output_field,
                batch_size, per_gpu, errors))
            p.start()
            procs.append(p)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    total = sum(len(s) for s in shards)
    done = lambda: dataset.exists(output_field).count()  # noqa: E731
    try:
        with tqdm(total=total, desc="Computing embeddings", unit="samples") as bar:
            while any(p.is_alive() for p in procs) and errors.empty():
                bar.update(done() - bar.n)
                time.sleep(poll_seconds)
            bar.update(done() - bar.n)
        for p in procs:
            p.join(timeout=5 if not errors.empty() else None)
    except KeyboardInterrupt:
        _kill(procs)
        raise
    if not errors.empty():
        _kill(procs)
        gpu, tb = errors.get()
        raise RuntimeError(f"embedding worker on GPU {gpu} failed:\n{tb}")
    failed = [(g, p.exitcode) for g, p in zip(gpus, procs) if p.exitcode != 0]
    if failed:
        raise RuntimeError(f"embedding workers exited abnormally: {failed}")
    dataset.reload()
    return total


def _kill(procs):
    for p in procs:
        if p.is_alive():
            p.kill()
    for p in procs:
        p.join(timeout=5)
