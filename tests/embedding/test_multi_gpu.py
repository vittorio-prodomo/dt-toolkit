"""Multi-GPU embedding: GPU parsing, sharding, the worker loop, and parity.

The worker must run the SAME backend code as the single-GPU path (load the
model through the backend, then FiftyOne's compute_embeddings on a view), so
a multi-GPU run can never write a different kind of vector into the field.
"""
import os
import uuid

import numpy as np
import pytest

from detection_toolkit.embedding.multi_gpu import (
    compute_multi_gpu,
    loader_workers_per_gpu,
    parse_gpus,
    shard_ids,
)


def test_parse_gpus():
    assert parse_gpus("0") == ["0"]
    assert parse_gpus("0,1") == ["0", "1"]
    assert parse_gpus(" 1 , 0 ") == ["1", "0"]
    for bad in ("", "0,,1", "a", "0,0", "-1"):
        with pytest.raises(ValueError):
            parse_gpus(bad)


def test_shard_ids_is_contiguous_balanced_and_complete():
    ids = [str(i) for i in range(10)]
    shards = shard_ids(ids, 3)
    assert [len(s) for s in shards] == [4, 3, 3]
    assert sum(shards, []) == ids
    assert shard_ids(ids[:1], 3) == [["0"]]            # never an empty shard


def test_loader_workers_are_split_between_gpus():
    # FiftyOne's default for one run is half the CPUs; two GPU workers must
    # not each take that much (measured: 4K decoding is the bottleneck)
    assert loader_workers_per_gpu(1, cpu_count=128) == 64
    assert loader_workers_per_gpu(2, cpu_count=128) == 32
    assert loader_workers_per_gpu(8, cpu_count=4) == 1


def test_api_backend_is_rejected_for_multi_gpu():
    with pytest.raises(ValueError, match="api"):
        compute_multi_gpu(None, "api", "m", "emb", gpus=["0", "1"])


# ── worker loop, in-process, with a fake model ────────────────────────


@pytest.fixture
def fo_dataset(tmp_path):
    import fiftyone as fo
    from PIL import Image

    ds = fo.Dataset(f"dt-embed-test-{uuid.uuid4().hex[:8]}")
    for i in range(5):
        p = tmp_path / f"{i}.png"
        Image.new("RGB", (8, 8), (i * 40, 0, 0)).save(p)
        ds.add_sample(fo.Sample(filepath=str(p)))
    yield ds
    ds.delete()


class _FakeModel:
    loads = 0


class _FakeBackend:
    """Stands in for backends.zoo/huggingface: load once, embed per view."""

    def __init__(self):
        self.embedded = []

    def load_model(self, name):
        _FakeModel.loads += 1
        return _FakeModel()

    def embed(self, view, model, output_field, batch_size=None, num_workers=None):
        self.num_workers = num_workers
        ids = view.values("id")
        self.embedded.append(len(ids))
        view.set_values(output_field, {i: np.full(3, len(ids), float) for i in ids},
                        key_field="id")


@pytest.mark.requires_mongod
def test_worker_loads_once_and_embeds_its_whole_shard_in_one_call(fo_dataset, monkeypatch):
    import multiprocessing as mp
    from detection_toolkit.embedding import multi_gpu as M

    fake = _FakeBackend()
    monkeypatch.setattr(M, "get_backend", lambda name: fake)
    _FakeModel.loads = 0
    errors = mp.Queue()
    ids = fo_dataset.values("id")

    M._worker("0", fo_dataset.name, ids, "zoo", "m", "emb", None, 3, errors)

    assert errors.empty()
    assert _FakeModel.loads == 1
    # one compute call per shard: each call starts its own image-loader pool,
    # and per-chunk calls measured 2.5x slower
    assert fake.embedded == [5] and fake.num_workers == 3
    assert all(v is not None for v in fo_dataset.values("emb"))


# ── real parity on two GPUs (opt-in: DT_EMBED_GPU_TESTS=1) ────────────


@pytest.mark.requires_mongod
@pytest.mark.skipif(os.environ.get("DT_EMBED_GPU_TESTS") != "1",
                    reason="set DT_EMBED_GPU_TESTS=1 (needs 2 GPUs + the model in the HF cache)")
def test_two_gpus_write_the_same_vectors_as_one(fo_dataset):
    from detection_toolkit.embedding import EmbeddingComputer

    model = os.environ.get("DT_EMBED_TEST_MODEL", "facebook/dinov3-vits16-pretrain-lvd1689m")
    EmbeddingComputer().compute(fo_dataset, "huggingface", model, "single", batch_size=2)
    compute_multi_gpu(fo_dataset, "huggingface", model, "multi", gpus=["0", "1"],
                      batch_size=2, poll_seconds=0.2)
    single = np.stack(fo_dataset.values("single"))
    multi = np.stack(fo_dataset.values("multi"))
    assert single.shape == multi.shape
    np.testing.assert_allclose(multi, single, rtol=1e-4, atol=1e-4)
