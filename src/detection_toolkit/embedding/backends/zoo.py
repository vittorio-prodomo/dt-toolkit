"""FiftyOne Model Zoo embedding backend."""

import logging

logger = logging.getLogger(__name__)


def load_model(model_name):
    import fiftyone.zoo as foz

    logger.info("Loading zoo model: %s", model_name)
    return foz.load_zoo_model(model_name)


def embed(dataset, model, output_field, batch_size=None, num_workers=None):
    logger.info("Computing embeddings for %d samples", len(dataset))
    kwargs = {}
    if batch_size is not None:
        kwargs["batch_size"] = batch_size
    if num_workers is not None:
        kwargs["num_workers"] = num_workers

    dataset.compute_embeddings(
        model,
        embeddings_field=output_field,
        **kwargs,
    )


def compute(dataset, model_name, output_field, batch_size=None):
    embed(dataset, load_model(model_name), output_field, batch_size=batch_size)
