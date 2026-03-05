"""FiftyOne Model Zoo embedding backend."""

import logging

logger = logging.getLogger(__name__)


def compute(dataset, model_name, output_field, batch_size=None):
    import fiftyone.zoo as foz

    logger.info("Loading zoo model: %s", model_name)
    model = foz.load_zoo_model(model_name)

    logger.info("Computing embeddings for %d samples", len(dataset))
    kwargs = {}
    if batch_size is not None:
        kwargs["batch_size"] = batch_size

    dataset.compute_embeddings(
        model,
        embeddings_field=output_field,
        **kwargs,
    )
