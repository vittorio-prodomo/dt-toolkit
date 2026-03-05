"""HuggingFace transformers embedding backend."""

import logging

logger = logging.getLogger(__name__)


def compute(dataset, model_name, output_field, batch_size=None):
    """Compute embeddings using a HuggingFace transformers model.

    Loads the model with AutoModel.from_pretrained() and delegates to
    FiftyOne's dataset.compute_embeddings().
    """
    try:
        from transformers import AutoModel
    except ImportError:
        raise ImportError(
            "transformers is required for the 'huggingface' backend. "
            "Install it with: pip install detection-toolkit[embed]"
        )

    logger.info("Loading HuggingFace model: %s", model_name)
    model = AutoModel.from_pretrained(model_name)

    logger.info("Computing embeddings for %d samples", len(dataset))
    kwargs = {}
    if batch_size is not None:
        kwargs["batch_size"] = batch_size

    dataset.compute_embeddings(
        model, embeddings_field=output_field, **kwargs,
    )
