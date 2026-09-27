"""HuggingFace transformers embedding backend."""

import logging

logger = logging.getLogger(__name__)


def load_model(model_name):
    """Load a HuggingFace model with AutoModel.from_pretrained()."""
    try:
        from transformers import AutoModel
    except ImportError:
        raise ImportError(
            "transformers is required for the 'huggingface' backend. "
            "Install it with: pip install detection-toolkit[embed]"
        )

    logger.info("Loading HuggingFace model: %s", model_name)
    return AutoModel.from_pretrained(model_name)


def embed(dataset, model, output_field, batch_size=None, num_workers=None):
    """Delegate to FiftyOne's compute_embeddings() with a loaded model."""
    logger.info("Computing embeddings for %d samples", len(dataset))
    kwargs = {}
    if batch_size is not None:
        kwargs["batch_size"] = batch_size
    if num_workers is not None:
        kwargs["num_workers"] = num_workers

    dataset.compute_embeddings(
        model, embeddings_field=output_field, **kwargs,
    )


def compute(dataset, model_name, output_field, batch_size=None):
    """Compute embeddings using a HuggingFace transformers model.

    Loads the model with AutoModel.from_pretrained() and delegates to
    FiftyOne's dataset.compute_embeddings().
    """
    embed(dataset, load_model(model_name), output_field, batch_size=batch_size)
