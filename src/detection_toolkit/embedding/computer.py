"""EmbeddingComputer: compute embeddings on FiftyOne datasets via pluggable backends."""

import logging

from .backends import _BACKENDS

logger = logging.getLogger(__name__)


class EmbeddingComputer:
    def compute(
        self,
        dataset,
        backend,
        model,
        output_field,
        batch_size=32,
        overwrite=False,
        api_url=None,
        instruction=None,
    ):
        if backend not in _BACKENDS:
            raise ValueError(
                f"Unknown embedding backend '{backend}'. "
                f"Choose from: {sorted(_BACKENDS)}"
            )

        if not overwrite and dataset.has_sample_field(output_field):
            raise ValueError(
                f"Field '{output_field}' already exists. "
                f"Use overwrite=True to replace it."
            )

        logger.info(
            "Computing embeddings: backend=%s, model=%s, field=%s",
            backend,
            model,
            output_field,
        )

        if backend == "zoo":
            from .backends.zoo import compute as zoo_compute

            zoo_compute(dataset, model, output_field, batch_size=batch_size)

        elif backend == "huggingface":
            from .backends.huggingface import compute as hf_compute

            hf_compute(dataset, model, output_field, batch_size=batch_size)

        elif backend == "api":
            from .backends.api import compute as api_compute

            if api_url is None:
                raise ValueError("--api-url is required for the 'api' backend.")
            api_compute(
                dataset,
                model,
                output_field,
                api_url=api_url,
                batch_size=batch_size,
                instruction=instruction,
            )

        logger.info("Embeddings stored in '%s'", output_field)
