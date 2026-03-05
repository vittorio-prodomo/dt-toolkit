"""OpenAI-compatible HTTP API embedding backend.

Works with vLLM, SGLang, Ollama, or any server implementing the
OpenAI /v1/embeddings endpoint.
"""

import base64
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _encode_image_base64(image_path):
    """Read an image file and return its base64-encoded string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _get_embeddings_batch(filepaths, model, api_url, instruction=None):
    """Send images to the embeddings endpoint one at a time."""
    import requests

    embeddings = []
    for fp in filepaths:
        b64 = _encode_image_base64(fp)
        content = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            },
        ]
        if instruction:
            content.append({"type": "text", "text": instruction})

        response = requests.post(
            f"{api_url}/embeddings",
            json={
                "model": model,
                "messages": [{"role": "user", "content": content}],
                "encoding_format": "float",
            },
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        embeddings.append(data["data"][0]["embedding"])

    return embeddings


def compute(dataset, model, output_field, api_url, batch_size=32, instruction=None):
    """Compute embeddings via an OpenAI-compatible API."""
    filepaths = dataset.values("filepath")
    n = len(filepaths)

    logger.info(
        "Computing embeddings via API: model=%s, url=%s, samples=%d",
        model, api_url, n,
    )

    all_embeddings = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = filepaths[start:end]
        batch_embs = _get_embeddings_batch(
            batch, model, api_url, instruction=instruction,
        )
        all_embeddings.extend(batch_embs)
        logger.info("Progress: %d/%d", end, n)

    dataset.set_values(output_field, all_embeddings)
