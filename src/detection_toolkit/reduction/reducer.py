"""DimensionalityReducer: reduce embedding dimensions on FiftyOne datasets."""

import logging

import numpy as np

logger = logging.getLogger(__name__)

_SUPPORTED_METHODS = {"pca", "tsne", "umap", "pacmap"}


class DimensionalityReducer:
    """Reduce high-dimensional embeddings stored on a FiftyOne dataset.

    Usage::

        reducer = DimensionalityReducer()
        result = reducer.reduce(dataset, "clip_embedding", "clip_reduced",
                                method="pca", n_components=8)
    """

    def reduce(
        self,
        dataset,
        input_field,
        output_field,
        method="umap",
        n_components=32,
        random_state=42,
        overwrite=False,
        **method_kwargs,
    ):
        """Reduce embeddings and store on the dataset.

        Args:
            dataset: FiftyOne Dataset or DatasetView.
            input_field: Name of the embedding field to read.
            output_field: Name of the field to store reduced embeddings.
            method: Reduction algorithm — one of "pca", "tsne", "umap", "pacmap".
            n_components: Target dimensionality.
            random_state: Random seed for reproducibility.
            overwrite: If False, raise when output_field already exists.
            **method_kwargs: Extra keyword arguments forwarded to the model.

        Returns:
            numpy array of shape (n_samples, n_components).
            Rows with missing input embeddings contain NaN.
        """
        # --- Validate method ---
        if method not in _SUPPORTED_METHODS:
            raise ValueError(
                f"Unknown reduction method '{method}'. "
                f"Supported: {sorted(_SUPPORTED_METHODS)}"
            )

        # --- Overwrite protection ---
        if not overwrite and dataset.has_sample_field(output_field):
            raise ValueError(
                f"Output field '{output_field}' already exists. "
                f"Set overwrite=True to replace it."
            )

        # --- Read embeddings and track None indices ---
        raw_embeddings = dataset.values(input_field)
        n_total = len(raw_embeddings)

        valid_indices = []
        valid_embeddings = []
        for i, emb in enumerate(raw_embeddings):
            if emb is not None:
                valid_indices.append(i)
                valid_embeddings.append(emb)

        logger.info(
            "Reducing %d/%d valid embeddings: method=%s, n_components=%d",
            len(valid_embeddings), n_total, method, n_components,
        )

        # --- Handle all-None edge case ---
        full_result = np.full((n_total, n_components), np.nan, dtype=np.float32)
        if not valid_embeddings:
            logger.warning("No valid embeddings found; returning all-NaN result.")
            dataset.set_values(output_field, [None] * n_total)
            return full_result

        # --- Build model and fit_transform ---
        model = self._build_model(method, n_components, random_state, **method_kwargs)
        valid_array = np.asarray(valid_embeddings, dtype=np.float32)
        reduced_valid = model.fit_transform(valid_array)

        # --- Reconstruct full output ---
        full_result[valid_indices] = reduced_valid

        # --- Store on dataset (use index membership, not NaN check) ---
        valid_set = set(valid_indices)
        stored_values = [
            full_result[i] if i in valid_set else None
            for i in range(n_total)
        ]
        dataset.set_values(output_field, stored_values)

        logger.info(
            "Stored reduced embeddings in '%s' (%d valid, %d total)",
            output_field, len(valid_embeddings), n_total,
        )

        return full_result

    def _build_model(self, method, n_components, random_state, **kwargs):
        """Construct the dimensionality-reduction model.

        Args:
            method: One of "pca", "tsne", "umap", "pacmap".
            n_components: Target dimensionality.
            random_state: Random seed.
            **kwargs: Extra keyword arguments for the model constructor.

        Returns:
            A fitted model with a ``fit_transform`` method.
        """
        if method == "pca":
            from sklearn.decomposition import PCA
            return PCA(n_components=n_components, random_state=random_state, **kwargs)

        if method == "tsne":
            from sklearn.manifold import TSNE
            return TSNE(n_components=n_components, random_state=random_state, **kwargs)

        if method == "umap":
            try:
                import umap  # noqa: F811
            except ImportError as exc:
                raise ImportError(
                    "UMAP requires the 'umap-learn' package. "
                    "Install it with: pip install umap-learn"
                ) from exc
            return umap.UMAP(
                n_components=n_components, random_state=random_state, **kwargs
            )

        if method == "pacmap":
            try:
                import pacmap  # noqa: F811
            except ImportError as exc:
                raise ImportError(
                    "PaCMAP requires the 'pacmap' package. "
                    "Install it with: pip install pacmap"
                ) from exc
            return pacmap.PaCMAP(
                n_components=n_components, random_state=random_state, **kwargs
            )

        # Should not reach here because of validation, but just in case
        raise ValueError(f"Unknown reduction method '{method}'")
