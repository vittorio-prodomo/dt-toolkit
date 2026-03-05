"""ClusterComputer: apply clustering methods to FiftyOne embedding fields."""

import logging

import numpy as np

logger = logging.getLogger(__name__)

_METHODS = {"hdbscan", "kmeans", "dbscan", "agglomerative"}


class ClusterComputer:
    """Apply clustering to embedding fields on a FiftyOne dataset."""

    def cluster(
        self,
        dataset,
        input_field,
        output_field=None,
        method="hdbscan",
        overwrite=False,
        **method_kwargs,
    ):
        """Cluster embeddings and optionally store labels on the dataset.

        Args:
            dataset: FiftyOne Dataset or DatasetView.
            input_field: Name of the embedding field to read.
            output_field: Field to store labels. None for dry run.
            method: One of "hdbscan", "kmeans", "dbscan", "agglomerative".
            overwrite: Allow overwriting an existing field.
            **method_kwargs: Forwarded to the underlying algorithm.

        Returns:
            numpy array of integer cluster labels.
        """
        if method not in _METHODS:
            raise ValueError(
                f"Unknown clustering method '{method}'. "
                f"Choose from: {sorted(_METHODS)}"
            )

        if (
            output_field is not None
            and not overwrite
            and dataset.has_sample_field(output_field)
        ):
            raise ValueError(
                f"Field '{output_field}' already exists. "
                f"Use overwrite=True to replace it."
            )

        embeddings = np.array(dataset.values(input_field), dtype=np.float32)

        logger.info(
            "Clustering %d samples (%dd) via %s",
            len(embeddings), embeddings.shape[1], method,
        )

        model = self._build_model(method, **method_kwargs)
        model.fit(embeddings)
        labels = model.labels_

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int((labels == -1).sum())

        logger.info("Found %d clusters, %d noise samples", n_clusters, n_noise)

        # Print cluster distribution
        unique, counts = np.unique(labels[labels >= 0], return_counts=True)
        if len(unique) > 0:
            sorted_idx = np.argsort(-counts)
            top = ", ".join(
                f"#{unique[i]}:{counts[i]}" for i in sorted_idx[:10]
            )
            logger.info("Top clusters: %s", top)

        if output_field is not None:
            label_strs = [str(int(l)) for l in labels]
            dataset.set_values(output_field, label_strs)
            logger.info("Saved cluster labels to '%s'", output_field)

        return labels

    def _build_model(self, method, **kwargs):
        if method == "hdbscan":
            from sklearn.cluster import HDBSCAN
            return HDBSCAN(**kwargs)
        elif method == "kmeans":
            from sklearn.cluster import KMeans
            return KMeans(n_init="auto", **kwargs)
        elif method == "dbscan":
            from sklearn.cluster import DBSCAN
            return DBSCAN(**kwargs)
        elif method == "agglomerative":
            from sklearn.cluster import AgglomerativeClustering
            return AgglomerativeClustering(**kwargs)
