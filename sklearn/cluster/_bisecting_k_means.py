
from cProfile import label
import numpy as np
import scipy.sparse as sp
from ..base import (
    _ClassNamePrefixFeaturesOutMixin,
    BaseEstimator,
    ClusterMixin,
    TransformerMixin
)

from ..utils._openmp_helpers import _openmp_effective_n_threads
from ..utils.validation import _check_sample_weight
from ._kmeans import KMeans
from ._k_means_common import _inertia_dense
from ._k_means_common import _inertia_sparse

class BisectingKMeans(
    _ClassNamePrefixFeaturesOutMixin, TransformerMixin, ClusterMixin, BaseEstimator
):

    def __init__(
        self,
        n_clusters=8,
        *,
        init="k-means++",
        n_init=10,
        max_iter=300,
        tol=1e-4,
        verbose=0,
        random_state=None,
        copy_x=True,
        algorithm="lloyd",
    ):

        self.n_split = 2
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = KMeans(
            n_clusters=self.n_split,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            random_state = self.random_state,
            copy_x=copy_x,
            algorithm=algorithm
            )
        self._n_threads = _openmp_effective_n_threads()
        self.max_iter = max_iter
        pass

    def fit(self, X, y=None, sample_weight=None):
        """Compute bisecting k-means.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training instances to cluster. It must be noted that the data
            will be converted to C ordering, which will cause a memory
            copy if the given data is not C-contiguous.
            If a sparse matrix is passed, a copy will be made if it's not in
            CSR format.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

        Returns
        ---
        self: object
            Fitted estimator.
        """
        # Initial split of data.
        kmeans_bisect = self.kmeans.fit(X)
        sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)

        # Keep track of all clusters. Update after each split. Pick out cluster with highest SSE for splitting.
        all_clusters = self._split_cluster_points(
            X, sample_weight=sample_weight, centers=kmeans_bisect.cluster_centers_, labels=kmeans_bisect.labels_)
        
        self.cluster_centers_ = kmeans_bisect.cluster_centers_
        self._n_features_out = kmeans_bisect.cluster_centers_.shape[0]
        self.labels_ = kmeans_bisect.labels_

        while self.cluster_centers_.shape[0] < self.n_clusters:
            # Select cluster with highest SSE
            max_sse_idx = np.argmax([c["inertia"] for c in all_clusters])
            selected_cluster = all_clusters[max_sse_idx]

            # Performs kmeans (k=2), on the selected cluster.
            # Replace the old cluster (selected_cluster) with the clusters obtained
            # from kmeans 2. This way, we keep track of all clusters. Both the ones
            # obtained from splitting, and the old ones that didn't qualify for splitting.
            kmeans_bisect = self.kmeans.fit(selected_cluster["X"])
            all_clusters[max_sse_idx:max_sse_idx] = self._split_cluster_points(
                selected_cluster["X"],
                sample_weight=selected_cluster["sample_weight"],
                centers=kmeans_bisect.cluster_centers_,
                labels=kmeans_bisect.labels_
            )
            
            # Update cluster_centers_. Replace cluster center of max sse in self.cluster_centers_
            # with new centers obtained from performing kmeans 2.
            max_sse_cluster_idx = np.where(
                np.all(self.cluster_centers_ == selected_cluster["centers"], axis=1))[0][0]
            # Remove old center
            self.cluster_centers_ = np.delete(self.cluster_centers_, max_sse_cluster_idx, axis=0)
            # Insert new center in place of old one
            self.cluster_centers_ = np.insert(
                self.cluster_centers_, max_sse_cluster_idx, kmeans_bisect.cluster_centers_, axis=0)

            # Update labels_. Replace labels of max sse in self.labels_ with new labels
            # obtained from performing kmeans 2.
            # Update labels to correspond to the indices of updated self.cluster_centers_
            max_sse_labels_idxs = np.where(self.labels_ == max_sse_cluster_idx)[0]
            self.labels_[max_sse_labels_idxs] = kmeans_bisect.labels_ + max_sse_cluster_idx
            
            self._n_features_out = self.cluster_centers_.shape[0]

        return self

    def fit_predict(X, y=None, sample_weight=None):
        pass

    def fit_transform(X, y=None, sample_weight=None):
        pass
    
    def get_params(deep=True):
        pass

    def set_params(**params):
        pass

    def score(X, y=None, sample_weight=None):
        pass

    def transform(X):
        pass

    def predict(X, sample_weight=None):
        pass

    def _split_cluster_points(self, X, sample_weight, centers, labels):
        """
        Returns
        -------
        split X with numpy, call _inertia_dense
        split X manually (for i in range), calculating inertia as we go
        (unique lables + 1) * n
        """
        split_clusters = []
        for i in np.unique(labels):
            cluster_data={}
            cluster_data["X"] = np.array(X[labels == i], dtype=np.float64) # Had to specify dtype, otherwise _inertia gives error.
            cluster_data["sample_weight"] = sample_weight[labels == i]
            cluster_data["centers"] = np.reshape(centers[i], (1, -1)) # Reshape 1D array to 2D: (1, 1).
            cluster_data["labels"] = np.full(cluster_data["X"].shape[0], i) # Every datapoint in X is labeled current label.
            if sp.issparse(cluster_data["X"]):
                _inertia = _inertia_sparse
            else:
                _inertia = _inertia_dense
            cluster_data["inertia"] = _inertia(cluster_data["X"], cluster_data["sample_weight"], cluster_data["centers"], cluster_data["labels"], self._n_threads)
            split_clusters.append(cluster_data)
        return split_clusters

        
