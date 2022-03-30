
import numpy as np
import scipy.sparse as sp

from ..utils._openmp_helpers import _openmp_effective_n_threads
from ._kmeans import KMeans
from ._k_means_common import _inertia_dense
from ._k_means_common import _inertia_sparse

class BisectingKMeans():

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
        pass

    def fit(X, y=None, sample_weight=None):
        pass

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
            cluster_data["label"] = i
            cluster_data["X"] = X[labels == i]
            cluster_data["sample_weight"] = sample_weight[labels == i]
            cluster_data["center"] = centers[i]
            if sp.issparse(cluster_data["X"]):
                _inertia = _inertia_sparse
            else:
                _inertia = _inertia_dense
            cluster_data["inertia"] = _inertia(cluster_data["X"], cluster_data["sample_weight"], [cluster_data["center"]], np.zeros(shape=cluster_data["X"].shape[0]), self._n_threads)
            split_clusters.append(cluster_data)
        return split_clusters

        
