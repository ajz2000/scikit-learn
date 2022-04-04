"""Bisecting K-means clustering."""

# Authors: Zining (Jenny) Yu <zi.yu@mail.utoronto.ca>
#          Jingrun Long <jingrun.long@mail.utoronto.ca>
#          Aidan Zorbas <aidan.zorbas@mail.utoronto.ca>
#          Dawson Brown <dawson.brown@mail.utoronto.ca>
#          Kara Autumn Jiang <autumn.jiang@mail.utoronto.ca>
#          Vanessa Pierre <vanessa.pierre@mail.utoronto.ca>

import numpy as np
import scipy.sparse as sp
import warnings

from ..base import (
    _ClassNamePrefixFeaturesOutMixin,
    BaseEstimator,
    ClusterMixin,
    TransformerMixin
)

from ..metrics.pairwise import euclidean_distances
from ..utils._openmp_helpers import _openmp_effective_n_threads
from ..utils.extmath import row_norms
from ..utils.validation import _check_sample_weight, check_is_fitted
from ._kmeans import KMeans, check_is_fitted, _labels_inertia_threadpool_limit
from ._k_means_common import _inertia_dense
from ._k_means_common import _inertia_sparse


class BisectingKMeans(
    _ClassNamePrefixFeaturesOutMixin, TransformerMixin, ClusterMixin, BaseEstimator
):
    """Bisecting K-Means clustering.

    Parameters
    ----------

    n_clusters : int, default=8
        The number of clusters to form. Equivalent to the number of bisection steps - 1.

    init : {'k-means++', 'random'}, default='k-means++'
        Method for initialization of the internal K-Means algorithm. This has no effect on the bisection step.
        Options:

        'k-means++' : selects initial cluster centers for k-mean
        clustering in a smart way to speed up convergence. See section notes in k_init for details.

        'random': choose `n_clusters` observations (rows) at random from data
        for the initial centroids.

    n_init : int, default=10
        Number of time the internal K-Means algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of inertia. This has no effect on the bisection step.

    max_iter : int, default=300
        Maximum number of iterations of the internal K-Means algorithm for a given
        bisection step.

    tol : float, default=1e-4
        Relative tolerance with regards to Frobenius norm of the difference
        in the cluster centers of two consecutive iterations to declare
        convergence for the internal K-means algorithm. This has no effect on the bisection step.

    verbose : int, default=0
        Verbosity mode.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for centroid initialization on the internal K-Means
        algorithm. This has no effect on the bisection step. Use
        an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.

    copy_x : bool, default=True
        When pre-computing distances in the internal K-Means algorithm it is more 
        numerically accurate to center the data first. If copy_x is True (default), 
        then the original data is not modified. If False, the original data is modified, 
        and put back before the function returns, but small numerical differences may be
        introduced by subtracting and then adding the data mean. Note that if
        the original data is not C-contiguous, a copy will be made even if
        copy_x is False. If the original data is sparse, but not in CSR format,
        a copy will be made even if copy_x is False. Note this will also copy
        the array during the operations of the bisection step to avoid side effects
        which may arise from calculations (the array's shape will always remain the
        same, however).

    algorithm : {"lloyd", "elkan", "auto", "full"}, default="lloyd"
        K-Means algorithm to use for the internal K-Means. The classical EM-style algorithm is `"lloyd"`.
        The `"elkan"` variation can be more efficient on some datasets with
        well-defined clusters, by using the triangle inequality. However it's
        more memory intensive due to the allocation of an extra array of shape at most
        `(n_samples, n_clusters)`. Note the extra array is re-allocated at each bisection step,
        however due to the nature of the algorithm it's size is always non-increasing.

        `"auto"` and `"full"` are deprecated and they will be removed in
        Scikit-Learn 1.3. They are both aliases for `"lloyd"`.

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers.

    labels_ : ndarray of shape (n_samples,)
        Labels of each point
    
    inertia_ : float
        Sum of squared distances of samples to their assigned cluster center,
        weighted by the sample weights if provided.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.
    """

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
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state
        self.copy_x = copy_x
        self.algorithm = algorithm
        self.kmeans = KMeans(
            n_clusters=self.n_split,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            random_state=self.random_state,
            copy_x=copy_x,
            algorithm=algorithm
        )
        self._n_threads = _openmp_effective_n_threads()
        self.max_iter = max_iter


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
        # Check parameters
        self._check_params(X)

        # Data validation -- sets n_features_in
        X = self._validate_data(
           X,
           accept_sparse="csr",
           dtype=[np.float64, np.float32],
           order="C",
           copy=self.copy_x,
           accept_large_sparse=False,
        )

        sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)

        # Initial split of data.
        kmeans_bisect = self.kmeans.fit(X)
        sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)

        # Keep track of all clusters. Update after each split. Pick out cluster
        # with highest SSE for splitting.
        all_clusters = self._split_cluster_points(
            X,
            sample_weight=sample_weight,
            centers=kmeans_bisect.cluster_centers_,
            labels=kmeans_bisect.labels_
        )

        self.cluster_centers_ = kmeans_bisect.cluster_centers_
        self._n_features_out = kmeans_bisect.cluster_centers_.shape[0]
        self.labels_ = kmeans_bisect.labels_

        while self.cluster_centers_.shape[0] < self.n_clusters:
            # Select cluster with highest SSE
            max_sse_idx = np.argmax([c["inertia"] for c in all_clusters])
            selected_cluster = all_clusters[max_sse_idx]

            # Performs kmeans (k=2), on the selected cluster.
            # Replace the old cluster (selected_cluster) with the clusters obtained
            # from kmeans 2. This way, we keep track of all clusters, both the ones
            # obtained from splitting and the old ones that didn't qualify
            # for splitting.
            kmeans_bisect = self.kmeans.fit(selected_cluster["X"])
            all_clusters = all_clusters[:max_sse_idx] + self._split_cluster_points(
                selected_cluster["X"],
                sample_weight=selected_cluster["sample_weight"],
                centers=kmeans_bisect.cluster_centers_,
                labels=kmeans_bisect.labels_
            ) + all_clusters[max_sse_idx+1:]

            # Update cluster_centers_. Replace cluster center of max sse in
            # self.cluster_centers_ with new centers obtained from performing kmeans 2.
            max_sse_center_idx = np.where(
                np.all(self.cluster_centers_ == selected_cluster["centers"], axis=1)
            )[0][0]
            # Remove old center
            self.cluster_centers_ = np.delete(
                self.cluster_centers_,
                max_sse_center_idx,
                axis=0
            )
            # Insert new center in place of old one
            self.cluster_centers_ = np.insert(
                self.cluster_centers_,
                max_sse_center_idx,
                kmeans_bisect.cluster_centers_,
                axis=0
            )

            # Update labels_. Replace labels of max sse in self.labels_ with
            # new labels obtained from performing kmeans 2. Update labels to
            # correspond to the indices of updated self.cluster_centers_
            # [1, 2, 2, 3, 3, 4, 4, 5]
            idx_to_change = np.where(self.labels_ > max_sse_center_idx)[0]
            self.labels_[idx_to_change] = self.labels_[idx_to_change] + 1
            max_sse_labels_idxs = np.where(self.labels_ == max_sse_center_idx)[0]
            self.labels_[max_sse_labels_idxs] = (kmeans_bisect.labels_
                                                 + max_sse_center_idx)

            self._n_features_out = self.cluster_centers_.shape[0]
        
        self.inertia_ = np.sum([c["inertia"] for c in all_clusters])

        return self

    def fit_predict(self, X, y=None, sample_weight=None):
        """Compute cluster centers and predict cluster index for each sample using
        bisecting K-Means.

        Convenience method; equivalent to calling fit(X) followed by
        predict(X).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to transform.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        return self.fit(X, sample_weight=sample_weight).labels_

    def fit_transform(self, X, y=None, sample_weight=None):
        """Compute clustering by KMeans and transform X to cluster-distance space (see 
        tranform for a description of this space).

        Equivalent to fit(X).transform(X), but more efficiently implemented.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Data to fit on, then transform.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_clusters)
            X transformed in cluster-distance space.
        """
        return self.fit(X, sample_weight=sample_weight)._transform(X)

    def score(self, X, y=None, sample_weight=None):
        """Opposite(Negative) of the value of X on the K-means objective.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to score.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

        Returns
        -------
        score : float
            Opposite(Negative) of the value of X on the Bisecting K-means objective.
        """
        check_is_fitted(self)

        X = self._check_test_data(X)
        x_squared_norms = row_norms(X, squared=True)
        sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)

        return -_labels_inertia_threadpool_limit(
            X, sample_weight, x_squared_norms, self.cluster_centers_, self._n_threads
        )[1]

    def transform(self, X):
        """Transform X to a cluster-distance space.

        In the new space, each dimension is the distance to the cluster
        centers. Note that even if X is sparse, the array returned by
        `transform` will typically be dense.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to transform.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_clusters)
            X transformed in the new space.
        """
        check_is_fitted(self)

        X = self._check_test_data(X)
        return self._transform(X)

    def _transform(self, X):
        """Primary functionality of the transform method; run without input validation."""
        return euclidean_distances(X, self.cluster_centers_)

    def predict(self, X, sample_weight=None):
        """Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to predict.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Labels of the cluster each sample belongs to.
        """
        check_is_fitted(self)

        x_squared_norms = row_norms(X, squared=True)
        sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)

        return _labels_inertia_threadpool_limit(
            X, sample_weight, x_squared_norms, self.cluster_centers_, self._n_threads
        )[0]
    
    def _check_params(self, X):
        if self.n_init <= 0:
            raise ValueError(f"n_init should be > 0, got {self.n_init} instead.")
        self._n_init = self.n_init

        if self.max_iter <= 0:
            raise ValueError(f"max_iter should be > 0, got {self.max_iter} instead.")

        if X.shape[0] < self.n_clusters:
            raise ValueError(
                f"n_samples={X.shape[0]} should be >= n_clusters={self.n_clusters}."
            )

        if self.algorithm not in ("lloyd", "elkan", "auto", "full"):
            raise ValueError(
                "Algorithm must be either 'lloyd' or 'elkan', "
                f"got {self.algorithm} instead."
            )

        self._algorithm = self.algorithm
        if self._algorithm in ("auto", "full"):
            warnings.warn(
                f"algorithm='{self._algorithm}' is deprecated, it will be "
                "removed in 1.3. Using 'lloyd' instead.",
                FutureWarning,
            )
            self._algorithm = "lloyd"
        if self._algorithm == "elkan" and self.n_clusters == 1:
            warnings.warn(
                "algorithm='elkan' doesn't make sense for a single "
                "cluster. Using 'lloyd' instead.",
                RuntimeWarning,
            )
            self._algorithm = "lloyd"

        if not (isinstance(self.init, str) and self.init in ["k-means++", "random"]
        ):
            raise ValueError(
                "init should be either 'k-means++', 'random', a ndarray or a "
                f"callable, got '{self.init}' instead."
            )

    def _split_cluster_points(self, X, sample_weight, centers, labels):
        """Separate X into several objects, each of which describes a different cluster in X.
        
        Parameters
        ----------

        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Data to separate.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.
        
        centers : array-like of shape (n_labels,)
            The centers for each label in X. It is assumed the label matches the cluster's index.
        
        labels : ndarray of shape (n_samples,)
            Labels of the cluster each sample belongs to.

        Returns
        -------

        split_clusters : list of dict
            Split data and information. Each dictionary contains the following attributes:

            X : {array-like, sparse matrix} of shape (n_samples_in_cluster, n_features)
                All data from X corresponding to the cluster of the dictionary.
            
            sample_weight : array-like of shape (n_samples_in_label,), default=None
                The weights for each observation in X corresponsing to this particular cluster.
            
            centers : array-like of shape (1, n_features)
                The center of this cluster.
            
            labels : ndarray of shape (n_samples,)
                Array of all zeros, as there is only one cluster in this dataset. 
                This array can be used when calculating inertia to ensure it identifies
                the points properly.
            
            inertia : float
                Sum of squared distances of all data in this cluster to the center of the cluster.
        """
        split_clusters = []
        for i in np.unique(labels):
            cluster_data = {}
            # Have to specify dtype, otherwise _inertia gives error.
            cluster_data["X"] = np.array(X[labels == i], dtype=np.float64)
            cluster_data["sample_weight"] = sample_weight[labels == i]
            # Reshape 1D array to 2D: (1, 1).
            cluster_data["centers"] = np.reshape(centers[i], (1, -1))
            # Every datapoint in X is labeled 0.
            cluster_data["labels"] = np.full(cluster_data["X"].shape[0], 0, dtype=np.int32)
            if sp.issparse(cluster_data["X"]):
                _inertia = _inertia_sparse
            else:
                _inertia = _inertia_dense
            cluster_data["inertia"] = _inertia(
                cluster_data["X"],
                cluster_data["sample_weight"],
                cluster_data["centers"],
                cluster_data["labels"],
                self._n_threads
            )
            split_clusters.append(cluster_data)
        return split_clusters

    def _check_test_data(self, X):
        X = self._validate_data(
            X,
            accept_sparse="csr",
            reset=False,
            dtype=[np.float64, np.float32],
            order="C",
            accept_large_sparse=False,
        )
        return X
