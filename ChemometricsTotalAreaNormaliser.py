from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.extmath import _incremental_mean_and_var
from sklearn.utils.sparsefuncs import (inplace_column_scale,
                                 mean_variance_axis, incr_mean_variance_axis)
from sklearn.utils.validation import check_is_fitted, FLOAT_DTYPES
from scipy import sparse
import numpy
from copy import deepcopy


class ChemometricsTotalAreaNormaliser(BaseEstimator, TransformerMixin):
    """
    A scikit-learn like scaler object for normalisation (row-scaling)

    :param copy: Whether to copy or not the array to be scaled

    """

    """Standardize features by removing the mean and scaling to unit variance
    Centering and scaling happen independently on each feature by computing
    the relevant statistics on the samples in the training set. Mean and
    standard deviation are then stored to be used on later data using the
    `transform` method.
    Standardization of a dataset is a common requirement for many
    machine learning estimators: they might behave badly if the
    individual feature do not more or less look like standard normally
    distributed data (e.g. Gaussian with 0 mean and unit variance).
    For instance many elements used in the objective function of
    a learning algorithm (such as the RBF kernel of Support Vector
    Machines or the L1 and L2 regularizers of linear models) assume that
    all features are centered around 0 and have variance in the same
    order. If a feature has a variance that is orders of magnitude larger
    that others, it might dominate the objective function and make the
    estimator unable to learn from other features correctly as expected.
    This scaler can also be applied to sparse CSR or CSC matrices by passing
    `with_mean=False` to avoid breaking the sparsity structure of the data.
    Read more in the :ref:`User Guide <preprocessing_scaler>`.
    Parameters
    ----------
    copy : boolean, optional, default True
        If False, try to avoid a copy and do inplace normalization instead.
    Attributes
    ----------
    integrals_ : ndarray, shape (n_features,)
        Per feature relative scaling of the data.
        .. versionadded:: 0.17
           *scale_*
    n_samples_seen_ : int
        The number of samples processed by the estimator. Will be reset on
        new calls to fit, but increments across ``partial_fit`` calls.
    """

    def __init__(self, copy=True):
        self.copy = copy

    def _reset(self):
        """Reset internal data-dependent state of the scaler, if necessary.
        __init__ parameters are not touched.
        """

        # Checking one attribute is enough, because they are all set together
        # in partial_fit
        if hasattr(self, 'integrals_'):
            del self.integrals_
            del self.n_samples_seen_

    def fit(self, X):
        """
        Scikit-learn Fit method:
        Replaced for a non-functional version as TA normalization always has to be
        calculated and used on the same set of samples.
        :param y:
        :return:
        """
        return NotImplementedError

    def _fit(self, X):

        """Compute the mean and std to be used for later scaling.
        Not meant to be used directly - Remove this method?
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape [n_samples, n_features]
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.
        y : Pass through for ``Pipeline`` compatibility.
        """

        # Reset internal state before fitting
        self._reset()

        X = check_array(X, accept_sparse=('csr', 'csc'), copy=self.copy,
                        warn_on_dtype=True, estimator=self, dtype=FLOAT_DTYPES)

        self.integrals_ = numpy.sum(X, axis=1)
        self.n_samples_seen_ = X.shape[0]
        
        return self

    def fit_transform(self, X, copy=True):
        """
        Normalize a set of samples by Total Area Normalization

        :param X:
        :param copy:
        :return:
        """

        self._fit(X)

        copy = copy if copy is not None else self.copy

        X = check_array(X, accept_sparse='csr', copy=copy, warn_on_dtype=True,
                        estimator=self, dtype=FLOAT_DTYPES)

        X = X/self.integrals_[:, None]

        return X

    def transform(self, X, copy=None):
        """
        Sklearn transform method - Replaced for a non-functional version as
        TA normalization always has to be calculated and used on the same set of samples.
        :param X:
        :param y:
        :param copy:
        :return:
        """
        return NotImplementedError

    def inverse_transform(self, X, copy=None):
        """Not easy to implement with Total Area
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data used to scale along the features axis.
        """

        return NotImplementedError

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result