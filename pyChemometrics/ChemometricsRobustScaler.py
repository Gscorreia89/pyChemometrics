from copy import deepcopy

import numpy
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.extmath import _incremental_mean_and_var
from sklearn.utils.sparsefuncs import (inplace_column_scale,
                                       mean_variance_axis, incr_mean_variance_axis)
from sklearn.utils.validation import check_is_fitted, FLOAT_DTYPES

"""

Source code adapted from scikit-learn preprocessing object StandardScaler. 
(Original unadapted code can be found in version 0.19.1, "scikit-learn/sklearn/preprocessing/data.py") 

"""


class ChemometricsScaler(BaseEstimator, TransformerMixin):
    """
    Extension of Scikit-learn's StandardScaler which allows scaling by different powers of the standard deviation.

    :param scale_power: To which power should the standard deviation of each variable be raised for scaling. 0: Mean centering; 0.5: Pareto; 1:Unit Variance.
    :type scale_power: Float
    :param bool copy: Copy the array containing the data.
    :param bool with_mean: Perform mean centering.
    :param bool with_std: Scale the data.
    """

    def __init__(self, scale_power=1, copy=True, with_mean=True, with_std=True):
        self.scale_power = scale_power
        self.with_mean = with_mean
        self.with_std = with_std
        self.copy = copy

    def _reset(self):
        """
        Reset internal data-dependent state of the scaler, if necessary.
        __init__ parameters are not touched.

        """

        # Checking one attribute is enough, because they are all set together
        # in partial_fit
        if hasattr(self, 'scale_'):
            del self.scale_
            del self.n_samples_seen_
            del self.mean_
            del self.var_

    def fit(self, X, y=None):
        """
        Compute the mean and standard deviation from a dataset to use in future scaling operations.

        :param X: Data matrix to scale.
        :type X: numpy.ndarray, shape [n_samples, n_features]
        :param y: Passthrough for Scikit-learn ``Pipeline`` compatibility.
        :type y: None
        :return: Fitted object.
        :rtype: pyChemometrics.ChemometricsScaler
        """

        # Reset internal state before fitting
        self._reset()
        return self.partial_fit(X, y)

    def partial_fit(self, X, y=None):
        """
        Performs online computation of mean and standard deviation on X for later scaling.
        All of X is processed as a single batch.
        This is intended for cases when `fit` is
        not feasible due to very large number of `n_samples`
        or because X is read from a continuous stream.

        The algorithm for incremental mean
        and std is given in Equation 1.5a,b in Chan, Tony F., Gene H. Golub, and Randall J. LeVeque. "Algorithms
        for computing the sample variance: Analysis and recommendations."
        The American Statistician 37.3 (1983): 242-247

        :param X: Data matrix to scale.
        :type X: numpy.ndarray, shape [n_samples, n_features]
        :param y: Passthrough for Scikit-learn ``Pipeline`` compatibility.
        :type y: None
        :return: Fitted object.
        :rtype: pyChemometrics.ChemometricsScaler

        """

        X = check_array(X, accept_sparse=('csr', 'csc'), copy=self.copy,
                        estimator=self, dtype=FLOAT_DTYPES)

        # Even in the case of `with_mean=False`, we update the mean anyway
        # This is needed for the incremental computation of the var
        # See incr_mean_variance_axis and _incremental_mean_variance_axis

        if sparse.issparse(X):
            if self.with_mean:
                raise ValueError(
                    "Cannot center sparse matrices: pass `with_mean=False` "
                    "instead. See docstring for motivation and alternatives.")
            if self.with_std:
                # First pass
                if not hasattr(self, 'n_samples_seen_'):
                    self.mean_, self.var_ = mean_variance_axis(X, axis=0)
                    self.n_samples_seen_ = X.shape[0]
                # Next passes
                else:
                    self.mean_, self.var_, self.n_samples_seen_ = \
                        incr_mean_variance_axis(X, axis=0,
                                                last_mean=self.mean_,
                                                last_var=self.var_,
                                                last_n=self.n_samples_seen_)
            else:
                self.mean_ = None
                self.var_ = None
        else:
            # First pass
            if not hasattr(self, 'n_samples_seen_'):
                self.mean_ = .0
                self.n_samples_seen_ = 0
                if self.with_std:
                    self.var_ = .0
                else:
                    self.var_ = None

            self.mean_, self.var_, self.n_samples_seen_ = \
                _incremental_mean_and_var(X, self.mean_, self.var_,
                                          self.n_samples_seen_)

        if self.with_std:
            self.scale_ = _handle_zeros_in_scale(numpy.sqrt(self.var_)) ** self.scale_power
        else:
            self.scale_ = None

        return self

    def transform(self, X, y=None, copy=None):
        """
        Perform standardization by centering and scaling using the parameters.

        :param X: Data matrix to scale.
        :type X: numpy.ndarray, shape [n_samples, n_features]
        :param y: Passthrough for scikit-learn ``Pipeline`` compatibility.
        :type y: None
        :param bool copy: Copy the X matrix.
        :return: Scaled version of the X data matrix.
        :rtype: numpy.ndarray, shape [n_samples, n_features]
        """
        check_is_fitted(self, 'scale_')

        copy = copy if copy is not None else self.copy

        X = check_array(X, accept_sparse='csr', copy=copy,
                        estimator=self, dtype=FLOAT_DTYPES)

        if sparse.issparse(X):
            if self.with_mean:
                raise ValueError(
                    "Cannot center sparse matrices: pass `with_mean=False` "
                    "instead. See docstring for motivation and alternatives.")
            if self.scale_ is not None:
                inplace_column_scale(X, 1 / self.scale_)
        else:
            if self.with_mean:
                X -= self.mean_
            if self.with_std:
                X /= self.scale_
        return X

    def inverse_transform(self, X, copy=None):
        """
        Scale back the data to the original representation.

        :param X: Scaled data matrix.
        :type X: numpy.ndarray, shape [n_samples, n_features]
        :param bool copy: Copy the X data matrix.
        :return: X data matrix with the scaling operation reverted.
        :rtype: numpy.ndarray, shape [n_samples, n_features]
        """
        check_is_fitted(self, 'scale_')

        copy = copy if copy is not None else self.copy
        if sparse.issparse(X):
            if self.with_mean:
                raise ValueError(
                    "Cannot uncenter sparse matrices: pass `with_mean=False` "
                    "instead See docstring for motivation and alternatives.")
            if not sparse.isspmatrix_csr(X):
                X = X.tocsr()
                copy = False
            if copy:
                X = X.copy()
            if self.scale_ is not None:
                inplace_column_scale(X, self.scale_)
        else:
            X = numpy.asarray(X)
            if copy:
                X = X.copy()
            if self.with_std:
                X *= self.scale_
            if self.with_mean:
                X += self.mean_

        return X

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result


def _handle_zeros_in_scale(scale, copy=True):
    """
    Makes sure that whenever scale is zero, we handle it correctly.
    This happens in most scalers when we have constant features.
    """

    # if we are fitting on 1D arrays, scale might be a scalar
    if numpy.isscalar(scale):
        if scale == .0:
            scale = 1.
        return scale
    elif isinstance(scale, numpy.ndarray):
        if copy:
            # New array to avoid side-effects
            scale = scale.copy()
        scale[scale == 0.0] = 1.0
        return scale
