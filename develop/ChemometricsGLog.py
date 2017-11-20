from copy import deepcopy

import numpy
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.extmath import _incremental_mean_and_var
from sklearn.utils.sparsefuncs import (inplace_column_scale,
                                       mean_variance_axis, incr_mean_variance_axis)
from sklearn.utils.validation import check_is_fitted, FLOAT_DTYPES


class ChemometricsGLog(BaseEstimator, TransformerMixin):
    """

    Scaler object to perform generalized log transform, as in Durbin et al.

    :param scale_power: To which power should the standard deviation of each variable be raised for scaling. 0: Mean centering; 0.5: Pareto; 1:Unit Variance.
    :type scale_power: Float
    :param bool copy: Copy the array containing the data.
    :param bool with_mean: Perform mean centering.
    :param bool with_std: Scale the data.
    """

    def __init__(self):
        self.copy = True
        self.lam = None
        self.alpha = None

    def _reset(self):
        """
        Reset internal data-dependent state of the scaler, if necessary.
        __init__ parameters are not touched.

        """
        #
        if hasattr(self, 'lam'):
            self.lam = None
            self.alpha = None

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

        X = check_array(X, accept_sparse=('csr', 'csc'), copy=self.copy,
                        warn_on_dtype=True, estimator=self, dtype=FLOAT_DTYPES)

        def glog(x, lam):
            return numpy.log(x + numpy.sqrt(x**2 + lam))

        def obj_fun(x, alpha, lam):
            x = x + alpha
            glx = glog(X, lam)
            score = 1

            return score

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

        X = check_array(X, accept_sparse='csr', copy=copy, warn_on_dtype=True,
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