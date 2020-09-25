from copy import deepcopy
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.decomposition import PCA as skPCA
from sklearn.decomposition._base import _BasePCA
from .ChemometricsScaler import ChemometricsScaler

__author__ = 'gd2212'


class ChemometricsPCA(_BasePCA, BaseEstimator):
    """

    ChemometricsPCA object - Wrapper for sklearn.decomposition PCA algorithms, with tailored methods
    for Chemometric Data analysis.

    :param n_components: Number of PCA components desired.
    :type n_components: int
    :param sklearn.decomposition._BasePCA pca_algorithm: scikit-learn PCA models (inheriting from _BasePCA).
    :param scaler: The object which will handle data scaling.
    :type scaler: ChemometricsScaler object, scaling/preprocessing objects from scikit-learn or None
    :param kwargs pca_type_kwargs: Keyword arguments to be passed during initialization of pca_algorithm.
    :raise TypeError: If the pca_algorithm or scaler objects are not of the right class.
    """

    # Constant usage of kwargs might look excessive but ensures that most things from scikit-learn can be used directly
    # no matter what PCA algorithm is used
    def __init__(self, n_components=2, pca_algorithm=skPCA, scaler=ChemometricsScaler(), **pca_type_kwargs):

        try:
            # Perform the check with is instance but avoid abstract base class runs. PCA needs number of comps anyway!
            init_pca_algorithm = pca_algorithm(n_components=n_components, **pca_type_kwargs)
            if not isinstance(init_pca_algorithm, (_BasePCA, BaseEstimator, TransformerMixin)):
                raise TypeError("Must be a Scikit-learn PCA model")
            if not (isinstance(scaler, TransformerMixin) or scaler is None):
                raise TypeError("Must be a Scikit-learn Transformer-like object or None")
            if scaler is None:
                scaler = ChemometricsScaler(0, with_std=False)

            self.pca_algorithm = init_pca_algorithm

            # Most initialized as None, before object is fitted.
            self.scores = None
            self.loadings = None
            self._n_components = n_components
            self._scaler = scaler
            self.modelParameters = None
            self._is_fitted = False

        except TypeError as terp:
            print(terp.args[0])
            raise terp

    def fit(self, x, **fit_params):
        """

        Perform model fitting on the provided x data matrix and calculate basic goodness-of-fit metrics.
        Equivalent to scikit-learn's default BaseEstimator method.

        :param x: Data matrix to fit the PCA model.
        :type x: numpy.ndarray, shape [n_samples, n_features].
        :param kwargs fit_params: Keyword arguments to be passed to the .fit() method of the core sklearn model.
        :raise ValueError: If any problem occurs during fitting.
        """

        try:
            # This scaling check is always performed to ensure running model with scaling or with scaling == None
            # always give consistent results (same type of data scale expected for fitting,
            # returned by inverse_transform, etc
            if self.scaler is not None:
                xscaled = self.scaler.fit_transform(x)
                self.pca_algorithm.fit(xscaled, **fit_params)
                self.scores = self.pca_algorithm.transform(xscaled)

            else:
                self.pca_algorithm.fit(x, **fit_params)
                self.scores = self.pca_algorithm.transform(x)
            self.modelParameters = {'R2X': self.pca_algorithm.explained_variance_ratio_.sum(),
                                    'VarExp': self.pca_algorithm.explained_variance_,
                                    'VarExpRatio': self.pca_algorithm.explained_variance_ratio_}

            # Kernel PCA and other non-linear methods might not have explicit loadings - safeguard against this
            if hasattr(self.pca_algorithm, 'components_'):
                self.loadings = self.pca_algorithm.components_
            self._is_fitted = True

        except ValueError as verr:
            raise verr

    def partial_fit(self, x, **partial_fit_params):
        """
        under construction...
        :param x:
        :return:
        """
        # TODO add tests and check how many models implement this
        try:
            # This scaling check is always performed to ensure running model with scaling or with scaling == None
            # always give consistent results (same type of data scale expected for fitting,
            # returned by inverse_transform, etc
            if self.scaler is not None:
                self.scaler.partial_fit(x)
                xscaled = self.scaler.transform(x)
                self.pca_algorithm.partial_fit(xscaled, **partial_fit_params)
                self.scores = self.pca_algorithm.transform(xscaled)

            else:
                self.pca_algorithm.partial_fit(x, **partial_fit_params)
                self.scores = self.pca_algorithm.transform(x)
            self.modelParameters = {'R2X': self.pca_algorithm.explained_variance_ratio_.sum(),
                                    'VarExp': self.pca_algorithm.explained_variance_,
                                    'VarExpRatio': self.pca_algorithm.explained_variance_ratio_}

            # Kernel PCA and other non-linear methods might not have explicit loadings - safeguard against this
            if hasattr(self.pca_algorithm, 'components_'):
                self.loadings = self.pca_algorithm.components_
            self._is_fitted = True

        except ValueError as verr:
            raise verr

    def fit_transform(self, x, **fit_params):
        """

        Fit a model and return the scores, as per the scikit-learn's TransformerMixin method.

        :param x: Data matrix to fit and project.
        :type x: numpy.ndarray, shape [n_samples, n_features]
        :param kwargs fit_params: Optional keyword arguments to be passed to the fit method.
        :return: PCA projections (scores) corresponding to the samples in X.
        :rtype: numpy.ndarray, shape [n_samples, n_comps]
        :raise ValueError: If there are problems with the input or during model fitting.
        """

        try:
            self.fit(x, **fit_params)
            return self.transform(x)
        except ValueError as exp:
            raise exp

    def transform(self, x):
        """

        Calculate the projections (scores) of the x data matrix. Similar to scikit-learn's TransformerMixin method.

        :param x: Data matrix to fit and project.
        :type x: numpy.ndarray, shape [n_samples, n_features]
        :param kwargs transform_params: Optional keyword arguments to be passed to the transform method.
        :return: PCA projections (scores) corresponding to the samples in X.
        :rtype: numpy.ndarray, shape [n_samples, n_comps]
        :raise ValueError: If there are problems with the input or during model fitting.
        """
        try:
            if self.scaler is not None:
                xscaled = self.scaler.transform(x)
                return self.pca_algorithm.transform(xscaled)
            else:
                return self.pca_algorithm.transform(x)
        except ValueError as verr:
            raise verr

    def score(self, x, sample_weight=None):
        """

        Return the average log-likelihood of all samples. Same as the underlying score method from the scikit-learn
        PCA objects.

        :param x: Data matrix to score model on.
        :type x: numpy.ndarray, shape [n_samples, n_features]
        :param numpy.ndarray sample_weight: Optional sample weights during scoring.
        :return: Average log-likelihood over all samples.
        :rtype: float
        :raises ValueError: if the data matrix x provided is invalid.
        """
        try:
            # Not all sklearn pca objects have a "score" method...
            score_method = getattr(self.pca_algorithm, "score", None)
            if not callable(score_method):
                raise NotImplementedError
            # Scaling check for consistency
            if self.scaler is not None:
                xscaled = self.scaler.transform(x)
                return self.pca_algorithm.score(xscaled, sample_weight)
            else:
                return self.pca_algorithm.score(x, sample_weight)
        except ValueError as verr:
            raise verr

    def inverse_transform(self, scores):
        """

        Transform scores to the original data space using the principal component loadings.
        Similar to scikit-learn's default TransformerMixin method.

        :param scores: The projections (scores) to be converted back to the original data space.
        :type scores: numpy.ndarray, shape [n_samples, n_comps]
        :return: Data matrix in the original data space.
        :rtype: numpy.ndarray, shape [n_samples, n_features]
        :raises ValueError: If the dimensions of score mismatch the number of components in the model.
        """
        # Scaling check for consistency
        if self.scaler is not None:
            xinv_prescaled = self.pca_algorithm.inverse_transform(scores)
            xinv = self.scaler.inverse_transform(xinv_prescaled)
            return xinv
        else:
            return self.pca_algorithm.inverse_transform(scores)

    @property
    def n_components(self):
        try:
            return self._ncomponents
        except AttributeError as atre:
            raise atre

    @n_components.setter
    def n_components(self, n_components):
        """

        Setter for number of components. Resets the model and changes n_components

        :param int n_components: Number of components to use in the model.
        :raise AttributeError: If there is a problem changing the number of components and resetting the model.
        """
        # To ensure changing number of components effectively resets the model
        try:
            self._n_components = n_components
            self.pca_algorithm = clone(self.pca_algorithm, safe=True)
            self.pca_algorithm.n_components = n_components
            self.modelParameters = None
            self.loadings = None
            self.scores = None
            return None
        except AttributeError as atre:
            raise atre

    @property
    def scaler(self):
        try:
            return self._scaler
        except AttributeError as atre:
            raise atre

    @scaler.setter
    def scaler(self, scaler):
        """

        Setter for the model scaler.

        :param scaler: The object which will handle data scaling.
        :type scaler: ChemometricsScaler object, scaling/preprocessing objects from scikit-learn or None
        :raise AttributeError: If there is a problem changing the scaler and resetting the model.
        :raise TypeError: If the new scaler provided is not a valid object.
        """
        try:
            if not (isinstance(scaler, TransformerMixin) or scaler is None):
                raise TypeError("Scikit-learn Transformer-like object or None")
            if scaler is None:
                scaler = ChemometricsScaler(0, with_std=False)

            self._scaler = scaler
            self.pca_algorithm = clone(self.pca_algorithm, safe=True)
            self.modelParameters = None
            self.loadings = None
            self.scores = None
            return None

        except AttributeError as atre:
            raise atre
        except TypeError as typerr:
            raise typerr

    @property
    def is_fitted(self):
        try:
            return self._is_fitted
        except AttributeError as atre:
            raise atre

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result
