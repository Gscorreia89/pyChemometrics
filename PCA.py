import pandas as pds
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.decomposition import PCA as skPCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import numpy as np

__author__ = 'gd2212'


class PCA(BaseEstimator, RegressorMixin, TransformerMixin):
    """
    General PCA class
    Inherits from Base Estimator, RegressorMixin and TransformerMixin, to act as a completely fake
    # Scikit classifier
    """

    # This class inherits from Base Estimator, RegressorMixin and TransformerMixin to act as believable
    # scikit-learn PCA classifier
    # Constant usage of kwargs ensures that everything from scikit-learn can be used directly
    def __init__(self, n_comps=2, pca_algorithm=skPCA, metadata=None, **pca_type_kwargs):
        """
        :param metadata:
        :param n_comps:
        :param pca_algorithm:
        :param pca_type_kwargs:
        :return:
        """
        try:
            # Metadata assumed to be pandas dataframe only
            if metadata is not None:
                if not isinstance(metadata, pds.DataFrame):
                    raise TypeError("Metadata must be provided as pandas dataframe")
            # The actual classifier can change, but needs to be a scikit-learn BaseEstimator
            # Changes this later for "PCA like only" - either check their hierarchy or make a list
            if not issubclass(pca_algorithm, BaseEstimator):
                raise TypeError("Scikit-learn model please")
            # The kwargs provided for the model are exactly the same as those
            # go and check for these examples the correct exception to throw when kwarg is not valid
            # TO DO: Set the sklearn params for PCA to be a junction of the custom ones and the "core" params of model
            self._model = pca_algorithm(n_comps, **pca_type_kwargs)
            # These will be none until object is fitted.
            self._scores = None
            self._centeringvector = None
            self._scalingvector = None
            self.ncomps = n_comps

        except TypeError as terp:
            print(terp.args[0])

        except ValueError as verr:
            print(verr.args[0])

    def fit(self, x, y=None, **fit_params):
        """
        Fit function. Acts exactly as in scikit-learn, but
        :param x:
        :param scale:
        :return:

        """
        # always Include cross-validation here?
        # The answer is no, otherwise we force a load of useless CV's if
        # we use this into other sklearn pipelines...
        # Finish the independent cv method and
        # study the best way to do this
        # split **fit_params from **crossval_kwargs
        try:

            # Add scaling here...

            self._model.fit(x, **fit_params)
            self.scores = self.transform(x)

        except Exception as exp:
            raise exp

    def fit_transform(self, x, **fit_params):
        """
        Combination of fit and output the scores, nothing special
        :param x:
        :param y:
        :param fit_params:
        :return:
        """

        return self._model.fit_transform(x, **fit_params)

    def transform(self, x):
        """

        :param x:
        :return:
        """

        return self._model.transform(x)

    def score(self, x):
        """
        Enables output of an R^2, and is expected by regressor mixin
        In theory PCA can be used.
        :param x:
        :param sample_weight:
        :return:
        """
        self.model_.score(self, x, x, sample_weight=None)

        return None

    def inverse_transform(self, scores):
        """

        :param scores:
        :return:
        """
        return self._model.inverse_transform(scores)

    def predict(self, x):
        """
        A bit weird for pca... but as part of regressor mixin?
        :param x:
        :return:
        """

        return self.inverse_transform(x)

    @property
    def performance_metrics(self):
        """
        Getter
        :return:
        """
        # this is wrong now, but find the default way to see if the model has been fited
        if self._model.fit():

            return {'R2X': self._model.explained_variance_}
        else:
            # check this properyl, and might need to calculate var explained in other ways
            metricsdict = {'VarExplained': self._model.Var_exp}
        return metricsdict

    @property
    def centeringvector(self):
        """
        Getter for the centering vector
        :return:
        """
        return self._centeringvector

    @centeringvector.setter
    def centeringvector(self, value):
        """
        Setter for the centering vector, to allow custom centring
        :param value:
        :return:
        """
        try:
            if not isinstance(value, np.array):
                raise TypeError('Value provided must be numpy array')
            elif not value.shape == self._centeringvector.shape:
                raise ValueError('Value provided must have same length as number of variables')
            self._centeringvector = value
        except Exception as exp:
            raise exp

    @property
    def scalingvector(self):
        """
        Getter for the scaling vector
        :return:
        """
        return self._scalingvector

    @scalingvector.setter
    def scalingvector(self, value):
        """
        Setter for the scaling vector
        :param value:
        :return:
        """
        try:
            if not isinstance(value, np.array):
                raise TypeError('Value provided must be numpy array')
            elif not value.shape == self.scalingvector.shape:
                raise ValueError('Value provided must have same length as number of variables')
            self.scalingvector = value
        except Exception as exp:
            raise exp

    @property
    def loadings(self, comp=1):
        """

        :param comp:
        :return:
        """
        try:
            loading = self._model.components_[:, comp-1]
            return loading
        except AttributeError as atre:
            raise atre

    @property
    def scores(self, comp=1):
        """

        :param comp:
        :return:
        """
        try:
            return self._scores[:, comp-1]
        except AttributeError as atre:
            raise atre

    def scale(self, scaling=1):
        """
        The usual scaling functions will be here
        :scaling power:

        :return:
        """
        # Reshape this, no need for self.scale_power
        if self.scaling != 0:
            # do this properly and add something for a log
            if callable(scaling):
                self.scalingvector = scaling
            else:
                x_std = self.X.std()
                self.scalingvector = self.x_std ** scaling
        else:
            self.x /= self.scalingvector

        return None

    # Check this one carefully ... good oportunity to build in nested cross-validation
    # and stratified k-folds
    def cross_validation(self, data,  method=7, **crossval_kwargs):

        try:
            if not isinstance(method, _PartitionIterator):
                raise TypeError("Scikit-learn cross-validation object please")

            scores = cross_val_score(self._model, data, cv=method, **crossval_kwargs)

            return None

        except TypeError as terp:
            raise terp

    def score_plot(self, pcs=[1,2], hotelingt=0.95):
        if len(pcs) == 1:
            # do something decent for the 1D  score plot
            pass

        return None

    def trellisplot(self, pcs):
        """
        Trellis plot, maybe quickly wrapping up the stuff from seaborn
        :param pcs:
        :return:
        """
        return None

    def coeffs_plot(self, lv=1, coeffs='weightscv'):
        """

        :param lv:
        :param coeffs:
        :return:
        """
        return None


def bro_svd(V, X):
    a = np.sum(np.dot(np.dot(V.T, X.T), np.dot(V.T, X))**2)

    return None