import pandas as pds
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.decomposition import PCA as skPCA
from sklearn.pipeline import make_pipeline
from sklearn.cross_validation import KFold, StratifiedKFold
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
                if not not isinstance(metadata, pds.DataFrame):
                    raise TypeError("Metadata must be provided as pandas dataframe")
            # The actual classifier can change, but needs to be a scikit-learn BaseEstimator
            # Changes this later for "PCA like only" - either check their hierarchy or make a list
            if not isinstance(pca_algorithm, BaseEstimator):
                raise TypeError("Scikit-learn model please")
            # The kwargs provided for the model are exactly the same as those
            # go and check for these examples the correct exception to throw when kwarg is not valid
            self._model = pca_algorithm(n_comps, **pca_type_kwargs)

        except TypeError as terp:
            print(terp.args[0])

        except ValueError as verr:
            print(verr.args[0])

    def fit(self, x, scale=1, method=KFold, **crossval_kwargs):
        """
        Fit function. Acts exacly as in scikit-learn, but
        :param x:
        :return:

        """
        # always Include cross-validation here?
        # Finish the independent cv method and
        # study the best way to do this
        # split **fit_params from **crossval_kwargs
        try:
            self._model.fit(x, copy=True)

            self.pipeline = make_pipeline(method, self._model)

            for ncomps in range(0, self.model._ncomps):
                self.model._ncomps = ncomps
                cvoutput = self.pipeline(self.x, self.y)

            return None

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

    def score(self, x, sample_weight=None):
        """
        Enables output of an R^2, and is expected by regressor mixin
        In theory PCA can be used without this, but makes some sense.
        :param x:
        :param sample_weight:
        :return:
        """
        return None

    def predict(self, x=None):
        """
        A bit weird for pca... but should h
        :param x:
        :return:
        """
        return None

    @property
    def performance_metrics(self):
        """
        Getter
        :return:
        """
        # this is wrong now, but find the default way to see if the model has been fited
        if self._model.fit():
            print
            return None
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
            elif not value.shape == self.centeringvector.shape:
                raise ValueError('Value provided must have same length as number of variables')
            self.centeringvector = value
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

    def scale(self, power=1, scale_y=True):
        """
        The usual scaling functions will be here
        :param power:
        :param scale_y:
        :return:
        """

        # Reshape this, no need for self.scale_power
        if self.scale_power != 0:
            # do this properly and add something for a log
            x_std = self.X.std()
            self.scalingvector = self.x_std ** power

        return None

    # Check this one carefully ... good oportunity to build in nested cross-validation
    # and stratified k-folds
    def cross_validation(self, method=KFold, **crossval_kwargs):
        self.pp = make_pipeline
        return None

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