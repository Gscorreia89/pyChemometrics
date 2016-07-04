from sklearn.base import RegressorMixin
import DimensionalityReductionAbstract
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import make_pipeline
from sklearn.cross_validation import KFold, StratifiedKFold
import numpy as np

__author__ = 'gd2212'


class PLS(DimensionalityReductionAbstract, RegressorMixin):

    def __init__(self, n_comps=2, pls_algorithm=PLSRegression,  metadata=None, **pls_type_kwargs):
        """

        :param x:
        :param copy:
        :param metadata:
        :param n_comps:
        :param pca_algorithm:
        :param pca_type_kwargs:
        :return:
        """

        try:
            super(self, PLS).__init__(metadata)
            self._model = pls_algorithm(n_comps, **pls_type_kwargs)

            self.x_means = np.mean(self.x, axis=0)
            self.x_std = np.std(self.x, axis=0)

            self.y_means = np.mean(y, axis=0)
            self.y_std = np.std(y, axis=0)
            # Start with no scaling
            self.x_scalepower = 0
            self.y_scalepower = 0

        except TypeError:
            print()

    def scale(self, power=1, scale_y=True):
        """

        :param power:
        :param scale_y:
        :return:
        """
        super(self, DimensionalityReductionAbstract).scale(power, scale_y)
        return None

    def fit(self, x, y):
        """

        :param x:
        :return:
        """
        self._model.fit(x, y)
        return None

    def fit_transform(self, x, y=None, **fit_params):
        """

        :param x:
        :param y:
        :param fit_params:
        :return:
        """
        return self._model.fit_transform(x, y, **fit_params)

    def inverse_transform(self, ):

        return self._model.inverse_transform(x, y)

    def score(self, x, y, sample_weight=None):
        """

        :param x:
        :param sample_weight:
        :return:
        """

        r2x = self._model.score(x, y)
        r2y = self._model.score(y, x)

        return None

    def predict(cls, x=None, y=None):

        return None

    def cross_validation(self, method=KFold, **crossval_kwargs):
        self.pp = Pipeline()
        return None

    def score_plot(self, lvs=[1,2], scores="T"):

        return None

    def coeffs_plot(self, lv=1, coeffs='weights'):
        return None

    def __hotellingT2(self, ):

