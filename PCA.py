import pandas as pds
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.cross_validation import KFold, StratifiedKFold

__author__ = 'gd2212'


class PCA(BaseEstimator, RegressorMixin, TransformerMixin):
    """
    General PCA class
    """

    def __init__(self, n_comps=2, pca_algorithm=PCA, metadata=None, **pca_type_kwargs):
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
                    raise TypeError("Metadata")
            if not isinstance(pca_algorithm, BaseEstimator):
                raise TypeError("Scikit-learn model please")

            self._model = pca_algorithm(n_comps, **pca_type_kwargs)

        except TypeError as terp:
            print(terp.args[0])

    def scale(self, power=1, scale_y=True):
        """

        :param power:
        :param scale_y:
        :return:
        """
        if self.scale_power != 0:
            self.x_std = self.x.std()
            self.x_std /= power

        if self.y is not None and scale_y is True:
            self.y_std = self.y.std()
            self.y_std /= power

        return None

    def fit(self, x):
        """

        :param x:
        :return:
        """

        self._model.fit(x)

        return None

    def fit_transform(self, x, **fit_params):
        """

        :param x:
        :param y:
        :param fit_params:
        :return:
        """
        return self._model.fit_transform(x, **fit_params)

    def score(self, x, sample_weight=None):
        """

        :param x:
        :param sample_weight:
        :return:
        """
        return None

    def predict(self, x=None, y=None):

        return None

    def cross_validation(self, method=KFold, **crossval_kwargs):

        self.pipeline = make_pipeline(Kfold)
        return None

    def score_plot(self, lvs=[1,2], which_scores='T'):

        if which_scores == 'T':
            pass
        elif which_scores == 'Tcv':
            pass
        elif which_scores == 'U':
            pass
        elif which_scores == 'Ucv':
            pass
        elif which_scores == 'TU':
            pass
        elif which_scores == 'TUcv':
            pass

        return None

    def coeffs_plot(self, lv=1, coeffs='weights'):

        return None

