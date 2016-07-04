from sklearn.base import TransformerMixin, RegressorMixin
import DimensionalityReductionAbstract
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.cross_validation import KFold, StratifiedKFold
__author__ = 'gd2212'


class PCA(DimensionalityReductionAbstract, RegressorMixin):
    """
    General PCA class
    """

    def __init__(self, n_comps=2, pca_algorithm=PCA, metadata=None, **pca_type_kwargs):
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
            super(self, PCA).__init__(metadata)
            self._model = pca_algorithm(n_comps, **pca_type_kwargs)

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

    def predict(cls, x=None, y=None):

        return None

    def cross_validation(self, method=KFold, **crossval_kwargs):
        self.pipeline = make_pipeline(Kfold)
        return None

