import ModelAbstract
from sklearn.base import RegressorMixin
from abc import ABCMeta, abstractmethod, abstractproperty

__author__ = 'gd2212'


class RegressionAbstract(ABCMeta, ModelAbstract, RegressorMixin):

    def __init__(cls, model, metadata):
        try:
            if not isinstance(model, RegressorMixin):
                raise TypeError
            super(cls, ModelAbstract).__init__(model, metadata)

        except TypeError:
            print("")

    @abstractmethod
    def score(self, X, y, sample_weight=None):
        return