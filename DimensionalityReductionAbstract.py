import ModelAbstract
from sklearn.base import TransformerMixin
from abc import ABCMeta, abstractmethod, abstractproperty
__author__ = 'gd2212'


class DimensionalityReductionAbstract(ABCMeta, ModelAbstract, TransformerMixin):

    def __init__(cls, model, metadata):
        try:
            if not isinstance(model, TransformerMixin):
                raise TypeError

            super(cls, DimensionalityReductionAbstract).__init__(model, metadata)
        except TypeError:
            print("")
            
    @abstractmethod
    def fit_transform(cls, x, y=None, **fit_params):
        return cls.model.fit_transform(x, y=None, **fit_params)

    @abstractmethod
    def inverse_transform(cls, x, y):
        return cls.model.inverse_transform(x, y)

    @abstractmethod
    def score_plot(cls):
        return