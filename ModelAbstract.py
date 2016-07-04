import numpy as np
from sklearn.base import BaseEstimator
from abc import ABCMeta, abstractmethod, abstractproperty
import pandas as pds

__author__ = 'gd2212'


class ModelAbstract(ABCMeta, BaseEstimator):
    """
    Abstract class for "chemometric" wrappers of scikit-learn tools
    """

    def __init__(cls, model=None, metadata=None):
        """
        :param X: The X block of data
        :param Y: Outcome variable or even Y-block
        :param copy: Whether or not to copy the object - No copy = lower memory consumption, but
        things can go terribly wrong when algorithms have deflations (NIPALS and most PLS's)
        :param metadata: Pandas dataframe with metadata. To be used for score plot annotation,
        change regressand, covariate adjustment or multilevel designs
        :return:

        """

        try:

            cls.model = model
            # Metadata assumed to be pandas dataframe only
            if metadata is not None:
                if not not isinstance(metadata, pds.DataFrame):
                    raise TypeError()

            # Store X means and Y means
            cls.x_means = np.mean(x, axis=0)
            cls.x_std = np.std(x, axis=0)

            if y is not None:
                cls.y_means = np.mean(y, axis=0)
                cls.y_std = np.std(y, axis=0)
            # Start with no scaling
            cls.x_scalepower = 0
            cls.y_scalepower = 0

        except TypeError as terp:
            print("Metadata must be supplied as pandas dataframe")

    @abstractproperty
    def model(cls):
        """
        Core model from scikit learn
        """
        return cls.model

    @model.setter
    def model(cls, model):
        if isinstance(BaseEstimator, model):
            cls.model = model
        else:
            raise TypeError

    @abstractmethod
    def scale(self, power=1, scale_y=True):

        if self.scale_power != 0:
            self.x_std = self.x.std()
            self.x_std /= power

        if self.y is not None and scale_y is True:
            self.y_std = self.y.std()
            self.y_std /= power

        return None

    @abstractmethod
    def fit(cls, x, y=None):
        return

    @abstractmethod
    def predict(cls, x=None, y=None):
        return

    @abstractmethod
    def cross_validation(cls, **cross_valparams):
        return

