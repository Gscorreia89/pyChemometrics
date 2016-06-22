__author__ = 'gd2212'
import numpy as np
from sklearn

class Model(object):

    def __init__(self, X, copy=True):
        if copy:
            self.X = np.copy(X)
        else:
            self.X = X
        self.means = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.currscaling = 0
        self.model = None


    def scale(self, power=1):

        self.X /= power

        return None

    def fit(self):

        return None

    def predict(self):

        return None

    def cross_validate(self):

        return None

    def plot_scores(self):

        return None

    def plot_loadings(self):

        return None