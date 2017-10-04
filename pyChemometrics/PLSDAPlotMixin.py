from abc import ABCMeta
import matplotlib.pyplot as plt
import numpy as np
from pyChemometrics.ChemometricsPLSDA import ChemometricsPLSDA

class PLSDAPlotMixin(ABCMeta):
    """

    Mixin Class to add plotting methods to ChemometricsPLS objects if desired.

    """

    def __init__(self):
        pass

    def plotScores(self, x, y, comps=[0, 1], dir='x', col=None):

        if len(comps) == 1:
            plt.scatter(range(self.))
        else:
            plt.scatter(self..scores[:, 0], PCA_model.scores[:, 1])

        t2 = self.hotelling_T2(comps=comps)

        angle = np.arange(-np.pi, np.pi, 0.01)
        x = t2[0] * np.cos(angle)
        y = t2[1] * np.sin(angle)

        plt.plot(self.scores_t[:, comps[0]], self.scores_t[:, comps[1]], 'ro')

        plt.axhline()
        plt.axvline()
        plt.plot(x, y)
        # plt.vlines(0, 0, 100)
        # plt.hlines(0, np.min(PCA_model.scores[:,0])*10, np.max(PCA_model.scores[:,0])*10)
        plt.title("PLS score plot")
        plt.xlabel("T[{0}] - Variance Explained : T{[1]} %".format(((comps[0] + 1), self.modelParameters['R']))

        plt.ylabel("PC2 - Variance Explained : " + str(PCA_model.modelParameters['VarExpRatio'][1]))
        plt.show()

        return None

    def plotLoadings(self):
        cls.loading
        return None

    def plotScree(self, x, ):
        models = list()
        total_comps = 10
        for ncomps in range(1, total_comps):
            currmodel = self.ChemometricsPLS(ncomps=ncomps)
            currmodel.fit(x, y)
            currmodel.cross_validation(X, outputdist=False, press_impute=False)
            models.append(currmodel)
        return None

    def plotROC(self, which_class=0):
        return None

    def plotConfusionMatrix(cls):
        return None

    def plotPermutationResults(cls, metric='Q2'):
        return None