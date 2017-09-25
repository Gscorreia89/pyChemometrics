from abc import ABCMeta
import matplotlib.pyplot as plt


class PCAPlotMixin(ABCMeta):
    """

    Mixin Class to add plotting methods to ChemometricsPCA objects if desired.

    """
    def __init__(cls):
        pass

    def plotScores(cls):
        return None

    def plotLoadings(cls):
        cls.loading
        return None

    def plotScree(cls):
        return None

    def plotROC(cls, which_class=0):
        return None
    
    def plotConfusionMatrix(cls):
        return None

    def plotPermutationResults(cls, metric='Q2'):

        return None