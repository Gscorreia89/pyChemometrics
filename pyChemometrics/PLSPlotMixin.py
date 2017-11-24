from abc import ABCMeta
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from copy import deepcopy
from pyChemometrics.ChemometricsPLS import ChemometricsPLS

# TODO Unfinished do not use


class PLSPlotMixin(ABCMeta):
    """

    Mixin Class to add plotting methods to ChemometricsPLS objects if desired.

    """

    def __init__(self):
        pass

    def plot_scores(self, comps=[0, 1], col=None):
        plt.figure()
        if len(comps) == 1:
            plt.scatter(range(self.scores_t.shape[0]), self.scores_t, color=col)
        else:
            plt.scatter(self.scores_t[:, comps[0]], self.scores_t[:, comps[1]], color=col)

            t2 = self.hotelling_T2(comps=comps)

            angle = np.arange(-np.pi, np.pi, 0.01)
            x = t2[0] * np.cos(angle)
            y = t2[1] * np.sin(angle)

            plt.axhline(c='k')
            plt.axvline(c='k')
            plt.plot(x, y, c='k')
            plt.title("PLS score plot")
            plt.xlabel("T[{0}]".format((comps[0] + 1)))
            plt.ylabel("T[{0}]".format((comps[1] + 1)))
            plt.show()
        return None

    def scree_plot(self, x, y, total_comps=5):

        plt.figure()
        models = list()
        for ncomps in range(1, total_comps + 1):
            currmodel = deepcopy(self)
            currmodel.ncomps = ncomps
            currmodel.fit(x, y)
            currmodel.cross_validation(x, y)
            models.append(currmodel)
            q2 = np.array([x.cvParameters['PLS']['Q2Y'] for x in models])
            r2 = np.array([x.modelParameters['PLS']['R2Y'] for x in models])

        plt.bar([x - 0.1 for x in range(1, total_comps + 1)], height=r2, width=0.2)
        plt.bar([x + 0.1 for x in range(1, total_comps + 1)], height=q2, width=0.2)
        plt.legend(['R2', 'Q2'])
        plt.xlabel("Number of components")
        plt.ylabel("R2/Q2X")

        # Specific case where n comps = 2 # TODO check if this edge case works
        if q2.size == 2:
            plateau_index = np.where(np.diff(q2) / q2[0] < 0.05)[0]
            if plateau_index.size == 0:
                print("Consider exploring a higher level of components")
            else:
                plateau = np.min(np.where(np.diff(q2)/q2[0] < 0.05)[0])
                plt.vlines(x=(plateau + 1), ymin=0, ymax=1, colors='red', linestyles='dashed')
                print("Q2X measure stabilizes (increase of less than 5% of previous value or decrease) "
                      "at component {0}".format(plateau + 1))

        else:
            plateau_index = np.where((np.diff(q2) / q2[0:-1]) < 0.05)[0]
            if plateau_index.size == 0:
                print("Consider exploring a higher level of components")
            else:
                plateau = np.min(plateau_index)
                plt.vlines(x=(plateau + 1), ymin=0, ymax=1, colors='red', linestyles='dashed')
                print("Q2X measure stabilizes (increase of less than 5% of previous value or decrease) "
                      "at component {0}".format(plateau + 1))
        plt.show()

        return None

    def plot_permutation_test(self, permt_res, metric='Q2Y'):
        try:
            plt.figure()
            hst = plt.hist(permt_res[0][metric], 100)
            if metric == 'Q2Y':
                plt.vlines(x=self.cvParameters['Q2Y'], ymin=0, ymax=max(hst[0]))
            plt.show()
            return None

        except KeyError:
            print("Run cross-validation before calling the plotting function")
        except Exception as exp:
            raise exp

    def plot_weights(self, component=1, bar=False):

        # Adjust the indexing so user can refer to component 1 as component 1 instead of 0
        component -= 1
        plt.figure()
        # For "spectrum/continuous like plotting"
        if bar is False:
            ax = plt.plot(self.weights_w[:, component])
            if self.cvParameters is not None:
                plt.fill_between(range(self.weights_w[:, component].size),
                                 self.cvParameters['PLS']['Mean_Weights_w'][:, component] - 2*self.cvParameters['PLS']['Stdev_Weights_w'][:, component],
                                 self.cvParameters['PLS']['Mean_Weights_w'][:, component] + 2*self.cvParameters['PLS']['Stdev_Weights_w'][:, component],
                                 alpha=0.2, color='red')
        # To use with barplots for other types of data
        else:
            plt.bar(range(self.weights_w[:, component].size), height=self.weights_w[:, component], width=0.2)

        plt.xlabel("Variable No")
        plt.ylabel("Loading for PC{0}".format((component + 1)))
        plt.show()

        return None

    def external_validation_set(self, x, y):

        y_pred = self.predict(x)

        validation_set_results = dict()

        return validation_set_results
