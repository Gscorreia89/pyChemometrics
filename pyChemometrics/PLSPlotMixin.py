from abc import ABCMeta
from copy import deepcopy

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from pyChemometrics.PlotMixin import PlotMixin


# TODO Unfinished do not use


class PLSPlotMixin(PlotMixin, metaclass=ABCMeta):
    """

    Mixin Class to add plotting methods to ChemometricsPLS objects if desired.

    """

    def plot_scores(self, comps=[0, 1], color=None, discrete=False):
        """

        Score plot figure wth an Hotelling T2.

        :param comps: Components to use in the 2D plot
        :param color: Variable used to color points
        :return: Score plot figure
        """
        try:
            plt.figure()

            # Use a constant color if no color argument is passed

            t2 = self.hotelling_T2(alpha=0.05, comps=comps)
            outlier_idx = np.where(((self.scores_t[:, comps] ** 2) / t2 ** 2).sum(axis=1) > 1)[0]

            if len(comps) == 1:
                x_coord = np.arange(0, self.scores_t.shape[0])
                y_coord = self.scores_t[:, comps[0]]
            else:
                x_coord = self.scores_t[:, comps[0]]
                y_coord = self.scores_t[:, comps[1]]

            if color is None:
                plt.scatter(x_coord, y_coord)
                plt.scatter(x_coord[outlier_idx], y_coord[outlier_idx],
                            marker='x', s=1.5 * mpl.rcParams['lines.markersize'] ** 2)
            else:
                if discrete is False:
                    cmap = cm.jet
                    cnorm = Normalize(vmin=min(color), vmax=max(color))

                    plt.scatter(x_coord, y_coord, c=color, cmap=cmap, norm=cnorm)
                    plt.scatter(x_coord[outlier_idx], y_coord[outlier_idx],
                                c=color[outlier_idx], cmap=cmap, norm=cnorm, marker='x',
                                s=1.5 * mpl.rcParams['lines.markersize'] ** 2)
                    plt.colorbar()
                else:
                    cmap = cm.Set1
                    subtypes = np.unique(color)
                    for subtype in subtypes:
                        subset_index = np.where(color == subtype)
                        plt.scatter(x_coord[subset_index], y_coord[subset_index],
                                    c=cmap(subtype), label=subtype)
                    plt.legend()
                    plt.scatter(x_coord[outlier_idx], y_coord[outlier_idx],
                                c=color[outlier_idx], cmap=cmap, marker='x',
                                s=1.5 * mpl.rcParams['lines.markersize'] ** 2)
            if len(comps) == 2:
                angle = np.arange(-np.pi, np.pi, 0.01)
                x = t2[0] * np.cos(angle)
                y = t2[1] * np.sin(angle)
                plt.axhline(c='k')
                plt.axvline(c='k')
                plt.plot(x, y, c='k')

                xmin = np.minimum(min(x_coord), np.min(x))
                xmax = np.maximum(max(x_coord), np.max(x))
                ymin = np.minimum(min(y_coord), np.min(y))
                ymax = np.maximum(max(y_coord), np.max(y))

                axes = plt.gca()
                axes.set_xlim([(xmin + (0.2 * xmin)), xmax + (0.2 * xmax)])
                axes.set_ylim([(ymin + (0.2 * ymin)), ymax + (0.2 * ymax)])
            else:
                plt.axhline(y=t2, c='k', ls='--')
                plt.axhline(y=-t2, c='k', ls='--')
                plt.legend(['Hotelling $T^{2}$ 95% limit'])

        except (ValueError, IndexError) as verr:
            print("The number of components to plot must not exceed 2 and the component choice cannot "
                  "exceed the number of components in the model")
            raise Exception

        plt.title("PLS score plot")
        if len(comps) == 1:
            plt.xlabel("T[{0}]".format((comps[0] + 1)))
        else:
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
                plateau = np.min(np.where(np.diff(q2) / q2[0] < 0.05)[0])
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

    def plot_model_parameters(self, parameter='w', component=1, cross_val=False, sigma=2, bar=False, xaxis=None):

        choices = {'w': self.weights_w, 'c': self.weights_c, 'p': self.loadings_p, 'q': self.loadings_q,
                   'beta': self.beta_coeffs, 'ws': self.rotations_ws, 'cs': self.rotations_cs,
                   'VIP': self.VIP(), 'bu': self.b_u, 'bt': self.b_u}
        choices_cv = {'w': 'Weights_w', 'c': 'Weights_c', 'cs': 'Rotations_cs', 'ws': 'Rotations_ws',
                      'q': 'Loadings_q', 'p': 'Loadings_p', 'beta': 'Beta', 'VIP': 'VIP'}

        # decrement component to adjust for python indexing
        component -= 1
        # Beta and VIP don't depend on components so have an exception status here
        if cross_val is True:
            if parameter in ['beta', 'VIP']:
                mean = self.cvParameters['Mean_' + choices_cv[parameter]].squeeze()
                error = sigma * self.cvParameters['Stdev_' + choices_cv[parameter]].squeeze()
            else:
                mean = self.cvParameters['Mean_' + choices_cv[parameter]][:, component]
                error = sigma * self.cvParameters['Stdev_' + choices_cv[parameter]][:, component]
        else:
            error = None
            if parameter in ['beta', 'VIP']:
                mean = choices[parameter].squeeze()
            else:
                mean = choices[parameter][:, component]
        if bar is False:
            self._lineplots(mean, error=error, xaxis=xaxis)
        # To use with barplots for other types of data
        else:
            self._barplots(mean, error=error, xaxis=xaxis)

        plt.xlabel("Variable No")
        if parameter in ['beta', 'VIP']:
            plt.ylabel("{0} for PLS model".format(parameter))
        else:
            plt.ylabel("{0} for PLS component {1}".format(parameter, (component + 1)))
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

    def external_validation_set(self, x, y):

        y_pred = self.predict(x)

        validation_set_results = dict()

        return validation_set_results
