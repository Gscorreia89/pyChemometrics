from abc import ABCMeta
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import scipy.stats as st
from sklearn.model_selection import KFold
from copy import deepcopy
from pyChemometrics.PlotMixin import PlotMixin


class PCAPlotMixin(PlotMixin, metaclass=ABCMeta):
    """

    Mixin Class to add plotting methods to ChemometricsPCA objects if desired.

    """

    def plot_scores(self, comps=[0, 1], color=None):
        """

        Score plot figure wth an Hotelling T2.

        :param comps: Components to use in the 2D plot
        :param color: Variable used to color points
        :return: Score plot figure
        """
        try:
            plt.figure()
            if color is not None:
                cmap = cm.get_cmap(name='Set1')
                color = cmap(color)
                comps = np.array(comps)

            t2 = self.hotelling_T2(alpha=0.05, comps=comps)
            outlier_idx = np.where(((self.scores[:, comps] ** 2) / t2 ** 2).sum(axis=1) > 1)[0]

            if len(comps) == 1:
                plt.scatter(range(self.scores.shape[0]), self.scores[:, comps], color=color)
                plt.scatter(range(self.scores.shape[0]), self.scores[outlier_idx, comps[0]], color=color, marker='x',
                            s=1.5*mpl.rcParams['lines.markersize'] ** 2)
            else:
                plt.scatter(self.scores[:, comps[0]], self.scores[:, comps[1]], color=color)
                plt.scatter(self.scores[outlier_idx, comps[0]], self.scores[outlier_idx, comps[1]],
                            color=color, marker='x', s=1.5*mpl.rcParams['lines.markersize'] ** 2)

                t2 = self.hotelling_T2(comps=comps)

                angle = np.arange(-np.pi, np.pi, 0.01)
                x = t2[0] * np.cos(angle)
                y = t2[1] * np.sin(angle)
                plt.axhline(c='k')
                plt.axvline(c='k')
                plt.plot(x, y, c='k')

                xmin = np.minimum(min(self.scores[:, comps[0]]), np.min(x))
                xmax = np.maximum(max(self.scores[:, comps[0]]), np.max(x))
                ymin = np.minimum(min(self.scores[:, comps[1]]), np.min(y))
                ymax = np.maximum(max(self.scores[:, comps[1]]), np.max(y))

                axes = plt.gca()
                axes.set_xlim([(xmin + (0.2 * xmin)), xmax + (0.2 * xmax)])
                axes.set_ylim([(ymin + (0.2 * ymin)), ymax + (0.2 * ymax)])

        except (ValueError, IndexError) as verr:
            print("The number of components to plot must not exceed 2 and the component choice cannot "
                  "exceed the number of components in the model")
            raise Exception

        plt.title("PCA score plot")
        if len(comps) == 1:
            plt.xlabel("PC[{0}] - Variance Explained : {1:.2} %".format((comps[0] + 1), self.modelParameters['VarExpRatio']*100))
        else:
            plt.xlabel("PC[{0}] - Variance Explained : {1:.2} %".format((comps[0] + 1), self.modelParameters['VarExpRatio'][comps[0]]*100))
            plt.ylabel("PC[{0}] - Variance Explained : {1:.2} %".format((comps[1] + 1), self.modelParameters['VarExpRatio'][comps[1]]*100))
        plt.show()
        return None

    def plot_model_parameters(self, parameter='p', component=1, cross_val=False, sigma=2, bar=False, xaxis=None):

        choices = {'p': self.loadings}
        choices_cv = {'p': 'Loadings'}

        # decrement component to adjust for python indexing
        component -= 1
        if cross_val is True:
            mean = self.cvParameters['Mean_' + choices_cv[parameter]][component, :]
            error = sigma * self.cvParameters['Stdev_' + choices_cv[parameter]][component, :]
        else:
            error = None
            mean = choices[parameter][component, :]

        if bar is False:
            self._lineplots(mean, error=error, xaxis=xaxis)
        # To use with barplots for other types of data
        else:
            self._barplots(mean, error=error, xaxis=xaxis)

        plt.xlabel("Variable No")
        plt.ylabel("{0} for PCA component {1}".format(parameter, (component + 1)))
        plt.show()

        return None

    def scree_plot(self, x, total_comps=5, cv_method=KFold(7, shuffle=True)):
        """

        Plot of the R2X and Q2X per number of component to aid in the selection of the component number.

        :param x: Data matrix [n samples, m variables]
        :param total_comps: Maximum number of components to fit
        :param cv_method: scikit-learn Base Cross-Validator to use
        :return: Figure with R2X and Q2X Goodness of fit metrics per component
        """
        plt.figure()
        models = list()

        for ncomps in range(1, total_comps + 1):
            currmodel = deepcopy(self)
            currmodel.ncomps = ncomps
            currmodel.fit(x)
            currmodel.cross_validation(x, outputdist=False, cv_method=cv_method)
            models.append(currmodel)

        q2 = np.array([x.cvParameters['Q2X'] for x in models])
        r2 = np.array([x.modelParameters['R2X'] for x in models])

        plt.bar([x - 0.1 for x in range(1, total_comps + 1)], height=r2, width=0.2)
        plt.bar([x + 0.1 for x in range(1, total_comps + 1)], height=q2, width=0.2)
        plt.legend(['R2', 'Q2'])
        plt.xlabel("Number of components")
        plt.ylabel("R2/Q2X")

        # Specific case where n comps = 2 # TODO check this edge case
        if len(q2) == 2:
            plateau = np.min(np.where(np.diff(q2)/q2[0] < 0.05)[0])
        else:
            percent_cutoff = np.where(np.diff(q2) / q2[0:-1] < 0.05)[0]
            if percent_cutoff.size == 0:
                print("Consider exploring a higher level of components")
            else:
                plateau = np.min(percent_cutoff)
                plt.vlines(x= (plateau + 1), ymin=0, ymax=1, colors='red', linestyles ='dashed')
                print("Q2X measure stabilizes (increase of less than 5% of previous value or decrease) "
                      "at component {0}".format(plateau + 1))
        plt.show()

        return None

    def repeated_cv(self, x, total_comps=7, repeats=15, cv_method=KFold(7, shuffle=True)):
        """

        Perform repeated cross-validation and plot Q2X values and their distribution (violin plot) per component
        number to help select the appropriate number of components.

        :param x: Data matrix [n samples, m variables]
        :param total_comps: Maximum number of components to fit
        :param repeats: Number of CV procedure repeats
        :param cv_method: scikit-learn Base Cross-Validator to use
        :return: Violin plot with Q2X values and distribution per component number.
        """

        q2x = np.zeros((total_comps, repeats))

        for ncomps in range(1, total_comps + 1):
            for rep in range(repeats):
                currmodel = deepcopy(self)
                currmodel.ncomps = ncomps
                currmodel.fit(x)
                currmodel.cross_validation(x, cv_method=cv_method, outputdist=False)
                q2x[ncomps - 1, rep] = currmodel.cvParameters['Q2X']

        plt.figure()
        ax = sns.violinplot(data=q2x.T, palette="Set1")
        ax = sns.swarmplot(data=q2x.T, edgecolor="black", color='black')
        ax.set_xticklabels(range(1, total_comps + 1))
        plt.xlabel("Number of components")
        plt.ylabel("Q2X")
        plt.show()

        return q2x

    def plot_dmodx(self, x, alpha=0.05):
        """

        Plot a figure with DmodX values and the F-statistic critical line.

        :param numpy.ndarray x: Data matrix [n samples, m variables]
        :param float alpha: Significance level
        :return: Plot with DmodX values and critical line
        """

        try:
            dmodx = self.dmodx(x)

            dcrit = self._dmodx_fcrit(x, alpha)
            outlier_idx = self.outlier(x, measure='DmodX')
            plt.figure()
            x_axis = np.array([x for x in range(x.shape[0])])
            plt.plot(x_axis, dmodx, 'o')
            plt.plot(x_axis[outlier_idx], dmodx[outlier_idx], 'rx')
            plt.xlabel('Sample Index')
            plt.ylabel('DmodX')
            plt.hlines(dcrit, xmin=0, xmax= x.shape[0], color='r', linestyles='--')
            plt.show()
            return None
        except TypeError as terr:
            raise terr
        except ValueError as verr:
            raise verr

    def plot_leverages(self):
        """
        Leverage (h) per observation, with a red line plotted at y = 1/Number of samples (expected
        :return: Plot with observation leverages (h)
        """
        plt.figure()
        lev = self.leverages()
        plt.xlabel('Sample Index')
        plt.ylabel('Leverage')
        plt.bar(left=range(lev.size), height=lev)
        plt.hlines(y=1/lev.size, xmin=0, xmax=lev.size, colors='r', linestyles='--')
        plt.show()
        return None