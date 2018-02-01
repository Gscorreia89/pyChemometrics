from abc import ABCMeta

from copy import deepcopy
import numpy as np
import pandas as pds
from scipy import interp
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, clone
from sklearn.exceptions import DataConversionWarning
from sklearn.cross_decomposition.pls_ import PLSRegression, _PLS
from sklearn.model_selection import BaseCrossValidator, KFold
from sklearn.model_selection._split import BaseShuffleSplit
from sklearn import metrics
from .ChemometricsPLS import ChemometricsPLS
from .ChemometricsScaler import ChemometricsScaler
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import seaborn as sns
import warnings


from pyChemometrics.PLSPlotMixin import PLSPlotMixin


# TODO Unfinished do not use


class PLSDAPlotMixin(PLSPlotMixin, metaclass=ABCMeta):
    """

    Mixin Class to add plotting methods to ChemometricsPLS objects if desired.

    """

    def scree_plot(self, x, y, total_comps=5):
        """

        :param x: Data to use in the scree plot
        :param y:
        :param total_comps:
        :return:
        """

        models = list()

        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=DataConversionWarning)
            for ncomps in range(1, total_comps + 1):
                currmodel = deepcopy(self)
                currmodel.ncomps = ncomps
                currmodel.fit(x, y)
                currmodel.cross_validation(x, y)
                models.append(currmodel)

        q2 = np.array([x.cvParameters['Q2Y'] for x in models])
        r2 = np.array([x.modelParameters['R2Y'] for x in models])

        plt.figure()
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

    def repeated_cv(self, x, y, total_comps=7, repeats=15, cv_method=KFold(7, True)):
        """

        Perform repeated cross-validation and plot Q2X values and their distribution (violin plot) per component
        number to help select the appropriate number of components.

        :param x: Data matrix [n samples, m variables]
        :param total_comps: Maximum number of components to fit
        :param repeats: Number of CV procedure repeats
        :param cv_method: scikit-learn Base Cross-Validator to use
        :return: Violin plot with Q2X values and distribution per component number.
        """

        q2y = np.zeros((total_comps, repeats))
        auc = np.zeros((total_comps, repeats))
        q2x = np.zeros((total_comps, repeats))

        # Suppress warnings after the first
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=DataConversionWarning)

            for ncomps in range(1, total_comps + 1):
                for rep in range(repeats):
                    currmodel = deepcopy(self)
                    currmodel.ncomps = ncomps
                    currmodel.fit(x, y)
                    currmodel.cross_validation(x, y, cv_method=cv_method, outputdist=False)
                    q2y[ncomps - 1, rep] = currmodel.cvParameters['Q2Y']
                    q2x[ncomps - 1, rep] = currmodel.cvParameters['Q2X']
                    auc[ncomps - 1, rep] = currmodel.cvParameters['DA']['Mean_AUC']

        plt.figure()
        ax = sns.violinplot(data=q2y.T, palette="Set1")
        ax2 = sns.swarmplot(data=q2y.T, edgecolor="black", color='black')
        ax2.set_xticklabels(range(1, total_comps + 1))
        plt.xlabel("Number of components")
        plt.ylabel("Q2Y")
        plt.show()

        return q2y, q2x

    def plot_cv_ROC(self):
        """
        :return: Figure with the Cross-Validated ROC curve and confidence interval
        """
        plt.figure()
        plt.plot(np.append(np.array([0]), self.modelParameters['DA']['ROC'][0]),
                 np.append(np.array([0]), self.cvParameters['DA']['Mean_ROC']), 'r-')

        upper = np.maximum(self.cvParameters['DA']['Mean_ROC'] - self.cvParameters['DA']['Stdev_ROC'], 0)
        lower = np.minimum(self.cvParameters['DA']['Mean_ROC'] + self.cvParameters['DA']['Stdev_ROC'], 1)
        plt.fill_between(np.append(np.array([0]), self.modelParameters['DA']['ROC'][0]), np.append(np.array([0]), lower),
                         np.append(np.array([0]), upper),
                         color='grey', alpha=0.2)

        plt.plot([0, 1], [0, 1], '--')
        plt.xlim([0, 1.00])
        plt.ylim([0, 1.05])
        plt.xlabel("False Positive Rate (1 - Specificity)")
        plt.ylabel("True Positive Rate (Sensitivity)")
        plt.show()
        print("Mean AUC: {0}".format(self.cvParameters['DA']['Mean_AUC']))
        return None

    def plot_permutation_test(self, permt_res, metric='AUC'):
        try:
            plt.figure()
            hst = plt.hist(permt_res[0][metric], 100)
            if metric == 'Q2Y':
                plt.vlines(x=self.cvParameters['Q2Y'], ymin=0, ymax=max(hst[0]))
            elif metric == 'AUC':
                plt.vlines(x=self.cvParameters['DA']['Mean_AUC'], ymin=0, ymax=max(hst[0]))
            elif metric == 'f1':
                plt.vlines(x=self.cvParameters['DA']['Mean_f1'], ymin=0, ymax=max(hst[0]))
            plt.show()
            return None

        except KeyError:
            print("Run cross-validation before calling the plotting function")
        except Exception as exp:
            raise exp

    def external_validation_set(self, x, y):

        r2y_valid = self.score(x)
        y_pred = self.predict(x)

        validation_set_results = {'R2Y': r2y_valid, 'Y_predicted':y_pred}
        plt.figure()
        plt.scatter(y, y_pred)
        plt.xlabel('Original Y')
        plt.ylabel('Predicted Y')
        plt.show()
        return validation_set_results
