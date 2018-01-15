from abc import ABCMeta

import matplotlib.pyplot as plt
import numpy as np

from pyChemometrics.PLSPlotMixin import PLSPlotMixin


# TODO Unfinished do not use


class PLSDAPlotMixin(PLSPlotMixin, metaclass=ABCMeta):
    """

    Mixin Class to add plotting methods to ChemometricsPLS objects if desired.

    """


    def plot_cv_ROC(self):
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
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.show()
        return None

    def plot_permutation_test(self, permt_res, metric='AUC'):
        try:
            plt.figure()
            hst = plt.hist(permt_res[0][metric], 100)
            if metric == 'Q2Y':
                plt.vlines(x=self.cvParameters['PLS']['Q2Y'], ymin=0, ymax=max(hst[0]))
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

        y_pred = self.predict(x)

        validation_set_results = dict()

        return validation_set_results
