import matplotlib.pyplot as plt
import numpy as np


def plotROC(rocCurve, xaxis, rocError=None, AUC=None):

    """
    :return: Figure with the Cross-Validated ROC curve and confidence interval
    """

    fig, ax = plt.subplots()

    ax.plot(np.append(np.array([0]), self.modelParameters['DA']['ROC'][0]),
             np.append(np.array([0]), self.cvParameters['DA']['Mean_ROC']), 'r-')

    upper = np.maximum(self.cvParameters['DA']['Mean_ROC'] - self.cvParameters['DA']['Stdev_ROC'], 0)
    lower = np.minimum(self.cvParameters['DA']['Mean_ROC'] + self.cvParameters['DA']['Stdev_ROC'], 1)
    ax.fill_between(np.append(np.array([0]), self.modelParameters['DA']['ROC'][0]), np.append(np.array([0]), lower),
                     np.append(np.array([0]), upper),
                     color='grey', alpha=0.2)

    ax.plot([0, 1], [0, 1], '--')
    ax.xlim([0, 1.00])
    ax.ylim([0, 1.05])
    ax.xlabel("False Positive Rate (1 - Specificity)")
    ax.ylabel("True Positive Rate (Sensitivity)")
    ax.show()
    print("Mean AUC: {0}".format(self.cvParameters['DA']['Mean_AUC']))

    return fig, ax


def plotConfusionMatrix():
    fig, ax = plt.subplots()
    return fig, ax


def plotExternalSetPredictions(x, y):
    """
    Plot a confusion matrix
    :param self:
    :param x:
    :param y:
    :return:
    """
    r2y_valid = self.score(x)
    y_pred = self.predict(x)

    validation_set_results = {'R2Y': r2y_valid, 'Y_predicted':y_pred}
    plt.figure()
    plt.scatter(y, y_pred)
    plt.xlabel('Original Y')
    plt.ylabel('Predicted Y')
    plt.show()
    return validation_set_results


def plotMisclassifications():
    fig, ax = plt.subplots()

    return fig, ax
