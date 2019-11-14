import matplotlib.pyplot as plt
import numpy as np


def plotPredictions():
    fig, ax = plt.subplots()

    return fig, ax


def plotResiduals():
    fig, ax = plt.subplots()
    return fig, ax


def externalValidationPredictionsPlot(x, y):
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
