import matplotlib.pyplot as plt


def _lineplots(mean, error=None, xaxis=None):

    fig, ax = plt.subplots()
    if xaxis is None:
        ax.plot(mean)
        xaxis = range(mean.size)
    else:
        ax.plot(xaxis, mean)
    if error is not None:
        ax.fill_between(xaxis, mean - error, mean + error, alpha=0.2, color='red')

    return fig, ax


def _barplots(mean, error=None, xaxis=None):

    fig, ax = plt.subplots()
    if xaxis is None:
        xaxis = range(mean.size)

    if error is None:
        ax.bar(xaxis, height=mean)
    else:
        ax.bar(xaxis, height=mean, yerr=error)
    return fig, ax


def _2DFeatureMap(x, y, color=None, xaxis=None, yaxis=None):

    fig, ax = plt.subplots()
    if xaxis is None:
        xaxis = range(mean.size)

    if error is None:
        ax.bar(xaxis, height=mean)
    else:
        ax.bar(xaxis, height=mean, yerr=error)
    return fig, ax


def _scatterplots(x, y, xaxis=None, yaxis=None, color):

    fig, ax = plt.subplots()
    if xaxis is None:
        ax.scatter(mean)
        xaxis = range(mean.size)
    else:
        ax.plot(xaxis, mean)
    if error is not None:
        ax.fill_between(xaxis, mean - error, mean + error, alpha=0.2, color='red')

    return fig, ax