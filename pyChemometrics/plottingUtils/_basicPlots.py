import matplotlib.pyplot as plt


def _lineplots(x, error=None, xaxis=None):

    fig, ax = plt.subplots()
    if xaxis is None:
        ax.plot(x)
        xaxis = range(x.size)
    else:
        ax.plot(xaxis, x)
    if error is not None:
        ax.fill_between(xaxis, x - error, x + error, alpha=0.2, color='red')

    return fig, ax


def _barplots(x, error=None, xaxis=None):

    fig, ax = plt.subplots()
    if xaxis is None:
        xaxis = range(x.size)

    if error is None:
        ax.bar(xaxis, height=x)
    else:
        ax.bar(xaxis, height=x, yerr=error)
    return fig, ax


def _2DFeatureMap(x, y, color=None, xaxis=None, yaxis=None):

    if x is None or y is None:
        raise ValueError("2D map plots require 2-coordinate input")
    fig, ax = plt.subplots()
    if xaxis is None:
        xaxis = range(x.size)
    if yaxis is None:
        yaxis = range(y.size)

    if color is None:
        ax.scatter(x, y)

    else:
        ax.scatter(x, y, color=color)
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

def _draw_ellipse(fig, ax, x, y):
    # Update a scatterplot?
    fig.canvas.redraw()
    return fig, ax