from abc import ABCMeta

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize


class PlotMixin(metaclass=ABCMeta):
    """

    Mixin Class containing general plotting methods.
    Underlying plotting mixin classes can re-use and override if needed.

    """

    @staticmethod
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

    @staticmethod
    def _barplots(mean, error=None, xaxis=None):
        fig, ax = plt.subplots()
        if xaxis is None:
            xaxis = range(mean.size)

        if error is None:
            ax.bar(xaxis, height=mean)
        else:
            ax.bar(xaxis, height=mean, yerr=error)
        return fig, ax

    @staticmethod
    def _scatterplot(x, y, colour=None, colourmap=cm, discrete_colour=False):
        fig, ax = plt.subplots()
        # Assemble C Map
        if discrete_colour:
            colour = Normalize(colour)
        else:
            colour = cm(colour)
        ax.scatter(x, y, c=colour, cm=colourmap)

        return fig, ax

    @staticmethod
    def _draw_ellipse(fig, ax, x, y):
        # Update a scatterplot?
        fig.canvas.redraw()
        return fig, ax