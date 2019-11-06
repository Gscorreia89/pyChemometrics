import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import matplotlib as mpl


def scorePlot(x, y, color=None, discreteColor=False, hotellingT2=False, outlierIdx=None):
    """

    Score plot figure wth an Hotelling T2.

    :param comps: Components to use in the 2D plot
    :param color: Variable used to color points
    :return: Score plot figure
    """
    try:

        plt.figure()

        # Use a constant color if no color argument is passed
        outlierIdx = 1

        if y is None:
            y = x
            x = np.arange(0, y.shape[0])
        else:
            x_coord = x
            y_coord = y

        if color is None:
            plt.scatter(x_coord, y_coord)
            plt.scatter(x_coord[outlier_idx], y_coord[outlier_idx],
                        marker='x', s=1.5 * mpl.rcParams['lines.markersize'] ** 2)
        else:

            if discreteColor is False:
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
                plt.scatter(x_coord[outlierIdx], y_coord[outlierIdx],
                            c=color[outlierIdx], cmap=cmap, marker='x',
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

    plt.title("PCA score plot")
    if len(comps) == 1:
        plt.xlabel("PC[{0}] - Variance Explained : {1:.2f} %".format((comps[0] + 1),
                                                                     self.modelParameters['VarExpRatio'] * 100))
    else:
        plt.xlabel("PC[{0}] - Variance Explained : {1:.2f} %".format((comps[0] + 1),
                                                                     self.modelParameters['VarExpRatio'][
                                                                         comps[0]] * 100))
        plt.ylabel("PC[{0}] - Variance Explained : {1:.2f} %".format((comps[1] + 1),
                                                                     self.modelParameters['VarExpRatio'][
                                                                         comps[1]] * 100))
    plt.show()
    return None


def scorePlotTrellis(scoreMatrix, color=None, discrete=False, hotellingT2=False, outlierIdx=None):
    """

    Trellis plot for the score plots

    :param comps: Components to use in the 2D plot
    :param color: Variable used to color points
    :return: Score plot figure
    """
    try:

        plt.figure()

        # Use a constant color if no color argument is passed
        outlierIdx = 1

        if len(comps) == 1:
            x_coord = np.arange(0, self.scores.shape[0])
            y_coord = self.scores[:, comps[0]]
        else:
            x_coord = self.scores[:, comps[0]]
            y_coord = self.scores[:, comps[1]]

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

    plt.title("PCA score plot")
    if len(comps) == 1:
        plt.xlabel("PC[{0}] - Variance Explained : {1:.2f} %".format((comps[0] + 1),
                                                                     self.modelParameters['VarExpRatio'] * 100))
    else:
        plt.xlabel("PC[{0}] - Variance Explained : {1:.2f} %".format((comps[0] + 1),
                                                                     self.modelParameters['VarExpRatio'][
                                                                         comps[0]] * 100))
        plt.ylabel("PC[{0}] - Variance Explained : {1:.2f} %".format((comps[1] + 1),
                                                                     self.modelParameters['VarExpRatio'][
                                                                         comps[1]] * 100))
    plt.show()
    return None
