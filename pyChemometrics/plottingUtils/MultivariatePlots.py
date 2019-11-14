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

        fig, ax = plt.subplot(111)

        # if only one component is passed, swap to barplot
        if y is None:
            y = x
            x = np.arange(0, y.shape[0])

        if color is None:
            ax.scatter(x, y)
            ax.scatter(x[outlierIdx], y[outlierIdx],
                        marker='x', s=1.5 * mpl.rcParams['lines.markersize'] ** 2)
        else:
            # Continuous color scale
            if discreteColor is False:
                cmap = cm.jet
                cnorm = Normalize(vmin=min(color), vmax=max(color))

                ax.scatter(x, y, c=color, cmap=cmap, norm=cnorm)
                ax.scatter(x[outlierIdx], y[outlierIdx],
                           c=color[outlierIdx], cmap=cmap, norm=cnorm, marker='x',
                           s=1.5 * mpl.rcParams['lines.markersize'] ** 2)
                ax.colorbar()
            # Discrete color scale
            else:
                cmap = cm.Set1
                subtypes = np.unique(color)

                for subtype in subtypes:
                    subset_index = np.where(color == subtype)

                    ax.scatter(x[subset_index], y[subset_index], c=cmap(subtype), label=subtype)
                ax.legend()
                ax.scatter(x[outlierIdx], y[outlierIdx],
                           c=color[outlierIdx], cmap=cmap, marker='x',
                           s=1.5 * mpl.rcParams['lines.markersize'] ** 2)

        if y is not None:

            angle = np.arange(-np.pi, np.pi, 0.01)
            x_t2 = hotellingT2[0] * np.cos(angle)
            y_t2 = hotellingT2[1] * np.sin(angle)
            ax.axhline(c='k')
            ax.axvline(c='k')
            ax.plot(x_t2, y_t2, c='k')

            xmin = np.minimum(min(x), np.min(x_t2))
            xmax = np.maximum(max(x), np.max(x_t2))
            ymin = np.minimum(min(y), np.min(y_t2))
            ymax = np.maximum(max(y), np.max(y_t2))

            #axes = ax.gca()
            ax.set_xlim([(xmin + (0.2 * xmin)), xmax + (0.2 * xmax)])
            ax.set_ylim([(ymin + (0.2 * ymin)), ymax + (0.2 * ymax)])
        else:
            ax.axhline(y=hotellingT2, c='k', ls='--')
            ax.axhline(y=hotellingT2, c='k', ls='--')
            ax.legend(['Hotelling $T^{2}$ 95% limit'])

    except (ValueError, IndexError) as verr:
        print("TZZZl")
        raise Exception

    # put this out of this function!
    ax.title("PCA score plot")
    if y is None:
        ax.xlabel("PC[{0}] - Variance Explained : {1:.2f} %".format((x + 1),
                                                                     self.modelParameters['VarExpRatio'] * 100))
    else:
        ax.xlabel("PC[{0}] - Variance Explained : {1:.2f} %".format((x + 1),
                                                                     self.modelParameters['VarExpRatio'][
                                                                         comps[0]] * 100))
        ax.ylabel("PC[{0}] - Variance Explained : {1:.2f} %".format((y + 1),
                                                                     self.modelParameters['VarExpRatio'][
                                                                         comps[1]] * 100))

    return fig, ax


def scorePlotTrellis(scoreMatrix, color=None, discrete=False, hotellingT2=False, outlierIdx=None):
    """

    Trellis plot for the score plots

    :param comps: Components to use in the 2D plot
    :param color: Variable used to color points
    :return: Score plot figure
    """
    try:

        fig, ax = plt.subplots(scoreMatrix.shape[1], scoreMatrix.shape[1])

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
    return fig, ax


def plotModelCoefficients(x, error=False, overlay=False, bar=False, xaxis=None):

    fig, ax = plt.subplots()

    if bar is False:
        ax = _lineplots(mean, error=error, xaxis=xaxis)
    # To use with barplots for other types of data
    else:
        ax._barplots(mean, error=error, xaxis=xaxis)

    ax.xlabel("Variable No")
    ax.ylabel("{0} for PCA component {1}".format(parameter, (component + 1)))

    return fig, ax


def biPlot():

    fig, ax = plt.subplots()
    return fig, ax


def screePlot(self, x, total_comps=5, cv_method=KFold(7, True)):
    """

    Plot of the R2X and Q2X per number of component to aid in the selection of the component number.

    :param x: Data matrix [n samples, m variables]
    :param total_comps: Maximum number of components to fit
    :param cv_method: scikit-learn Base Cross-Validator to use
    :return: Figure with R2X and Q2X Goodness of fit metrics per component
    """
    fig, ax = plt.subplots()
    models = list()

    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=DataConversionWarning)
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
    plt.legend(['R2X', 'Q2X'])
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

    return fig, ax


def plotOutlierDistance():
    fig, ax = plt.subplots()
    return fig, ax


def plotPermutationTest(permuationResults, nPerms=False, metric='Q2Y'):
    try:
        plt.figure()
        hst = plt.hist(permt_res[0][metric], 100)
        if metric == 'Q2Y':
            plt.vlines(x=self.cvParameters['Q2Y'], ymin=0, ymax=max(hst[0]))
        return None

    except KeyError:
        print("Run cross-validation before calling the plotting function")
    except Exception as exp:
        raise exp


def plotPredictedData(x=None, y=None):

    return None