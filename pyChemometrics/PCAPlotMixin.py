from abc import ABCMeta
import matplotlib.pyplot as plt
import matplotlib as mpl


class PCAPlotMixin(ABCMeta):
    """

    Mixin Class to add plotting methods to ChemometricsPCA objects if desired.

    """
    def __init__(self):
        self = super(self)

    def plot_scores(self, comps=[0, 1], color=None):

        try:
            plt.figure()
            comps = np.array(comps)

            t2 = np.array(self.hotelling_T2())
            outlier_idx = np.where(((self.scores ** 2) / t2 ** 2).sum(axis=1) > 1)[0]

            if len(comps) == 1:
                plt.scatter(range(self.scores.shape[0]), self.scores, color=color)
                plt.scatter(range(self.scores.shape[0]), self.scores[outlier_idx, comps[0]], color=color, marker='x',
                            s=1.5*mpl.rcParams['lines.markersize'] ** 2)
            else:

                plt.scatter(self.scores[:, comps[0]], self.scores[:, comps[1]], color=color)
                plt.scatter(self.scores[outlier_idx, comps[0]], self.scores[outlier_idx, comps[1]],
                            color=color, marker='x', s=1.5*mpl.rcParams['lines.markersize'] ** 2)

                t2 = np.array(self.hotelling_T2())

                angle = np.arange(-np.pi, np.pi, 0.01)
                x = t2[comps[0]] * np.cos(angle)
                y = t2[comps[1]] * np.sin(angle)
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

        except Exception as exp:
            print("Maximum number of components must be 2 and must match the number of components in the model")
            raise Exception

        plt.title("PCA score plot")
        if len(comps) == 1:
            plt.xlabel("PC[{0}] - Variance Explained : {1:.2} %".format((comps[0] + 1), self.modelParameters['VarExpRatio']))
        else:
            plt.xlabel("PC[{0}] - Variance Explained : {1:.2} %".format((comps[0] + 1), self.modelParameters['VarExpRatio'][comps[0]]))
            plt.ylabel("PC[{0}] - Variance Explained : {1:.2} %".format((comps[1] + 1), self.modelParameters['VarExpRatio'][comps[1]]))
        plt.show()
        return None

    def scree_plot(self, x, total_comps=5):

        plt.figure()
        models = list()
        for ncomps in range(1, total_comps + 1):
            currmodel = deepcopy(self)
            currmodel.ncomps = ncomps
            currmodel.fit(x)
            currmodel.cross_validation(x, outputdist=False, press_impute=False)
            models.append(currmodel)

        q2 = np.array([x.cvParameters['Q2'] for x in models])
        r2 = np.array([x.modelParameters['R2X'] for x in models])

        plt.bar([x - 0.1 for x in range(1, total_comps + 1)], height=r2, width=0.2)
        plt.bar([x + 0.1 for x in range(1, total_comps + 1)], height=q2, width=0.2)
        plt.legend(['R2', 'Q2'])
        plt.xlabel("Number of components")
        plt.ylabel("R2/Q2X")

        # Specific case where n comps = 2 # TODO fix this edge case
        if len(q2) == 2:
            plateau = np.min(np.where(np.diff(q2)/q2[0] < 0.05)[0])
        else:
            plateau = np.min(np.where(np.diff(q2)/q2[0:-1] < 0.05)[0])

        if plateau.size == 0:
            print("Consider exploring a higher level of components")
        else:
            plt.vlines(x= (plateau + 1), ymin=0, ymax=1, colors='red', linestyles ='dashed')
            print("Q2X measure stabilizes (increase of less than 5% of previous value or decrease) "
                    "at component {0}".format(plateau + 1))
        plt.show()

        return None

    def dmodx_plot(self, x):
        try:
            plt.figure()

            plt.show()
            return None
        except ValueError as verr:
            raise verr

    def outlier(self, x, measure='T2'):
        if measure == 'T2':
            scores = self.transform(x)
            t2 = np.array(self.hotelling_T2())

            outlier_idx = np.where(((scores ** 2) / t2 **2).sum(axis=1) > 1)[0]

        elif measure == 'DmodX':

            pass

        else:
            print("Select T2 (Hotelling T2) or DmodX as outlier exclusion criteria")

        return outlier_idx

    def plot_loadings(self, component=1, bar=False):

        # Adjust the indexing so user can refer to component 1 as component 1 instead of 0
        component -= 1
        plt.figure()

        # For "spectrum/continuous like plotting"
        if bar is False:
            ax = plt.plot(self.loadings[component, :])
            if self.cvParameters is not None:
                plt.fill_between(range(self.loadings[component, :].size),
                                 self.cvParameters['Mean_Loadings'][component] - 2*self.cvParameters['Stdev_Loadings'][component],
                                 self.cvParameters['Mean_Loadings'][component] + 2*self.cvParameters['Stdev_Loadings'][component],
                                 alpha=0.2, color='red')
        # To use with barplots for other types of data
        else:
            plt.bar(range(self.loadings[component, :].size), height=self.loadings[component, :], width=0.2)

        plt.xlabel("Variable No")
        plt.ylabel("Loading for PC{0}".format((component + 1)))
        plt.show()

        return None

    def plot_leverages(self):

        plt.figure()
        lev = self.leverages()
        plt.bar(left=range(lev.size), height=lev)
        plt.hlines(y=1/lev.size, xmin=0, xmax=lev.size, colors='r', linestyles='--')
        plt.show()

        return None
