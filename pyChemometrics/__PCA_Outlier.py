# Requires this in fit for dmod_x
# For "Normalised" DmodX calculation
resid_ssx = self._residual_ssx(x)
s0 = np.sqrt(resid_ssx.sum() / ((self.scores.shape[0] - self.n_components - 1) * (x.shape[1] - self.n_components)))
self.modelParameters['S0'] = s0

def _residual_ssx(self, x):
    """

    :param x: Data matrix [n samples, m variables]
    :return: The residual Sum of Squares per sample
    """
    pred_scores = self.transform(x)

    x_reconstructed = self.scaler.transform(self.inverse_transform(pred_scores))
    xscaled = self.scaler.transform(x)
    residuals = np.sum((xscaled - x_reconstructed) ** 2, axis=1)
    return residuals


def x_residuals(self, x, scale=True):
    """

    :param x: data matrix [n samples, m variables]
    :param scale: Return the residuals in the scale the model is using or in the raw data scale
    :return: X matrix model residuals
    """
    pred_scores = self.transform(x)
    x_reconstructed = self.scaler.transform(self.inverse_transform(pred_scores))
    xscaled = self.scaler.transform(x)

    x_residuals = np.sum((xscaled - x_reconstructed) ** 2, axis=1)
    if scale:
        x_residuals = self.scaler.inverse_transform(x_residuals)

    return x_residuals


def dmodx(self, x):
    """

    Normalised DmodX measure

    :param x: data matrix [n samples, m variables]
    :return: The Normalised DmodX measure for each sample
    """
    resids_ssx = self._residual_ssx(x)
    s = np.sqrt(resids_ssx / (self.loadings.shape[1] - self.ncomps))
    dmodx = np.sqrt((s / self.modelParameters['S0']) ** 2)
    return dmodx


def leverages(self):
    """

    Calculate the leverages for each observation

    :return: The leverage (H) for each observation
    :rtype: numpy.ndarray
    """
    return np.diag(np.dot(self.scores, np.dot(np.linalg.inv(np.dot(self.scores.T, self.scores)), self.scores.T)))


def _dmodx_fcrit(self, x, alpha=0.05):
    """

    :param alpha:
    :return:
    """

    # Degrees of freedom for the PCA model (denominator in F-stat) calculated as suggested in
    # Faber, Nicolaas (Klaas) M., Degrees of freedom for the residuals of a
    # principal component analysis - A clarification, Chemometrics and Intelligent Laboratory Systems 2008
    dmodx_fcrit = st.f.ppf(1 - alpha, x.shape[1] - self.ncomps - 1,
                           (x.shape[0] - self.ncomps - 1) * (x.shape[1] - self.ncomps))

    return dmodx_fcrit


def outlier(self, x, comps=None, measure='T2', alpha=0.05):
    """

    Use the Hotelling T2 or DmodX measure and F statistic to screen for outlier candidates.

    :param x: Data matrix [n samples, m variables]
    :param comps: Which components to use (for Hotelling T2 only)
    :param measure: Hotelling T2 or DmodX
    :param alpha: Significance level
    :return: List with row indices of X matrix
    """
    try:
        if measure == 'T2':
            scores = self.transform(x)
            t2 = self.hotelling_T2(comps=comps)
            outlier_idx = np.where(((scores ** 2) / t2 ** 2).sum(axis=1) > 1)[0]
        elif measure == 'DmodX':
            dmodx = self.dmodx(x)
            dcrit = self._dmodx_fcrit(x, alpha)
            outlier_idx = np.where(dmodx > dcrit)[0]
        else:
            print("Select T2 (Hotelling T2) or DmodX as outlier exclusion criteria")
        return outlier_idx
    except Exception as exp:
        raise exp

    def hotelling_T2(self, comps=None, alpha=0.05):
        """

        Obtain the parameters for the Hotelling T2 ellipse at the desired significance level.

        :param list comps:
        :param float alpha: Significance level
        :return: The Hotelling T2 ellipsoid radii at vertex
        :rtype: numpy.ndarray
        :raise AtributeError: If the model is not fitted
        :raise ValueError: If the components requested are higher than the number of components in the model
        :raise TypeError: If comps is not None or list/numpy 1d array and alpha a float
        """

        try:
            if self._isfitted is False:
                raise AttributeError("Model is not fitted")
            nsamples = self.scores.shape[0]
            if comps is None:
                ncomps = self.ncomps
                ellips = self.scores[:, range(self.ncomps)] ** 2
                ellips = 1 / nsamples * (ellips.sum(0))
            else:
                ncomps = len(comps)
                ellips = self.scores[:, comps] ** 2
                ellips = 1 / nsamples * (ellips.sum(0))

            # F stat
            fs = (nsamples - 1) / nsamples * ncomps * (nsamples ** 2 - 1) / (nsamples * (nsamples - ncomps))
            fs = fs * st.f.ppf(1-alpha, ncomps, nsamples - ncomps)

            hoteling_t2 = list()
            for comp in range(ncomps):
                hoteling_t2.append(np.sqrt((fs * ellips[comp])))

            return np.array(hoteling_t2)

        except AttributeError as atre:
            raise atre
        except ValueError as valerr:
            raise valerr
        except TypeError as typerr:
            raise typerr

