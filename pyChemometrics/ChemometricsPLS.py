from sklearn.base import RegressorMixin
from sklearn.cross_decomposition.pls_ import PLSRegression, _PLS
from sklearn.model_selection import BaseCrossValidator, KFold
from sklearn.model_selection._split import BaseShuffleSplit
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from sklearn.base import clone
from .ChemometricsScaler import ChemometricsScaler
import copy
from copy import deepcopy


__author__ = 'gd2212'


class ChemometricsPLS(BaseEstimator, RegressorMixin, TransformerMixin):
    """

    ChemometricsPLS object - Wrapper for sklearn.cross_decomposition PLS algorithms, with tailored methods
    for Chemometric Data analysis.

    :param int ncomps: Number of PLS components desired.
    :param sklearn._PLS pls_algorithm: Scikit-learn PLS algorithm to use - PLSRegression or PLSCanonical are supported.
    :param TransformerMixin xscaler: Scaler object (ChemometricsScaler or scaling/preprocessing objects from scikit-learn) to scale the x data block.
    :param TransformerMixin yscaler: Scaler object (ChemometricsScaler or scaling/preprocessing objects from scikit-learn) to scale the y data block.
    :param **kwargs pls_type_kwargs: Keyword arguments to be passed during initialization of pls_algorithm.
    :return self: instance of ChemometricsPLS object
    """

    """
    This object is designed to fit flexibly both PLSRegression with one or multiple Y and PLSCanonical, both
    with either NIPALS or SVD. PLS-SVD doesn't calculate the same type of model parameters, and should
    not be used with this object.
    For PLSRegression/PLS1/PLS2 and PLSCanonical/PLS-C2A/PLS-W2A, the actual components
    found may differ (depending on type of deflation, etc), and this has to be taken into consideration,
    but the actual nomenclature/definitions should be the "same".
    Nomenclature is as follows:
    X - T Scores - Projections of X, called T
    Y - U Scores - Projections of Y, called U
    X - Loadings P - Vector/multivariate directions associated with T on X are called P (equivalent to PCA)
    Y - Loadings Q - Vector/multivariate directions associated with U on Y are called q
    X - Weights W - Weights/directions of maximum covariance with Y of the X block are called W
    Y - Weights C - Weights/directions of maximum covariance with X of the Y block block are called C
    These "y-weights" tend to be almost negligible (not in the calculation though... ) of PLS1/PLS Regression
    but are necessary in multi-Y/SVD-PLS/PLS2
    X - Rotations W*/Ws/R - The rotation of X variables to LV space pinv(WP')W
    Y - Rotations C*/Cs - The rotation of Y variables to LV space pinv(CQ')C
    T = X W(P'W)^-1 = XW* (W* : p x k matrix)
    U = Y C(Q'C)^-1 = YC* (C* : q x k matrix)
    Loadings and weights after the first component do not represent
    the original variables. The SIMPLS W*/Ws/R and C*/Cs act weight vectors
    which relate to the original X and Y variables, and not to their deflated versions.
    See Sijmen de Jong, "SIMPLS: an alternative approach to partial least squares regression", Chemometrics
    and Intelligent Laboratory Systems 1992
    "Inner" relation regression coefficients of T b_t: U = Tb_t
    "Inner" relation regression coefficients of U b_U: T = Ub_u
    These are obtained by regressing the U's and T's, applying standard linear regression to them.
    B = pinv(X'X)X'Y
    b_t = pinv(T'T)T'U
    b_u = pinv(U'U)U'T
    or in a more familiar form: b_t are the betas from regressing T on U - t'u/u'u
    and b_u are the betas from regressing U on T - u't/t't

    In summary, there are various ways to approach the model. Following a general nomenclature applicable
    for both single and block Y:
    The predictive model, assuming the Latent variable formulation, uses an "inner relation"
    between the latent variable projections, where U = Tb_t and T = Ub_u.
    Prediction using the so-called "mixed relations" (relate T with U and subsequently Y/relate
    U with T and subsequently X), works through the following formulas
    Y = T*b_t*C' + G
    X = U*b_u*W' + H
    The b_u and b_s are effectively "regression coefficients" between the latent variable scores

    In parallel, we can think in terms of "outer relations", data decompositions or linear approximations to
    the original data blocks, similar to PCA components
    Y = UQ' + F
    X = TP' + E
    For PLS regression with single y, Y = UC' + F = Y = UQ' + F, due to Q = C, but not necessarily true for
    multi Y, so Q' is used here. Notice that this formula cannot be used directly to
    predict Y from X and vice-versa, the inner relation regression using latent variable scores is necessary.

    Finally, assuming PLSRegression (single or multi Y, but asymmetric deflation):
    The PLS model can be approached from a multivariate regression/regularized regression point of view,
    where Y is related to the original X variables, through regression coefficients Beta,
    bypassing the latent variable definition and concepts.
    Y = XBQ', Y = XB, where B are the regression coefficients and B = W*Q' (the W*/ws is the SIMPLS_R,
    X rotation in sklearn default PLS).
    These Betas (regression coefficients) are obtained in this manner directly relate the original X variables
    to the prediction of Y.

    This MLR approach to PLS has the advantage of exposing the PLS betas and PLS mechanism
    as a biased regression applying a degree of shrinkage, which decreases with the number of components
    all the way up to B(OLS), when Number of Components = number of variables/columns.
    See Frank and Friedman, Jong and Kramer/Rosipal

    """

    def __init__(self, ncomps=2, pls_algorithm=PLSRegression, xscaler=ChemometricsScaler(), yscaler=None,
                 **pls_type_kwargs):

        try:

            # Perform the check with is instance but avoid abstract base class runs.
            pls_algorithm = pls_algorithm(ncomps, scale=False, **pls_type_kwargs)
            if not isinstance(pls_algorithm, (BaseEstimator, _PLS)):
                raise TypeError("Scikit-learn model please")
            if not (isinstance(xscaler, TransformerMixin) or xscaler is None):
                raise TypeError("Scikit-learn Transformer-like object or None")
            if not (isinstance(yscaler, TransformerMixin) or yscaler is None):
                raise TypeError("Scikit-learn Transformer-like object or None")
            # 2 blocks of data = two scaling options
            if xscaler is None:
                xscaler = ChemometricsScaler(0, with_std=False)
                # Force scaling to false, as this will be handled by the provided scaler or not
            if yscaler is None:
                yscaler = ChemometricsScaler(0, with_std=False)

            self.pls_algorithm = pls_algorithm
            # Most initialized as None, before object is fitted...
            self.scores_t = None
            self.scores_u = None
            self.weights_w = None
            self.weights_c = None
            self.loadings_p = None
            self.loadings_q = None
            self.rotations_ws = None
            self.rotations_cs = None
            self.b_u = None
            self.b_t = None
            self.beta_coeffs = None

            self._ncomps = None
            self.ncomps = ncomps
            self._x_scaler = None
            self._y_scaler = None
            self.x_scaler = xscaler
            self.y_scaler = yscaler
            self.cvParameters = None
            self.modelParameters = None
            self._isfitted = False

        except TypeError as terp:
            print(terp.args[0])
        except ValueError as verr:
            print(verr.args[0])

    def fit(self, x, y, **fit_params):
        """

        Perform model fitting on the provided x and y data and calculate basic goodness-of-fit metrics.
        Equivalent to scikit-learn's BaseEstimator method.

        :param numpy.ndarray x: X data matrix.
        :param numpy.ndarray y: Y data vector/matrix.
        :param **kwargs fit_params: Optional keyword arguments to be passed to the pls_algorithm .fit method.
        """
        try:
            # This scaling check is always performed to ensure running model with scaling or with scaling == None
            # always gives consistent results (the same type of data scale used fitting will be expected or returned
            # by all methods of the ChemometricsPLS object)
            # For no scaling, mean centering is performed nevertheless - sklearn objects
            # do this by default, this is solely to make everything ultra clear and to expose the
            # interface for potential future modification
            # Comply with the sklearn-scaler behaviour convention
            if y.ndim == 1:
                y = y.reshape(-1, 1)
            if x.ndim == 1:
                x = x.reshape(-1, 1)

            xscaled = self.x_scaler.fit_transform(x)
            yscaled = self.y_scaler.fit_transform(y)

            self.pls_algorithm.fit(xscaled, yscaled, **fit_params)

            # Expose the model parameters
            self.loadings_p = self.pls_algorithm.x_loadings_
            self.loadings_q = self.pls_algorithm.y_loadings_
            self.weights_w = self.pls_algorithm.x_weights_
            self.weights_c = self.pls_algorithm.y_weights_
            self.rotations_ws = self.pls_algorithm.x_rotations_
            # scikit learn sets the rotation, causing a discrepancy between the scores calculated during fitting and the transform method
            # for now, we calculate the rotation and override it: C* = pinv(CQ')C
            self.rotations_cs = np.dot(np.linalg.pinv(np.dot(self.weights_c, self.loadings_q.T)), self.weights_c)
            self.scores_t = self.pls_algorithm.x_scores_
            self.scores_u = self.pls_algorithm.y_scores_
            self.b_u = np.dot(np.dot(np.linalg.pinv(np.dot(self.scores_u.T, self.scores_u)), self.scores_u.T), self.scores_t)
            self.b_t = np.dot(np.dot(np.linalg.pinv(np.dot(self.scores_t.T, self.scores_t)), self.scores_t.T), self.scores_u)
            self.beta_coeffs = self.pls_algorithm.coef_
            # Needs to come here for the method shortcuts down the line to work...
            self._isfitted = True


            # Calculate RSSy/RSSx, R2Y/R2X
            R2Y = self.score(x=x, y=y, block_to_score='y')
            R2X = self.score(x=x, y=y, block_to_score='x')

            # Obtain residual sum of squares for whole data set and per component
            cm_fit = self._cummulativefit(self.ncomps, x, y)

            self.modelParameters = {'R2Y': R2Y, 'R2X': R2X, 'SSX': cm_fit['SSX'], 'SSY': cm_fit['SSY'],
                                    'SSXcomp': cm_fit['SSXcomp'], 'SSYcomp': cm_fit['SSYcomp']}

        except Exception as exp:
            raise exp

    def fit_transform(self, x, y, **fit_params):
        """

        Fit a model to supplied data and return the scores. Equivalent to scikit-learn's TransformerMixin method.

        :param numpy.ndarray x: X data matrix
        :param numpy.nadarray y: Y data matrix
        :param **kwargs fit_params: Optional keyword arguments to be passed to the pls_algorithm .fit method.
        :return tuple: Latent Variable scores (T) for the X matrix and for the Y vector/matrix (U).
        """

        try:
            self.fit(x, y, **fit_params)
            # Comply with the sklearn scaler behaviour - y vector doesn't work
            if y.ndim == 1:
                y = y.reshape(-1, 1)
            if x.ndim == 1:
                x = x.reshape(-1, 1)

            xscaled = self.x_scaler.fit_transform(x)
            yscaled = self.y_scaler.fit_transform(y)

            return self.transform(xscaled, y=None), self.transform(x=None, y=yscaled)

        except Exception as exp:
            raise exp

    def transform(self, x=None, y=None):
        """

        Calculate the scores for a data block from the original data. Equivalent to sklearn's TransformerMixin method.

        :param (numpy.ndarray or None) x: x data matrix
        :param (numpy.ndarray or None) y: y data vector/matrix
        :return tuple: Latent Variable scores (T) for the X matrix and for the Y vector/matrix (U).
        """

        try:

            # Check if model is fitted
            if self._isfitted is True:
                # If X and Y are passed, complain and do nothing
                if (x is not None) and (y is not None):
                    raise ValueError('xx')
                # If nothing is passed at all, complain and do nothing
                elif (x is None and y is None):
                    raise ValueError('yy')
                # If Y is given, return U
                elif x is None:
                    if y.ndim == 1:
                        y = y.reshape(-1, 1)

                    yscaled = self.y_scaler.transform(y)
                    # Taking advantage of rotations_y
                    # Otherwise this would be the full calculation U = Y*pinv(CQ')*C
                    U = np.dot(yscaled, self.rotations_cs)
                    return U

                # If X is given, return T
                elif y is None:
                    # Comply with the sklearn scaler behaviour
                    if x.ndim == 1:
                        x = x.reshape(-1, 1)

                    xscaled = self.x_scaler.transform(x)
                    # Taking advantage of the rotation_x
                    # Otherwise this would be would the full calculation T = X*pinv(WP')*W
                    T = np.dot(xscaled, self.rotations_ws)
                    return T
            else:
                raise ValueError('Model not fitted')

        except ValueError as verr:
            raise verr
        except Exception as exp:
            raise exp

    def inverse_transform(self, t=None, u=None):
        """

        Transform scores to the original data space using their corresponding loadings.
        Equivalent to sklearn's TransformerMixin method.

        :param (numpy.ndarray or None) t: T scores corresponding to the X data matrix
        :param (numpy.ndarry or None) u: Y scores corresponding to the Y data vector/matrix
        :return:
        """
        try:
            if self._isfitted is True:
                if t is not None and u is not None:
                    raise ValueError('xx')
                # If nothing is passed at all, complain and do nothing
                elif t is None and u is None:
                    raise ValueError('yy')
                # If T is given, return U
                elif t is not None:
                    # Calculate X from T using X = TP'
                    xpred = np.dot(t, self.loadings_p.T)
                    if self.x_scaler is not None:
                        xscaled = self.x_scaler.inverse_transform(xpred)
                    else:
                        xscaled = xpred

                    return xscaled
                # If U is given, return T
                elif u is not None:
                    # Calculate Y from U - using Y = UQ'
                    ypred = np.dot(u, self.loadings_q.T)
                    if self.y_scaler is not None:
                        yscaled = self.y_scaler.inverse_transform(ypred)
                    else:
                        yscaled = ypred

                    return yscaled

        except ValueError as verr:
            raise verr
        except Exception as exp:
            raise exp

    def score(self, x, y, block_to_score='y', sample_weight=None):
        """

        Predict and calculate the R2 for the model using one of the data blocks (X or Y) provided.
        Equivalent to scikit-learn RegressorMixin score method.

        :param numpy.ndarray x:
        :param numpy.nadarry y:
        :param str block_to_score:
        :param None sample_weight: Not yet implemented
        :return:
        """
        # TO DO: actually use sample_weight
        try:
            if block_to_score not in ['x', 'y']:
                raise ValueError("x or y are the only accepted values for block_to_score")
            # Comply with the sklearn scaler behaviour
            if y.ndim == 1:
                y = y.reshape(-1, 1)
            if x.ndim == 1:
                x = x.reshape(-1, 1)

            xscaled = self.x_scaler.transform(x)
            yscaled = self.y_scaler.transform(y)

            # Calculate total sum of squares of X and Y for R2X and R2Y calculation
            tssy = np.sum(yscaled ** 2)
            tssx = np.sum(xscaled ** 2)

            # Calculate RSSy/RSSx, R2Y/R2X
            # The prediction here of both X and Y is done using the other block of data only
            # so these R2s can be interpreted as as a "classic" R2, and not as a proportion of variance modelled
            # Here we use X = Ub_uW', as opposed to (X = TP').
            ypred = self.y_scaler.transform(self.predict(x, y=None))
            xpred = self.x_scaler.transform(self.predict(x=None, y=y))
            rssy = np.sum((yscaled - ypred)**2)
            rssx = np.sum((xscaled - xpred)**2)
            R2Y = 1 - (rssy/tssy)
            R2X = 1 - (rssx/tssx)

            if block_to_score == 'y':
                return R2Y
            else:
                return R2X

        except ValueError as verr:
            raise verr

    def predict(self, x=None, y=None):
        """

        Predict the value of one data block using the other block. Equivalent to scikit-learn RegressorMixin method.

        :param (numpy.ndarray or None) x:
        :param (numpy.ndarray or None) y:
        :return numpy.ndarray prediction: The prediction of one of the data matrices using the one provided
        """

        try:
            if self._isfitted is True:
                if (x is not None) and (y is not None):
                    raise ValueError('xx')
                # If nothing is passed at all, complain and do nothing
                elif (x is None) and (y is None):
                    raise ValueError('yy')
                # Predict Y from X
                elif x is not None:
                    if x.ndim == 1:
                        x = x.reshape(-1, 1)
                    xscaled = self.x_scaler.transform(x)

                    # Using Betas to predict Y directly
                    predicted = np.dot(xscaled, self.beta_coeffs)
                    if predicted.ndim == 1:
                        predicted = predicted.reshape(-1, 1)
                    predicted = self.y_scaler.inverse_transform(predicted)
                    return predicted
                # Predict X from Y
                elif y is not None:
                    # Comply with the sklearn scaler behaviour
                    if y.ndim == 1:
                        y = y.reshape(-1, 1)
                    # Going through calculation of U and then X = Ub_uW'
                    u_scores = self.transform(x=None, y=y)
                    predicted = np.dot(np.dot(u_scores, self.b_u), self.weights_w.T)
                    if predicted.ndim == 1:
                        predicted = predicted.reshape(-1, 1)
                    predicted = self.x_scaler.inverse_transform(predicted)
                    return predicted
            else:
                raise ValueError("Model is not fitted")
        except Exception as exp:
            raise exp

    @property
    def ncomps(self):
        """

        Getter for number of components.

        :return int ncomps: The model's number of PLS components.
        """

        try:
            return self._ncomps
        except AttributeError as atre:
            raise atre

    @ncomps.setter
    def ncomps(self, ncomps=1):
        """

        Setter for number of components. Resets internal state of the model (model will have to be refitted).

        :param int ncomps: The desired new number of PLS components to set.
        """
        # To ensure changing number of components effectively resets the model
        try:
            self._ncomps = ncomps
            self.pls_algorithm = clone(self.pls_algorithm, safe=True)
            self.pls_algorithm.n_components = ncomps
            self.loadings_p = None
            self.scores_t = None
            self.scores_u = None
            self.loadings_q = None
            self.weights_c = None
            self.weights_w = None
            self.rotations_cs = None
            self.rotations_ws = None
            self.cvParameters = None
            self.modelParameters = None
            self.b_t = None
            self.b_u = None
            self.beta_coeffs = None

            return None
        except AttributeError as atre:
            raise atre

    @property
    def x_scaler(self):
        """

        Getter for the model x_scaler.

        :return:
        """
        try:
            return self._x_scaler
        except AttributeError as atre:
            raise atre

    @x_scaler.setter
    def x_scaler(self, scaler):
        """

        Setter for the model x_scaler. Resets internal state of the model (model will have to be refitted).

        :param TransformerMixin scaler: Sklearn scaler object
        :return:
        """
        try:

            if not (isinstance(scaler, TransformerMixin) or scaler is None):
                raise TypeError("Scikit-learn Transformer-like object or None")
            if scaler is None:
                scaler = ChemometricsScaler(0, with_std=False)

            self._x_scaler = scaler
            self.pls_algorithm = clone(self.pls_algorithm, safe=True)
            self.modelParameters = None
            self.cvParameters = None
            self.loadings_p = None
            self.weights_w = None
            self.weights_c = None
            self.loadings_q = None
            self.rotations_ws = None
            self.rotations_cs = None
            self.scores_t = None
            self.scores_u = None
            self.b_t = None
            self.b_u = None
            self.beta_coeffs = None

            return None
        except AttributeError as atre:
            raise atre
        except TypeError as typerr:
            raise typerr

    @property
    def y_scaler(self):
        """

        Getter for the model y_scaler

        :return:
        """
        try:
            return self._y_scaler
        except AttributeError as atre:
            raise atre

    @y_scaler.setter
    def y_scaler(self, scaler):
        """

        Setter for the model y_scaler. Resets internal state of the model (model will have to be refitted).

        :param TransformerMixin scaler: Sklearn scaler object
        :return:
        """
        try:
            if not (isinstance(scaler, TransformerMixin) or scaler is None):
                raise TypeError("Scikit-learn Transformer-like object or None")
            if scaler is None:
                scaler = ChemometricsScaler(0, with_std=False)

            self._y_scaler = scaler
            self.pls_algorithm = clone(self.pls_algorithm, safe=True)
            self.modelParameters = None
            self.cvParameters = None
            self.loadings_p = None
            self.weights_w = None
            self.weights_c = None
            self.loadings_q = None
            self.rotations_ws = None
            self.rotations_cs = None
            self.scores_t = None
            self.scores_u = None
            self.b_t = None
            self.b_u = None
            self.beta_coeffs = None


            return None

        except AttributeError as atre:
            raise atre
        except TypeError as typerr:
            raise typerr

    def VIP(self, mode='w', direction='y'):
        """

        Output the Variable importance for projection metric (VIP). With the default values it is calculated
        using the x variable weights and the variance explained of y.

        :param str mode: The type of model parameter to use in calculating the VIP. Default value is weights (w).
        :param str direction: The data block to be used to calculated the model fit and regression sum of squares
        :return numpy.ndarray VIP: The vector with the calculated VIP values.
        """

        try:

            if self._isfitted is False:
                raise AttributeError("Model is not fitted")
            if mode not in ['w', 'p', 'ws', 'cs', 'c', 'q']:
                raise ValueError("Invalid type of VIP coefficient")
            if direction not in ['x', 'y']:
                raise ValueError("direction must be x or y")

            choices = {'w': self.weights_w, 'p': self.loadings_p, 'ws': self.rotations_ws, 'cs': self.rotations_cs,
                       'c': self.weights_c, 'q': self.loadings_q}

            if direction == 'y':
                ss_dir = 'SSYcomp'
            else:
                ss_dir = 'SSXcomp'

            nvars = self.loadings_p.shape[0]
            vipnum = np.zeros(nvars)
            for comp in range(0, self.ncomps):
                vipnum += (choices[mode][:, comp] ** 2) * (self.modelParameters[ss_dir][comp])

            vip = np.sqrt(vipnum * nvars / self.modelParameters[ss_dir].sum())

            return vip

        except AttributeError as atre:
            raise AttributeError("Model not fitted")
        except ValueError as verr:
            raise verr

    def hotelling_T2(self, comps):
        """

        Obtain the parameters for the Hotelling T2 ellipse at the desired significance level.

        :param comps:
        :return:
        """
        try:
            if self._isfitted is False:
                raise AttributeError("Model is not fitted")
            for comp in comps:
                self.scores_t[:, comp]
            hoteling = 1
            return hoteling

        except AttributeError as atre:
            raise atre
        except ValueError as valerr:
            raise valerr
        except TypeError as typerr:
            raise typerr

    def dModX(self):
        """

        :return:
        """
        return NotImplementedError

    def leverages(self):
        """

        :return:
        """
        return NotImplementedError

    def cross_validation(self, x, y,  cv_method=KFold(7, True), outputdist=False, testset_scale=False,
                         **crossval_kwargs):
        """

        Cross-validation method for the model. Calculates Q2 and cross-validated estimates of model parameters.

        :param numpy.ndarray x: X data matrix
        :param numpy.ndarray y: Y data vector/matrix
        :param (BaseCrossValidator or BaseShuffleSplit) cv_method: scikit-learn cross-validation object.
        :param Boolean outputdist: Return the whole distribution of model parameters or just summary statistics.
        :param Boolean testset_scale: Use the scaling derived from the training set or re-scale the test set separately.
        :param **kwargs crossval_kwargs: Keyword arguments to be passed to the .fit method during cross-validation
        """

        try:
            if not (isinstance(cv_method, BaseCrossValidator) or isinstance(cv_method, BaseShuffleSplit)):
                raise TypeError("Scikit-learn cross-validation object please")

            # Check if global model is fitted... and if not, fit it using all of X
            if self._isfitted is False:
                self.fit(x, y)

            # Make a copy of the object, to ensure the internal state doesn't come out differently from the
            # cross validation method call...
            cv_pipeline = copy.deepcopy(self)
            ncvrounds = cv_method.get_n_splits()

            if x.ndim > 1:
                x_nvars = x.shape[1]
            else:
                x_nvars = 1

            if y.ndim > 1:
                y_nvars = y.shape[1]
            else:
                y_nvars = 1
                y = y.reshape(-1, 1)

            # Initialize list structures to contain the fit
            cv_loadings_p = np.zeros((ncvrounds, x_nvars, self.ncomps))
            cv_loadings_q = np.zeros((ncvrounds, y_nvars, self.ncomps))
            cv_weights_w = np.zeros((ncvrounds, x_nvars, self.ncomps))
            cv_weights_c = np.zeros((ncvrounds, y_nvars, self.ncomps))
            #cv_scores_t = np.zeros((ncvrounds, x.shape[0], self.ncomps))
            #cv_scores_u = np.zeros((ncvrounds, y.shape[0], self.ncomps))
            cv_rotations_ws = np.zeros((ncvrounds, x_nvars, self.ncomps))
            cv_rotations_cs = np.zeros((ncvrounds, y_nvars, self.ncomps))
            cv_betacoefs = np.zeros((ncvrounds, x_nvars))
            cv_vipsw = np.zeros((ncvrounds, x_nvars))

            # Initialise predictive residual sum of squares variable (for whole CV routine)
            pressy = 0
            pressx = 0

            # Calculate Sum of Squares SS in whole dataset for future calculations
            ssx = np.sum((cv_pipeline.x_scaler.fit_transform(x)) ** 2)
            ssy = np.sum((cv_pipeline.y_scaler.fit_transform(y)) ** 2)

            # As assessed in the test set..., opposed to PRESS
            R2X_training = np.zeros(ncvrounds)
            R2Y_training = np.zeros(ncvrounds)
            # R2X and R2Y assessed in the test set
            R2X_test = np.zeros(ncvrounds)
            R2Y_test = np.zeros(ncvrounds)

            for cvround, train_testidx in enumerate(cv_method.split(x, y)):
                # split the data explicitly
                train = train_testidx[0]
                test = train_testidx[1]

                # Check dimensions for the indexing
                if y_nvars == 1:
                    ytrain = y[train]
                    ytest = y[test]
                else:
                    ytrain = y[train, :]
                    ytest = y[test, :]
                if x_nvars == 1:
                    xtrain = x[train]
                    xtest = x[test]
                else:
                    xtrain = x[train, :]
                    xtest = x[test, :]

                cv_pipeline.fit(xtrain, ytrain, **crossval_kwargs)
                # Prepare the scaled X and Y test data
                # If testset_scale is True, these are scaled individually...

                # Comply with the sklearn scaler behaviour
                if ytest.ndim == 1:
                    ytest = ytest.reshape(-1, 1)
                    ytrain = ytrain.reshape(-1, 1)
                if xtest.ndim == 1:
                    xtest = xtest.reshape(-1, 1)
                    xtrain = xtrain.reshape(-1, 1)
                # Fit the training data

                if testset_scale is True:
                    xtest_scaled = cv_pipeline.x_scaler.fit_transform(xtest)
                    ytest_scaled = cv_pipeline.y_scaler.fit_transform(ytest)
                # Otherwise (default), training set mean and scaling vectors are used
                else:
                    xtest_scaled = cv_pipeline.x_scaler.transform(xtest)
                    ytest_scaled = cv_pipeline.y_scaler.transform(ytest)

                R2X_training[cvround] = cv_pipeline.score(xtrain, ytrain, 'x')
                R2Y_training[cvround] = cv_pipeline.score(xtrain, ytrain, 'y')
                ypred = cv_pipeline.predict(x=xtest, y=None)
                xpred = cv_pipeline.predict(x=None, y=ytest)

                xpred = cv_pipeline.x_scaler.transform(xpred).squeeze()

                ypred = cv_pipeline.y_scaler.transform(ypred).squeeze()
                ytest_scaled = ytest_scaled.squeeze()

                curr_pressx = np.sum((xtest_scaled - xpred)**2)
                curr_pressy = np.sum((ytest_scaled - ypred)**2)

                R2X_test[cvround] = cv_pipeline.score(xtest, ytest, 'x')
                R2Y_test[cvround] = cv_pipeline.score(xtest, ytest, 'y')

                pressx += curr_pressx
                pressy += curr_pressy

                cv_loadings_p[cvround, :, :] = cv_pipeline.loadings_p
                cv_loadings_q[cvround, :, :] = cv_pipeline.loadings_q
                cv_weights_w[cvround, :, :] = cv_pipeline.weights_w
                cv_weights_c[cvround, :, :] = cv_pipeline.weights_c
                #cv_scores_t[cvround, :, :] = cv_pipeline.scores_t
                #cv_scores_u[cvround, :, :] = cv_pipeline.scores_u
                cv_rotations_ws[cvround, :, :] = cv_pipeline.rotations_ws
                cv_rotations_cs[cvround, :, :] = cv_pipeline.rotations_cs
                cv_betacoefs[cvround, :] = cv_pipeline.beta_coeffs.T
                cv_vipsw[cvround, :] = cv_pipeline.VIP()

            # Align model parameters to account for sign indeterminacy.
            # The criteria here used is to select the sign that gives a more similar profile (by L1 distance) to the loadings fitted
            # on the model fitted with the whole data. Any other parameter can be used, but since the loadings in X capture
            # the covariance structure in X data block, in theory they should have more pronounced features even in cases of
            # null X-Y association, making the sign flip more resilient.
            for cvround in range(0, ncvrounds):
                for currload in range(0, self.ncomps):
                    # evaluate based on loadings _p
                    choice = np.argmin(np.array([np.sum(np.abs(self.loadings_p[:, currload] - cv_loadings_p[cvround, :, currload])),
                                                 np.sum(np.abs(self.loadings_p[:, currload] - cv_loadings_p[cvround, :, currload] * -1))]))
                    if choice == 1:
                        cv_loadings_p[cvround, :, currload] = -1 * cv_loadings_p[cvround, :, currload]
                        cv_loadings_q[cvround, :, currload] = -1 * cv_loadings_q[cvround, :, currload]
                        cv_weights_w[cvround, :, currload] = -1 * cv_weights_w[cvround, :, currload]
                        cv_weights_c[cvround, :, currload] = -1 * cv_weights_c[cvround, :, currload]
                        cv_rotations_ws[cvround, :, currload] = -1 * cv_rotations_ws[cvround, :, currload]
                        cv_rotations_cs[cvround, :, currload] = -1 * cv_rotations_cs[cvround, :, currload]
                        #cv_scores_t[cvround, currload, :] = -1 * cv_scores_t[cvround, currload, :]
                        #cv_scores_u[cvround, currload, :] = -1 * cv_scores_u[cvround, currload, :]

            # Calculate total sum of squares
            q_squaredy = 1 - (pressy/ssy)
            q_squaredx = 1 - (pressx/ssx)

            # Store everything...
            self.cvParameters = {'Q2X': q_squaredx, 'Q2Y': q_squaredy,
                                 'MeanR2X_Training':np.mean(R2X_training),
                                 'MeanR2Y_Training': np.mean(R2Y_training),
                                 'StdevR2X_Training': np.std(R2X_training),
                                 'StdevR2Y_Training': np.std(R2X_training),
                                 'MeanR2X_Test': np.mean(R2X_test),
                                 'MeanR2Y_Test': np.mean(R2Y_test),
                                 'StdevR2X_Test': np.std(R2X_test),
                                 'StdevR2Y_Test': np.std(R2Y_test)}

            # Means and standard deviations...
            self.cvParameters['Mean_Loadings_q'] = cv_loadings_q.mean(0)
            self.cvParameters['Stdev_Loadings_q'] = cv_loadings_q.std(0)
            self.cvParameters['Mean_Loadings_p'] = cv_loadings_p.mean(0)
            self.cvParameters['Stdev_Loadings_p'] = cv_loadings_q.std(0)
            self.cvParameters['Mean_Weights_c'] = cv_weights_c.mean(0)
            self.cvParameters['Stdev_Weights_c'] = cv_weights_c.std(0)
            self.cvParameters['Mean_Weights_w'] = cv_weights_w.mean(0)
            self.cvParameters['Stdev_Loadings_w'] = cv_weights_w.std(0)
            self.cvParameters['Mean_Rotations_ws'] = cv_rotations_ws.mean(0)
            self.cvParameters['Stdev_Rotations_ws'] = cv_rotations_ws.std(0)
            self.cvParameters['Mean_Rotations_cs'] = cv_rotations_cs.mean(0)
            self.cvParameters['Stdev_Rotations_cs'] = cv_rotations_cs.std(0)
            #self.cvParameters['Mean_Scores_t'] = cv_scores_t.mean(0)
            #self.cvParameters['Stdev_Scores_t'] = cv_scores_t.std(0)
            #self.cvParameters['Mean_Scores_u'] = cv_scores_u.mean(0)
            #self.cvParameters['Stdev_Scores_u'] = cv_scores_u.std(0)
            self.cvParameters['Mean_Beta'] = cv_betacoefs.mean(0)
            self.cvParameters['Stdev_Beta'] = cv_betacoefs.std(0)
            self.cvParameters['Mean_VIP'] = cv_vipsw.mean(0)
            self.cvParameters['Stdev_VIP'] = cv_vipsw.std(0)
            # Save everything found during CV
            if outputdist is True:
                self.cvParameters['CVR2X_Training'] = R2X_training
                self.cvParameters['CVR2Y_Training'] = R2Y_training
                self.cvParameters['CVR2X_Test'] = R2X_test
                self.cvParameters['CVR2Y_Test'] = R2Y_test
                self.cvParameters['CV_Loadings_q'] = cv_loadings_q
                self.cvParameters['CV_Loadings_p'] = cv_loadings_p
                self.cvParameters['CV_Weights_c'] = cv_weights_c
                self.cvParameters['CV_Weights_w'] = cv_weights_w
                self.cvParameters['CV_Rotations_ws'] = cv_rotations_ws
                self.cvParameters['CV_Rotations_cs'] = cv_rotations_cs
                #self.cvParameters['CV_Scores_t'] = cv_scores_t
                #self.cvParameters['CV_Scores_u'] = cv_scores_u
                self.cvParameters['CV_Beta'] = cv_betacoefs
                self.cvParameters['CV_VIPw'] = cv_vipsw

            return None

        except TypeError as terp:
            raise terp

    def permutation_test(self, x, y, nperms=1000, cv_method=KFold(7, True), **permtest_kwargs):
        """

        Permutation test for the classifier. Outputs permuted null distributions for model performance metrics (Q2X/Q2Y)
        most model parameters.

        :param numpy.ndarray x: x data matrix.
        :param numpy.ndarray y: Y data vector/matrix.
        :param int nperms: Number of permutations to perform.
        :param (BaseCrossValidator or BaseShuffleSplit) cv_method: scikit-learn cross-validation object.
        :param **kwargs permtest_kwargs: Keyword arguments to be passed to the .fit method during cross-validation
        and model fitting.
        :return tuple: Permuted null distributions for model parameters and the permutation p-value for the Q2Y value.
        """
        try:
            # Check if global model is fitted... and if not, fit it using all of X
            if self._isfitted is False or self.loadings_p is None:
                self.fit(x, y, **permtest_kwargs)
            # Make a copy of the object, to ensure the internal state doesn't come out differently from the
            # cross validation method call...
            permute_class = copy.deepcopy(self)

            if x.ndim > 1:
                x_nvars = x.shape[1]
            else:
                x_nvars = 1

            if y.ndim > 1:
                y_nvars = y.shape[1]
            else:
                y_nvars = 1

            # Initialize data structures for permuted distributions
            perm_loadings_q = np.zeros((nperms, y_nvars, self.ncomps))
            perm_loadings_p = np.zeros((nperms, x_nvars, self.ncomps))
            perm_weights_c = np.zeros((nperms, y_nvars, self.ncomps))
            perm_weights_w = np.zeros((nperms, x_nvars, self.ncomps))
            perm_rotations_cs = np.zeros((nperms, y_nvars, self.ncomps))
            perm_rotations_ws = np.zeros((nperms, x_nvars, self.ncomps))
            perm_beta = np.zeros((nperms, x_nvars, y_nvars))
            perm_vipsw = np.zeros((nperms, x_nvars))

            permuted_R2Y = np.zeros(nperms)
            permuted_R2X = np.zeros(nperms)
            permuted_Q2Y = np.zeros(nperms)
            permuted_Q2X = np.zeros(nperms)
            permuted_R2Y_test = np.zeros(nperms)
            permuted_R2X_test = np.zeros(nperms)

            for permutation in range(0, nperms):
                # Copy original column order, shuffle array in place...
                original_Y = np.copy(y)
                np.random.shuffle(y)
                # ... Fit model and replace original data
                permute_class.fit(x, y, **permtest_kwargs)
                permute_class.cross_validation(x, y, cv_method=cv_method, **permtest_kwargs)
                y = original_Y
                permuted_R2Y[permutation] = permute_class.modelParameters['R2Y']
                permuted_R2X[permutation] = permute_class.modelParameters['R2X']
                permuted_Q2Y[permutation] = permute_class.cvParameters['Q2Y']
                permuted_Q2X[permutation] = permute_class.cvParameters['Q2X']

                # Store the loadings for each permutation component-wise
                perm_loadings_q[permutation, :, :] = permute_class.loadings_q
                perm_loadings_p[permutation, :, :] = permute_class.loadings_p
                perm_weights_c[permutation, :, :] = permute_class.weights_c
                perm_weights_w[permutation, :, :] = permute_class.weights_w
                perm_rotations_cs[permutation, :, :] = permute_class.rotations_cs
                perm_rotations_ws[permutation, :, :] = permute_class.rotations_ws
                perm_beta[permutation, :, :] = permute_class.beta_coeffs
                perm_vipsw[permutation, :] = permute_class.VIP()
            # Align model parameters due to sign indeterminacy.
            # Solution provided is to select the sign that gives a more similar profile to the
            # Loadings calculated with the whole data.
            for perm_round in range(0, nperms):
                for currload in range(0, self.ncomps):
                    # evaluate based on loadings _p
                    choice = np.argmin(np.array([np.sum(np.abs(self.loadings_p[:, currload] - perm_loadings_p[perm_round, :, currload])),
                                                 np.sum(np.abs(self.loadings_p[:, currload] - perm_loadings_p[perm_round, :, currload] * -1))]))
                    if choice == 1:
                        perm_loadings_p[perm_round, :, currload] = -1 * perm_loadings_p[perm_round, :, currload]
                        perm_loadings_q[perm_round, :, currload] = -1 * perm_loadings_q[perm_round, :, currload]
                        perm_weights_w[perm_round, :, currload] = -1 * perm_weights_w[perm_round, :, currload]
                        perm_weights_c[perm_round, :, currload] = -1 * perm_weights_c[perm_round, :, currload]
                        perm_rotations_ws[perm_round, :, currload] = -1 * perm_rotations_ws[perm_round, :, currload]
                        perm_rotations_cs[perm_round, :, currload] = -1 * perm_rotations_cs[perm_round, :, currload]

            # Pack everything into a nice data structure and return
            # Calculate p-value for Q2Y as well
            permutationTest = dict()
            permutationTest['R2Y'] = permuted_R2Y
            permutationTest['R2X'] = permuted_R2X
            permutationTest['Q2Y'] = permuted_Q2Y
            permutationTest['Q2X'] = permuted_Q2X
            permutationTest['R2Y_Test'] = permuted_R2Y_test
            permutationTest['R2X_Test'] = permuted_R2X_test
            permutationTest['Loadings_p'] = perm_loadings_p
            permutationTest['Loadings_q'] = perm_loadings_q
            permutationTest['Weights_c'] = perm_weights_c
            permutationTest['Weights_w'] = perm_weights_w
            permutationTest['Rotations_ws'] = perm_rotations_ws
            permutationTest['Rotations_cs'] = perm_rotations_cs
            permutationTest['Beta'] = perm_beta
            permutationTest['VIPw'] = perm_vipsw

            obs_q2y = self.cvParameters['Q2Y']
            pvals = dict()
            pvals['Q2Y'] = (len(np.where(permuted_Q2Y >= obs_q2y)) + 1)/(nperms+1)

            return permutationTest, pvals

        except Exception as exp:
            raise exp

    def _cummulativefit(self, ncomps, x, y):
        """

        Measure the cumulative Regression sum of Squares for each individual component.

        :param int ncomps: First Number of components to
        :param numpy.ndarray x: X data block to score the model
        :param numpy.ndarray y: Y data block to score the model
        :return cumulative_fit: dictionary object containing the total Regression Sum of Squares and the Sum of Squares per components, for both the X and Y data blocks.
        """
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        xscaled = self.x_scaler.fit_transform(x)
        yscaled = self.y_scaler.fit_transform(y)

        ssx_comp = list()
        ssy_comp = list()

        # Obtain residual sum of squares for whole data set and per component
        SSX = np.sum(xscaled ** 2)
        SSY = np.sum(yscaled ** 2)
        ssx_comp = list()
        ssy_comp = list()

        for curr_comp in range(1, self.ncomps + 1):
            model = self._reduce_ncomps(curr_comp)

            ypred = self.y_scaler.transform(model.predict(x, y=None))
            xpred = self.x_scaler.transform(model.predict(x=None, y=y))
            rssy = np.sum((yscaled - ypred) ** 2)
            rssx = np.sum((xscaled - xpred) ** 2)
            ssx_comp.append(rssx)
            ssy_comp.append(rssy)

        cumulative_fit = {'SSX': SSX, 'SSY': SSY, 'SSXcomp': np.array(ssx_comp), 'SSYcomp': np.array(ssy_comp)}

        return cumulative_fit

    def _reduce_ncomps(self, ncomps):
        """

        Generate a new model with a smaller set of components.

        :param int ncomps: Number of ordered first N components from the original model to be kept. Must be smaller than the ncomps value of the original model.
        :return newmodel: ChemometricsPLS object with reduced number of components.
        """
        try:
            if ncomps > self.ncomps:
                raise ValueError('Fit a new model with more components instead')

            newmodel = copy.deepcopy(self)
            newmodel._ncomps = ncomps

            newmodel.modelParameters = None
            newmodel.cvParameters = None
            newmodel.loadings_p = self.loadings_p[:, 0:ncomps]
            newmodel.weights_w = self.weights_w[:, 0:ncomps]
            newmodel.weights_c = self.weights_c[:, 0:ncomps]
            newmodel.loadings_q = self.loadings_q[:, 0:ncomps]
            newmodel.rotations_ws = self.rotations_ws[:, 0:ncomps]
            newmodel.rotations_cs = self.rotations_cs[:, 0:ncomps]
            newmodel.scores_t = None
            newmodel.scores_u = None
            newmodel.b_t = self.b_t[0:ncomps, 0:ncomps]
            newmodel.b_u = self.b_u[0:ncomps, 0:ncomps]

            # These have to be recalculated from the rotations
            newmodel.beta_coeffs = np.dot(newmodel.rotations_ws, newmodel.loadings_q.T)
            # Line also in the original sklearn method, but unnecessary when scaling = False - kept here for testing...
            #newmodel.beta_coeffs = (1. / newmodel.x_scaler.scale_.reshape((newmodel.x_scaler.scale_.shape[0], 1)) *
            #                        newmodel.beta_coeffs * newmodel.y_scaler.scale_)

            return newmodel

        except Exception as exp:
            raise exp

    def __deepcopy__(self, memo):
        """
        Deep copy implementation
        :param memo:
        :return:
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result
