from sklearn.base import RegressorMixin
from sklearn.cross_decomposition.pls_ import PLSRegression, _PLS
from sklearn.pipeline import Pipeline
from sklearn.model_selection import BaseCrossValidator, KFold
import pandas as pds
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from sklearn.base import clone
from ChemometricsScaler import ChemometricsScaler
import copy
from copy import deepcopy


__author__ = 'gd2212'


class ChemometricsPLS(BaseEstimator, RegressorMixin, TransformerMixin):
    """
    This object is designed to fit flexibly both PLSRegression with one or multiple Y and PLSCanonical, both
    with either NIPALS or SVD. PLS-SVD provides a slightly different type of factorization, and should
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
    U = Y C(Q'C)^-1 = YC* (W* : q x k matrix)
    Loadings and weights after the first component do not represent
    the original variables. The SIMPLS W*/Ws/R and C*/Cs act weight vectors
    which relate to the original X and Y variables, and not to their deflated versions
    (order of the component becomes non-essential).
    See Sijmen de Jong, "SIMPLS: an alternative approach to partial least squares regression", Chemometrics
    and Intelligent Laboratory Systems 1992
    "Inner" relation regression coefficients of T b_t: U = Tb_t
    "Inner" relation regression coefficients of U b_U: T = Ub_u
    These are obtained by regressing the U's and T's, applying standard linear regression to them
    (inverse or pseudo-inverse can be used, but pseudo-inverse is more general)
    B = pinv(X'X)X'Y
    b_t = pinv(T'T)T'U
    b_u = pinv(U'U)U'T
    or in a more familiar form: b_t are the betas from regressing T on U - t'u/u'u
    and b_u are the betas from regressing U on T - u't/t't

    In summary, there are various ways to approach the model (following a general nomenclature applicable
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
    where Y is related to the original X variables, bypassing the latent variable definition and concepts.
    Y = XBQ', Y = XB, where B are the regression coefficients and B = W*Q' (the W*/ws is the SIMPLS_R,
    X rotation in sklearn default PLS, and C*/cs plays a similar role for Y).
    These Betas (regression coefficients) are obtained in this manner directly relate the original X variables
    to the prediction of Y.
    Question: Is there an equivalent formulation for the X, applicable in multi-Y settings as well?!?
    This MLR approach to PLS has the advantage of exposing the PLS betas and PLS mechanism
    as a biased regression applying a degree of shrinkage, which decreases with the number of components
    all the way up to B(OLS), when Number of Components = number of variables/columns.
    See Frank and Friedman, Jong and Kramer/Rosipal

    A component of OPLS is also provided:

    """

    def __init__(self, ncomps=2, pls_algorithm=PLSRegression, xscaler=ChemometricsScaler(), yscaler=None,
                 metadata=None, **pls_type_kwargs):
        """

        :param ncomps:
        :param pls_algorithm:
        :param xscaler:
        :param yscaler:
        :param metadata:
        :param pls_type_kwargs:
        """
        try:

            # Metadata assumed to be pandas dataframe only
            if (metadata is not None) and (metadata is not isinstance(metadata, pds.DataFrame)):
                raise TypeError("Metadata must be provided as pandas dataframe")

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

            # Shall we have OPLS here?

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
        Fit function. Acts exactly as in scikit-learn, but
        :param x:
        :param scale:
        :return:

        """
        try:
            # This scaling check is always performed to ensure running model with scaling or with scaling == None
            # always gives consistent results (the same type of data scale used fitting will be expected or returned
            # by all methods of the ChemometricsPLS object)
            # For no scaling, mean centering is performed nevertheless - sklearn objects
            # do this by default, this is solely to make everything ultra clear and to expose the
            # interface for potential future modification
            # (which might involve having to modifying the sklearn automatic scaling in the core objects...)
            if y.ndim == 1:
                y = y[np.newaxis]
            if x.ndim == 1:
                x = x[np.newaxis]
            if self.x_scaler is not None:
                xscaled = self.x_scaler.fit_transform(x)
            else:
                xscaled = ChemometricsScaler(0, with_std=False).fit_transform(x)
            if self.y_scaler is not None:
                yscaled = self.y_scaler.fit_transform(y)
            else:
                yscaled = ChemometricsScaler(0, with_std=False).fit_transform(y)

            self.pls_algorithm.fit(xscaled, yscaled, **fit_params)
            # Expose the model parameters
            self.loadings_p = self.pls_algorithm.x_loadings_
            self.loadings_q = self.pls_algorithm.y_loadings_
            self.loadings_q = self.pls_algorithm.y_loadings_
            self.weights_w = self.pls_algorithm.x_weights_
            self.weights_c = self.pls_algorithm.y_weights_
            self.rotations_ws = self.pls_algorithm.x_rotations_
            self.rotations_cs = self.pls_algorithm.y_rotations_
            self.scores_t = self.pls_algorithm.x_scores_
            self.scores_u = self.pls_algorithm.y_scores_
            self.b_u = np.dot(np.dot(np.linalg.pinv(np.dot(self.scores_u.T, self.scores_u)), self.scores_u.T), self.scores_t)
            self.b_t = np.dot(np.dot(np.linalg.pinv(np.dot(self.scores_t.T, self.scores_t)), self.scores_t.T), self.scores_u)

            # Calculate total sum of squares of X and Y for R2X and R2Y calculation
            tssy = np.sum(yscaled ** 2)
            tssx = np.sum(xscaled ** 2)

            # Calculate RSSy/RSSx, R2Y/R2X
            ypred = self.y_scaler.transform(self.predict(x=x, y=None))
            xpred = self.x_scaler.transform(self.predict(x=None, y=y))

            rssy = np.sum((yscaled - ypred)**2)
            rssx = np.sum((xscaled - xpred)**2)
            R2Y = self.score(x=x, y=None, block_to_score='y')
            R2X = self.score(x=None, y=y, block_to_score='x')

            self.modelParameters = {'R2Y': R2Y, 'R2X': R2X}
            self._isfitted = True

        except Exception as exp:
            raise exp

    def fit_transform(self, x, y, **fit_params):
        """
        Obtain scores in X
        :param x: Data to fit
        :param fit_params:
        :return:
        """
        try:
            self.fit(x, y, **fit_params)
            if y.ndim == 1:
                y = y[np.newaxis]
            if x.ndim == 1:
                x = x[np.newaxis]
            if self.x_scaler is not None:
                xscaled = self.x_scaler.fit_transform(x)
            else:
                xscaled = ChemometricsScaler(0, with_std=False).fit_transform(x)
            if self.y_scaler is not None:
                yscaled = self.y_scaler.fit_transform(y)
            else:
                yscaled = ChemometricsScaler(0, with_std=False).fit_transform(y)

            return self.transform(xscaled, y=None), self.transform(x=None, y=yscaled)

        except Exception as exp:
            raise exp

    def transform(self, x=None, y=None, **transform_kwargs):
        """
        Calculate the projection of the data into the lower dimensional space
        TO DO as PLS does not contain this...
        :param x:
        :return:
        """
        try:

            if x.ndim == 1:
                x = x[np.newaxis]
            # Check if model is fitted
            if self._isfitted is True:
                # If X and Y are passed, complain and do nothing
                if x is not None and y is not None:
                    raise ValueError('xx')
                # If nothing is passed at all, complain and do nothing
                elif x is None and y is None:
                    raise ValueError('yy')
                # If Y is given, return U
                elif x is None:
                    if y.ndim == 1:
                        y = y[np.newaxis]
                    if self.y_scaler is not None:
                        yscaled = self.y_scaler.fit_transform(y)
                    else:
                        yscaled = ChemometricsScaler(0, with_std=False).fit_transform(y)
                    # Taking advantage of rotations_y
                    # Otherwise this would be the full calculation U = Y*pinv(CQ')*C
                    U = np.dot(yscaled, self.rotations_cs)
                    return U
                # If X is given, return T
                elif y is None:
                    if x.ndim == 1:
                        x = x[np.newaxis]
                    if self.x_scaler is not None:
                        xscaled = self.x_scaler.fit_transform(x)
                    else:
                        xscaled = ChemometricsScaler(0, with_std=False).fit_transform(x)
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

        :param t:
        :param u:
        :return:
        """
        try:

            if self._isfitted is True:
                if t is not None and u is not None:
                    raise ValueError('xx')
                # If nothing is passed at all, complain and do nothing
                elif t is None and u is None:
                    raise ValueError('yy')
                # If  is given, return U
                elif t is not None:
                    # Calculate X from T using X = TP'
                    xpred = np.dot(t, self.loadings_p.T)
                    if self.x_scaler is not None:
                        xscaled = self.x_scaler.inverse_transform(xpred)
                    else:
                        xscaled = xpred

                    return xscaled

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

        :param x:
        :param y:
        :param block_to_score:
        :param sample_weight:
        :return:
        """
        # TO DO: actually use sample_weight
        try:
            if block_to_score not in ['x', 'y']:
                raise ValueError("message here")
            if y.ndim == 1:
                y = y[np.newaxis]
            if x.ndim == 1:
                x = x[np.newaxis]
            if self.x_scaler is not None:
                xscaled = self.x_scaler.fit_transform(x)
            else:
                xscaled = ChemometricsScaler(0, with_std=False).fit_transform(x)

            if self.y_scaler is not None:
                yscaled = self.y_scaler.fit_transform(y)
            else:
                yscaled = ChemometricsScaler(0, with_std=False).fit_transform(y)

            # Calculate total sum of squares of X and Y for R2X and R2Y calculation
            tssy = np.sum(yscaled ** 2)
            tssx = np.sum(xscaled ** 2)

            # Calculate RSSy/RSSx, R2Y/R2X
            # The prediction here of both X and Y is done using the other block of data only
            # so these R2s can be interpreted as as a "classic" R2, and not as a proportion of variance modelled
            # Here we use X = Ub_uW', as opposed to (X = TP').
            ypred = self.y_scaler.transform(self.predict(xscaled, y=None))
            xpred = self.x_scaler.transform(self.predict(x=None, y=yscaled))
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

        :param x:
        :param y:
        :return:
        """

        try:
            if self._fitted is True:
                if x is not None and y is not None:
                    raise ValueError('xx')
                # If nothing is passed at all, complain and do nothing
                elif x is None and y is None:
                    raise ValueError('yy')
                # Predict Y from X
                elif x is not None:
                    if x.ndim == 1:
                        x = x[np.newaxis]
                    if self.x_scaler is not None:
                        xscaled = self.x_scaler.fit_transform(x)
                    else:
                        xscaled = ChemometricsScaler(0, with_std=False).fit_transform(x)
                    # Using Betas to predict Y directly
                    predicted = np.dot(xscaled, self.coefs_)
                    predicted = self.y_scaler.inverse_transform(predicted)
                    return predicted
                # Predict X from Y
                elif y is not None:
                    if y.ndim == 1:
                        y = y[np.newaxis]
                    # Going through calculation of U and then X = Ub_uW'
                    u_scores = self.transform(x=None, y=y)
                    predicted = np.dot(np.dot(u_scores, self.b_u), self.weights_w.T)
                    predicted = self.x_scaler.inverse_transform(predicted)
                    return predicted

        except Exception as exp:
            raise exp

    @property
    def ncomps(self):
        """
        Getter for number of components
        :param ncomps:
        :return:
        """
        try:
            return self._ncomps
        except AttributeError as atre:
            raise atre

    @ncomps.setter
    def ncomps(self, ncomps=1):
        """
        Setter for number of components
        :param ncomps:
        :return:
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

            return None
        except AttributeError as atre:
            raise atre

    @property
    def x_scaler(self):
        """
        Getter for the model scaler
        :return:
        """
        try:
            return self._x_scaler
        except AttributeError as atre:
            raise atre

    @x_scaler.setter
    def x_scaler(self, scaler):
        """
        Setter for the model scaler
        :param scaler:
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

            return None
        except AttributeError as atre:
            raise atre
        except TypeError as typerr:
            raise typerr

    @property
    def y_scaler(self):
        """
        Getter for the model scaler
        :return:
        """
        try:
            return self._y_scaler
        except AttributeError as atre:
            raise atre

    @y_scaler.setter
    def y_scaler(self, scaler):
        """
        Setter for the model scaler
        :param scaler:
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

            return None

        except AttributeError as atre:
            raise atre
        except TypeError as typerr:
            raise typerr

    def VIP(self, mode='w', direction='y'):
        """

        :param mode:
        :param direction:
        :return:
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
                tvarpred = 1
            else:
                tvarpred = 1
            n = 1

            vip_numerator = 0

            for currcomp in range(0, self.ncomps):
                if direction == 'y':
                    var_pred = 1
                else:
                    var_pred = 1
                vip_numerator += (choices[mode]**2) * var_pred
            vip = np.sqrt(vip_numerator/(tvarpred)*n)

            return vip

        except AttributeError as atre:
            raise AttributeError("Model not fitted")
        except ValueError as verr:
            raise verr

    def hotelling_T2(self, comps):
        """

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

    def cross_validation(self, x, y,  cv_method=KFold(7, True), outputdist=False, testset_scale=False,
                         **crossval_kwargs):
        """

        :param x:
        :param y:
        :param cv_method:
        :param outputdist:
        :param testset_scale:
        :param crossval_kwargs:
        :return:
        """

        try:
            if not isinstance(cv_method, BaseCrossValidator):
                raise TypeError("Scikit-learn cross-validation object please")

            # Check if global model is fitted... and if not, fit it using all of X
            if self._isfitted is False or self.loadings is None:
                self.fit(x, y)

            # Make a copy of the object, to ensure the internal state doesn't come out differently from the
            # cross validation method call...
            cv_pipeline = copy.deepcopy(self)
            ncvrounds = cv_method.get_n_splits()

            # Initialize list structures to contain the fit
            cv_loadings_p = np.zeros((ncvrounds, self.ncomps, x.shape[1]))
            cv_loadings_q = np.zeros((ncvrounds, self.ncomps, y.shape[1]))
            cv_weights_w = np.zeros((ncvrounds, self.ncomps, x.shape[1]))
            cv_weights_c = np.zeros((ncvrounds, self.ncomps, y.shape[1]))
            cv_scores_t = np.zeros((ncvrounds, self.ncomps, x.shape[0]))
            cv_scores_u = np.zeros((ncvrounds, self.ncomps, y.shape[0]))
            cv_rotations_ws = np.zeros((ncvrounds, self.ncomps, x.shape[1]))
            cv_rotations_cs = np.zeros((ncvrounds, self.ncomps, y.shape[1]))
            cv_betacoefs = np.zeros((ncvrounds, self.ncomps, x.shape[1]))
            cv_vipsw = np.zeros((ncvrounds, self.ncomps, x.shape[1]))

            # Initialise predictive residual sum of squares variable (for whole CV routine)
            pressy = 0
            pressx = 0

            # Calculate Sum of Squares SS in whole dataset for future calculations
            ssx = np.sum((cv_pipeline.x_scaler.fit_transform(x)) ** 2)
            ssy = np.sum((cv_pipeline.y_scaler.fit_transform(y)) ** 2)

            # As assessed in the test set..., opossed to PRESS
            R2X_training = np.zeros((ncvrounds))
            R2Y_training = np.zeros((ncvrounds))
            # R2X and R2Y assessed in the test set
            R2X_test = np.zeros((ncvrounds))
            R2Y_test = np.zeros((ncvrounds))

            for cvround, train_testidx in enumerate(cv_method.split(x, y)):
                train = train_testidx[0]
                test = train_testidx[1]
                # Fit the training data
                cv_pipeline.fit(x[train, :], y[train, :], **crossval_kwargs)
                # Prepare the scaled X and Y test data
                # If testset_scale is True, these are scaled individually...
                if testset_scale is True:
                    xtest_scaled = cv_pipeline.x_scaler.fit_transform(x[test, :])
                    ytest_scaled = cv_pipeline.y_scaler.fit_transform(y[test, :])
                # Otherwise (default), training set mean and scaling vectors are used
                else:
                    xtest_scaled = cv_pipeline.x_scaler.transform(x[test, :])
                    ytest_scaled = cv_pipeline.y_scaler.transform(y[test, :])

                R2X_training[cvround] = cv_pipeline.score(y[train, :], 'x')
                R2Y_training[cvround] = cv_pipeline.score(x[train, :], 'y')
                ypred = cv_pipeline.predict(y[test, :], 'x')
                xpred = cv_pipeline.predict(y[test, :], 'x')

                currtest_ssx = np.sum(xtest_scaled**2)
                currtest_ssy = np.sum(ytest_scaled**2)
                curr_pressx = np.sum((xtest_scaled - xpred)**2)
                curr_pressy = np.sum((xtest_scaled - xpred)**2)

                R2X_test[cvround] = 1 - (curr_pressx/currtest_ssx)
                R2Y_test[cvround] = 1 - (curr_pressy/currtest_ssy)

                pressx += curr_pressx
                pressy += curr_pressy

            #    Pipeline.predict(xtopred)

            # Align model parameters to account for sign indeterminacy.
            # The criteria here used is to select the sign that gives a more similar profile (by L1 distance) to the loadings fitted
            # on the model fitted with the whole data. Any other parameter can be used, but since the loadings in X capture
            # the covariance structure in X data block, in theory they should have more pronounced features even in cases of
            # null X-Y association, making the sign flip more resilient.
            for cvround in range(0, ncvrounds):
                for currload in range(0, self.ncomps):
                    # evaluate based on loadings _p
                    choice = np.argmin(np.array([np.sum(np.abs(self.loadings_p - cv_loadings_p[currload][cvround, :])),
                                                 np.sum(np.abs(self.loadings_p - cv_loadings_p[currload][cvround, :] * -1))]))
                    if choice == 1:
                        cv_loadings_p[currload][cvround, :] = -1 * cv_loadings_p[currload][cvround, :]
                        cv_loadings_q[currload][cvround, :] = -1 * cv_loadings_p[currload][cvround, :]
                        cv_weights_w[currload][cvround, :] = -1 * cv_weights_w[currload][cvround, :]
                        cv_weights_c[currload][cvround, :] = -1 * cv_weights_c[currload][cvround, :]
                        cv_rotations_ws[currload][cvround, :] = -1 * cv_rotations_ws[currload][cvround, :]
                        cv_rotations_cs[currload][cvround, :] = -1 * cv_rotations_cs[currload][cvround, :]
                        cv_scores_t[currload][cvround, :] = -1 * cv_scores_t[currload][cvround, :]
                        cv_scores_u[currload][cvround, :] = -1 * cv_scores_u[currload][cvround, :]

            # Calculate total sum of squares
            q_squaredy = 1 - (pressy/ssy)
            q_squaredx = 1 - (pressx/ssx)
            # Assemble the stuff in the end

            self.cvParameters = {'Q2X': q_squaredx, 'Q2Y': q_squaredy}

            return None

        except TypeError as terp:
            raise terp

    def permute_test(self, nperms=1000, crossVal=KFold(7, True)):
        """

        :param nperms:
        :param crossVal:
        :return:
        """
        #permuted
        for perm in range(0, nperms):
            p = 1
        return None

    def score_plot(self, lvs=[1,2], scores="T"):
        """

        :param lvs:
        :param scores:
        :return:
        """

        return None

    def coeffs_plot(self, lv=1, coeffs='weights'):
        """

        :param lv:
        :param coeffs:
        :return:
        """
        return None

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result


