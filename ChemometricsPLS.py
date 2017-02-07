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


__author__ = 'gd2212'


class ChemometricsPLS(BaseEstimator, RegressorMixin, TransformerMixin):

    def __init__(self, ncomps=2, pls_algorithm=PLSRegression, xscaler=ChemometricsScaler(), yscaler=None, metadata=None,**pls_type_kwargs):
        """

        :param ncomps:
        :param pls_algorithm:
        :param scaling:
        :param metadata:
        :param pls_type_kwargs:
        """
        try:

            # Metadata assumed to be pandas dataframe only
            if (metadata is not None) and (metadata is not isinstance(metadata, pds.DataFrame)):
                raise TypeError("Metadata must be provided as pandas dataframe")

            # Perform the check with is instance but avoid abstract base class runs. PCA needs number of comps anyway!
            pls_algorithm = pls_algorithm(n_components=ncomps)
            if not isinstance(pls_algorithm, (BaseEstimator, _PLS)):
                raise TypeError("Scikit-learn model please")
            if not isinstance(xscaler, TransformerMixin) or xscaler is None:
                raise TypeError("Scikit-learn Transformer-like object or None")
            if not isinstance(yscaler, TransformerMixin) or yscaler is None:
                raise TypeError("Scikit-learn Transformer-like object or None")
            # Force scaling to false, as this will be handled by the provided scaler
            self.pls_algorithm = pls_algorithm(ncomps, scale=False, **pls_type_kwargs)

            # Most initialized as None, before object is fitted.
            self.scores_t = None
            self.scores_u = None
            self.weights_w = None
            self.weights_y = None
            self.loadings_p = None
            self.loadings_c = None
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
            # always give consistent results (same type of data scale expected for fitting,
            # returned by inverse_transform, etc
            if self.x_scaler is not None:
                xscaled = self.x_scaler.fit_transform(x)
            else:
                xscaled = x
            if self.y_scaler is not None:
                yscaled = self.y_scaler.fit_transform(y)
            else:
                yscaled = y

            self.pls_algorithm.fit(xscaled, yscaled, **fit_params)
            self.scores_t = self.transform(xscaled)
            self.scores_u = self.transform(None, yscaled)
            self.loadings_p = self.pls_algorithm.x_loadings_
            self.loadings_c = self.pls_algorithm.y_loadings_
            self.weights_w = self.pls_algorithm.x_weights_
            self.weights_y = self.pls_algorithm.y_weights_
            self.modelParameters = {'R2Y': self.pls_algorithm.explained_variance_, 'R2X': self.pls_algorithm.explained_variance_ratio_}
            self._isfitted = True

        except Exception as exp:
            raise exp

    def fit_transform(self, x, y ,**fit_params):
        """
        Obtain scores in X
        :param x: Data to fit
        :param fit_params:
        :return:
        """
        try:
            self.fit(x, y,**fit_params)
            if self.x_scaler is not None:
                xscaled = self.x_scaler.fit_transform(x)
            else:
                xscaled = x
            if self.y_scaler is not None:
                yscaled = self.y_scaler.fit_transform(y)
            else:
                yscaled = y
            return self.transform(xscaled, y=None), self.transform(x=None, y=yscaled)
        except Exception as exp:
            raise exp

    def transform(self, x=None, y=None, **transform_kwargs):
        """
        Calculate the projection of the data into the lower dimensional space
        TO DO as Pls does not contain this...
        :param x:
        :return:
        """
        try:
            if x is not None:
                if self.x_scaler is not None:
                    xscaled = self.x_scaler.fit_transform(x)
                else:
                    xscaled = x
            elif y is not None:
                if self.y_scaler is not None:
                    yscaled = self.y_scaler.fit_transform(y)
                else:
                    yscaled = y


            return self.pls_algorithm.transform(xscaled, **transform_kwargs)

        except Exception as exp:
            raise exp

    def inverse_transform(self, t=None, u=None):
        """

        :param scores:
        :return:
        """

        self._model.inverse_transform(t)

        if self.x_scaler is not None:
            xscaled = self.x_scaler.fit_transform(x)
        else:
            xscaled = x
        if self.y_scaler is not None:
            yscaled = self.y_scaler.fit_transform(y)
        else:
            yscaled = y


        return self._model.inverse_transform(t)


    def score(self, x, y, sample_weight=None):
        """

        :param x:
        :param sample_weight:
        :return:
        """
        # Check this
        r2x = self._model.score(x, y)
        r2y = self._model.score(y, x)

        return None

    def predict(self, x=None, y=None):
        try:

            if y is None:
                self.scores_t
                predicted = 1
            if x is None:
                prediction = np.dot(self.regression_coefficients)
                predicted = 1

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
            self.modelParameters = None
            self.loadings_p = None
            self.scores_t = None
            self.scores_u = None
            self.loadings_c = None
            self.x_weights = None
            self.cvParameters = None
            self.modelParameters = None

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
            if not isinstance(scaler, TransformerMixin) or scaler is None:
                raise TypeError("Scikit-learn Transformer-like object or None")
            self._x_scaler = scaler
            self.pls_algorithm = clone(self.pls_algorithm, safe=True)
            self.modelParameters = None
            self.cvParameters = None
            self.loadings_p = None
            self.weights = None
            self.loadings_c = None
            self.scores_t = None
            self.scores_u = None
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
            if not isinstance(scaler, TransformerMixin) or scaler is None:
                raise TypeError("Scikit-learn Transformer-like object or None")
            self._y_scaler = scaler
            self.pls_algorithm = clone(self.pls_algorithm, safe=True)
            self.modelParameters = None
            self.cvParameters = None
            self.loadings_p = None
            self.weights = None
            self.loadings_c = None
            self.scores_t = None
            self.scores_u = None
            return None
        except AttributeError as atre:
            raise atre
        except TypeError as typerr:
            raise typerr

    @property
    def VIP(self):
        try:
            np.sum(self.x_weights_**2)
            return None
        except AttributeError as atre:
            raise AttributeError("Model not fitted")

    @property
    def regression_coefficients(self):
        """

        :return:
        """
        try:
            if self._isfitted is not None:
                return self.pls_algorithm.coefs_
            else:
                return None

        except AttributeError as atre:
            raise AttributeError("Model not fitted")

    @property
    def r_SIMPLS(self):
        """

        :return:
        """
        try:
            if self._isfitted:
                return self.pls_algorithm.x_rotations_
            else:
                return None
        except AttributeError as atre:
            raise AttributeError("Model not fitted")

    @property
    def hotelling_T2(self, comps):
        try:
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

    def cross_validation(self, x, y,  cv_method=KFold(7, True), outputdist=False, bro_press=True,**crossval_kwargs):
        """
        # Check this one carefully ... good oportunity to build in nested cross-validation
        # and stratified k-folds
        :param data:
        :param method:
        :param outputdist: Output the whole distribution for (useful when Bootstrapping is used)
        :param crossval_kwargs:
        :return:
        """

        try:
            if not isinstance(cv_method, BaseCrossValidator):
                raise TypeError("Scikit-learn cross-validation object please")

            # Check if global model is fitted... and if not, fit it using all of X
            if self._isfitted is False or self.loadings is None:
                self.fit(x)
            # Make a copy of the object, to ensure the internal state doesn't come out differently from the
            # cross validation method call...
            cv_pipeline = copy.deepcopy(self)

            # Initialise predictive residual sum of squares variable (for whole CV routine)
            total_press = 0
            # Calculate Sum of Squares SS in whole dataset
            ssy = np.sum((y - np.mean(y, 0))**2)
            ssx = np.sum((x - np.mean(x, 0))**2)

            # Check if global model is fitted... and if not, fit using x
            # Initialise predictive residual sum of squares variable

            pressy = 0
            pressx = 0

            # Calculate Sum of Squares SS

            P = list()
            C = list()
            T = list()
            W = list()
            U = list()
            # As assessed in the test set..., opossed to PRESS
            R2X = list()
            R2Y = list()

            for xtrain, xtest, ytrain, ytest in cv_method.split(x, y):
                Pipeline.fit_transform()
                for var in range(0, xtest.shape[1]):
                xpred = Pipeline.predict(xtest, var)
                press += 1

                #    Pipeline.predict(xtopred)
            # Introduce loop here to align loadings due to sign indeterminacy.
            # Introduce loop here to align loadings due to sign indeterminacy.


            for cvround in range(0,KFold.n_splits(x)):
                for cv_comploadings in loads:
                    choice = np.argmin(np.array([np.sum(np.abs(self.loadings - cv_comploadings)), np.sum(np.abs(self.loadings[] - cv_comploadings * -1))]))
                    if choice == 1:
                        -1*choice

            # Calculate total sum of squares
            q_squaredy = 1 - (press/ssy)
            q_squaredx = 1 - (press/ssx)
            # Assemble the stuff in the end


            self.cvParameters = {}

            return None

        except TypeError as terp:
            raise terp

    def permute_test(self, nperms = 1000, crossVal=KFold(7, True)):
        #permuted
        for perm in range(0, nperms):

        return None

    def score_plot(self, lvs=[1,2], scores="T"):

        return None

    def coeffs_plot(self, lv=1, coeffs='weights'):
        return None


