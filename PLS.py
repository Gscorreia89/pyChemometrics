import numpy as np
import pandas as pds
from sklearn.cross_decomposition import PLSRegression
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.cross_decomposition.pls_ import PLSRegression
from ChemometricsScaler import ChemometricsScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import BaseCrossValidator, KFold
from ChemometricsScaler import ChemometricsScaler


__author__ = 'gd2212'


class PLS(BaseEstimator, RegressorMixin, TransformerMixin):

    def __init__(self, ncomps=2, pls_algorithm=PLSRegression, scaling=ChemometricsScaler(), metadata=None,**pls_type_kwargs):
        """

        :param ncomps:
        :param pls_algorithm:
        :param scaling:
        :param metadata:
        :param pls_type_kwargs:
        """
        try:
            # Metadata assumed to be pandas dataframe only
            if metadata is not None:
                if not isinstance(metadata, pds.DataFrame):
                    raise TypeError("Metadata must be provided as pandas dataframe")
            # The actual classifier can change, but needs to be a scikit-learn BaseEstimator
            # Changes this later for "PCA like only" - either check their hierarchy or make a list
            if not issubclass(pls_algorithm, BaseEstimator):
                raise TypeError("Scikit-learn model please")
            if not issubclass(scaling, TransformerMixin):
                raise TypeError("Scikit-learn Transformer-like object please")

            # Add a check for partial fit methods? As in deploy partial fit child class if PCA is incremental??
            #types.MethodType(self)
            # The kwargs provided for the model are exactly the same as those
            # go and check for these examples the correct exception to throw when kwarg is not valid
            # TO DO: Set the sklearn params for PCA to be a junction of the custom ones and the "core" params of model
            self._model = pls_algorithm(ncomps, **pls_type_kwargs)
            # These will be non-existant until object is fitted.
            # self.scores = None
            # self.loadings = None
            self.ncomps = ncomps
            self.scaler = scaling
            self.cvParameters = None
            self.modelParameters = None

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
            # Scaling
            xscaled = self.fit_transform(x)
            yscaled = self.fit_transform(y)
            self._model.fit(x=xscaled, y=yscaled, **fit_params)
            self.scores = self.transform(x)
            self.loadings = self._model.x_loadings_
            self.yloads = self._model.y_loadings_
            self.weights = self._model.x_weights_
            self.modelParameters = {'VarianceExplained'}

        except Exception as exp:
            raise exp

    def fit_transform(self, x, **fit_params):
        """
        Combination of fit and output the scores, nothing special
        :param x: Data to fit
        :param fit_params:
        :return:
        """

        return self._model.fit_transform(x, **fit_params)

    def transform(self, x):
        """
        Calculate the projection of the data into the lower dimensional space
        :param x:
        :return:
        """

        return self._model.transform(x)

    def inverse_transform(self, scores):
        """

        :param scores:
        :return:
        """
        return self._model.inverse_transform(scores)

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

        return None

    @property
    def ncomps(self):
        """
        important to make sure changing n comps here ACTUALLY changes the fit.
        :param ncomps:
        :return:
        """
        try:
            return self.ncomps
        except AttributeError as atre:
            raise atre

    @ncomps.setter
    def ncomps(self, ncomps=1):
        try:
            return None
        except AttributeError as atre:
            raise atre
    @property
    def scores(self, comp=1):
        """

        :param comp:
        :return:
        """
        try:
            return self._scores[:, comp-1]
        except AttributeError as atre:
            raise atre

    @property
    def scaler(self):
        try:
            return self.scaler
        except AttributeError as atre:
            raise atre

    @scaler.setter
    def scaler(self, scaler):
        try:
            if not issubclass(scaler, TransformerMixin):
                raise TypeError("Scikit-learn Transformer-like object please")
            self.scaler = scaler
            return None
        except AttributeError as atre:
            raise atre
        except TypeError as typerr:
            raise typerr

    def cross_validation(self, x, y,  method=KFold(7, True), outputdist=False, bro_press=True,**crossval_kwargs):
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
            if not isinstance(method, BaseCrossValidator):
                raise TypeError("Scikit-learn cross-validation object please")

            Pipeline = ([('scaler', self.scaler), ('pca', self._model)])

            # Check if global model is fitted... and if not, fit using x
            # Initialise predictive residual sum of squares variable
            press = 0
            # Calculate Sum of Squares SS
            ss_x = 0
            ss_y = 0
            P = []
            T = []
            W = []
            U = []
            # As assessed in the test set..., opossed to PRESS
            R2X = []
            R2Y = []

            for xtrain, xtest in KFold.split(x):
                Pipeline.fit_transform(xtest)
                if bro_press:
                    for var in range(0, xtest.shape[1]):
                        xpred = Pipeline.predict(xtest, var)
                        press += 1
                else:
                    xpred = Pipeline.fit_transform(xtest)
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
            q_squared = 1 - (press/ss)
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


