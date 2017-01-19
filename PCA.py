import pandas as pds
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.decomposition import PCA as skPCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, BaseCrossValidator, KFold
import numpy as np
from ChemometricsScaler import ChemometricsScaler
import types
__author__ = 'gd2212'


class PCA(BaseEstimator, TransformerMixin):
    """
    General PCA class
    Inherits from Base Estimator, and TransformerMixin, to act as a legit fake
    # Scikit classifier
    """

    # This class inherits from Base Estimator, and TransformerMixin to act as believable
    # scikit-learn PCA classifier
    # Constant usage of kwargs ensures that everything from scikit-learn can be used directly
    def __init__(self, ncomps=2, pca_algorithm=skPCA, scaling=ChemometricsScaler(), metadata=None, **pca_type_kwargs):
        """
        :param metadata:
        :param n_comps:
        :param pca_algorithm:
        :param pca_type_kwargs:
        :return:
        """
        try:
            # Metadata assumed to be pandas dataframe only
            if metadata is not None:
                if not isinstance(metadata, pds.DataFrame):
                    raise TypeError("Metadata must be provided as pandas dataframe")
            # The actual classifier can change, but needs to be a scikit-learn BaseEstimator
            # Changes this later for "PCA like only" - either check their hierarchy or make a list
            if not issubclass(pca_algorithm, BaseEstimator):
                raise TypeError("Scikit-learn model please")
            if not issubclass(scaling, TransformerMixin):
                raise TypeError("Scikit-learn Transformer-like object please")

            # Add a check for partial fit methods? As in deploy partial fit child class if PCA is incremental??
            types.MethodType(self)
            # The kwargs provided for the model are exactly the same as those
            # go and check for these examples the correct exception to throw when kwarg is not valid
            # TO DO: Set the sklearn params for PCA to be a junction of the custom ones and the "core" params of model
            self._model = pca_algorithm(ncomps, **pca_type_kwargs)
            # These will be non-existant until object is fitted.
            #self.scores = None
            #self.loadings = None
            self.ncomps = ncomps
            self.scaler = scaling
            self.cvParameters = None
            self.modelParameters = None

        except TypeError as terp:
            print(terp.args[0])

        except ValueError as verr:
            print(verr.args[0])

    def fit(self, x, **fit_params):
        """
        Fit function. Acts exactly as in scikit-learn, but
        :param x:
        :param scale:
        :return:

        """
        try:
            # Scaling
            xscaled = self.scaler(x)
            self._model.fit(xscaled, **fit_params)
            self.scores = self.transform(x)
            #self.modelParameters = {'VarianceExplained'}
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

    def score(self, x):
        """
        Keeping the original likelihood method of probabilistic PCA.
        The R2 is already obtained by the SVD so...
        :param x:
        :param sample_weight:
        :return:
        """
        self.model_.score(self, x, sample_weight=None)

        return None

    def inverse_transform(self, scores):
        """
        Reconstruct the full data matrix from vector of scores
        :param scores:
        :return:
        """
        return self._model.inverse_transform(scores)

    def predict(self, x, vars_to_pred):
        """
        A bit weird for pca...
        The idea is to place missing data imputation here!
        :param x:
        :param vars_to_pred: which variables are to be predicted
        :return:
        """
        # remove the vars from the original loadings
        for samp in range(0, x.shape[0]):
        #    xtopred = x
        #    Pipeline.predict(xtopred)
        #projection = np.dot(np.dot(to_pred, np.linalg.pinv(to_predloads).T), loadings)
        return self.inverse_transform(x)

    def impute(self, x):
        return False

    @property
    def loadings(self, comp=1):
        """

        :param comp:
        :return:
        """
        try:
            loading = self._model.components_[:, comp-1]
            return loading
        except AttributeError as atre:
            raise atre

    @property
    def ncomps(self, ncomps=1):
        """
        important to make sure changing n comps here ACTUALLY changes the fit.
        :param ncomps:
        :return:
        """
        try:
            pass
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

    def scale(self, scaling=1):
        """
        The usual scaling functions will be here
        # To destroy - replaced with a scaling object inside
         Use a setter just for this to be OO pedantic?
         or go with the nice and easy approach?
        :scaling power:

        :return:
        """
        # Reshape this, no need for self.scale_power
        if self.scaling != 0:
            # do this properly and add something for a log
            if callable(scaling):
                self.scalingvector = scaling
            else:
                x_std = self.X.std()
                self.scalingvector = self.x_std ** scaling
        else:
            self.x /= self.scalingvector

        return None

    def cross_validation(self, x,  method=KFold(7, True), outputdist=False, bro_press=True,**crossval_kwargs):
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
            press = 0
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
            # Calculate total sum of squares

            q_squared = 1 - (press/SS)
            # Assemble the stuff in the end
            self.cvParameters = {}

            return None

        except TypeError as terp:
            raise terp

    def permute_test(self):

        return None

    def score_plot(self, pcs=[1,2], hotelingt=0.95):
        if len(pcs) == 1:
            # do something decent for the 1D  score plot
            pass

        return None

    def trellisplot(self, pcs):
        """
        Trellis plot, maybe quickly wrapping up the stuff from seaborn
        :param pcs:
        :return:
        """
        return None

    def coeffs_plot(self, lv=1, coeffs='weightscv'):
        """

        :param lv:
        :param coeffs:
        :return:
        """
        return None