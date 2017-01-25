import pandas as pds
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.decomposition import PCA as skPCA
from sklearn.decomposition.base import _BasePCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import BaseCrossValidator, KFold
import numpy as np
from sklearn.base import clone
from ChemometricsScaler import ChemometricsScaler
__author__ = 'gd2212'


class ChemometricsPCA(_BasePCA):
    """
    General PCA class
    Inherits from Base Estimator, and TransformerMixin, to act as a legit fake
    # Scikit classifier
    """

    # This class inherits from Base Estimator, and TransformerMixin to act as believable
    # scikit-learn PCA classifier
    # Constant usage of kwargs ensures that everything from scikit-learn can be used directly
    def __init__(self, ncomps=2, pca_algorithm=skPCA, scaler=ChemometricsScaler(), metadata=None, **pca_type_kwargs):

        """
        :param metadata:
        :param n_comps:
        :param pca_algorithm:
        :param pca_type_kwargs:
        :return:
        """
        try:
            # Metadata assumed to be pandas dataframe only
            if (metadata is not None) and (metadata is not isinstance(metadata, pds.DataFrame)):
                    raise TypeError("Metadata must be provided as pandas dataframe")
            # The actual classifier can change, but needs to be a scikit-learn BaseEstimator
            # Changes this later for "PCA like only" - either check their hierarchy or make a list
            #print(type(pca_algorithm))
            # Perform the check with is instance but avoid abstract base class runs. PCA needs number of comps anyway!
            pca_algorithm = pca_algorithm(n_components=ncomps)
            if not isinstance(pca_algorithm, (_BasePCA, BaseEstimator, TransformerMixin)):
                raise TypeError("Scikit-learn model please")
            if not isinstance(scaler, TransformerMixin) or scaler is None:
                raise TypeError("Scikit-learn Transformer-like object or None")

            # Add a check for partial fit methods? As in deploy partial fit child class if PCA is incremental??
            # By default it will work, but having the partial_fit function acessible might be usefull
            #types.MethodType(self, partial_fit)
            #def partial_fit():
            #    returnx

            # The kwargs provided for the model are exactly the same as those
            # go and check for these examples the correct exception to throw when kwarg is not valid
            # TO DO: Set the sklearn params for PCA to be a junction of the custom ones and the "core" params of model
            self.pca_algorithm = pca_algorithm

            # Most initialized as non, before object is fitted.
            self.scores = None
            self.loadings = None
            self._ncomps = None
            self._scaler = None
            self.ncomps = ncomps
            self.scaler = scaler
            self.cvParameters = None
            self.modelParameters = None
            self._isfitted = False

        except TypeError as terp:
            print(terp.args[0])
            raise terp

        except ValueError as verr:
            print(verr.args[0])
            raise verr

    def fit(self, x, **fit_params):
        """
        Fit function. Acts exactly as in scikit-learn, but
        :param x:
        :param scale:
        :return:

        """
        try:
            # Scaling
            if self.scaler is not None:
                xscaled = self.scaler.fit_transform(x)
                self.pca_algorithm.fit(xscaled, **fit_params)
                self.scores = self.transform(xscaled)
            else:
                self.pca_algorithm.fit(x, **fit_params)
                self.scores = self.transform(x)
            # Kernel PCA and other non-linear methods might not have explicit loadings - safeguard against this
            if hasattr(self.pca_algorithm, 'components_'):
                self.loadings = self.pca_algorithm.components_
            self.modelParameters = {'VarianceExplained': self.pca_algorithm.explained_variance_, 'ProportionVarExp': self.pca_algorithm.explained_variance_ratio_}
            self._isfitted = True
        except Exception as exp:
            raise exp

    def fit_transform(self, x, **fit_params):
        """
        Combination of fit and output the scores, nothing special
        :param x: Data to fit
        :param fit_params:
        :return:
        """
        try:
            self.fit(x)
            return self.transform(x)
        except Exception as exp:
            raise exp

    def transform(self, x, **transform_kwargs):
        """
        Calculate the projection of the data into the lower dimensional space
        :param x:
        :return:
        """
        if self.scaler is not None:
            xscaled = self.scaler.transform(x)
            return self.pca_algorithm.transform(xscaled, **transform_kwargs)
        else:
            return self.pca_algorithm.transform(x, **transform_kwargs)

    def score(self, x, **score_kwargs):
        """
        Keeping the original likelihood method of probabilistic PCA.
        The R2 is already obtained by the SVD so...
        :param x:
        :param sample_weight:
        :return:
        """
        try:
            if self.scaler is not None:
                xscaled = self.scaler.transform(x)
                return self.pca_algorithm.score(xscaled, **score_kwargs)
            else:
                return self.pca_algorithm.score(x, **score_kwargs)
        except Exception as exp:
            return None

    def inverse_transform(self, scores):
        """
        Reconstruct the full data matrix from vector of scores
        :param scores:
        :return:
        """
        if self.scaler is not None:
            xinv_prescaled = self.pca_algorithm.inverse_transform(scores)
            xinv = self.scaler.inverse_transform(xinv_prescaled)
            return xinv
        else:
            return self.pca_algorithm.inverse_transform(scores)

    def _press_impute(self, x, var_to_pred):
        """
        A bit weird for pca...
        The idea is to place missing data imputation here!
        :param x:
        :param vars_to_pred: which variables are to be predicted
        :return:
        """
        if self.scaler is not None:
            xscaled = self.scaler.transform(x)
        else:
            xscaled = x

        to_pred = np.delete(xscaled, var_to_pred, axis=1)
        topred_loads = np.delete(self.loadings.T, var_to_pred, axis=0)
        imputed_x = np.dot(np.dot(to_pred, np.linalg.pinv(topred_loads).T), self.loadings)
        if self.scaler is not None:
            imputed_x = self.scaler.inverse_transform(imputed_x)

        return imputed_x

    @property
    def ncomps(self):
        """
        Although this getter is not so important
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
        The ncomps property can be used to
        :param ncomps:
        :return:
        """
        try:
            self._ncomps = ncomps
            self.pca_algorithm = clone(self.pca_algorithm, safe=True)
            self.pca_algorithm.n_components = ncomps
            self.modelParameters = None
            self.loadings = None
            self.scores = None
            self.cvParameters = None
            return None
        except AttributeError as atre:
            raise atre

    @property
    def scaler(self):
        try:
            return self._scaler
        except AttributeError as atre:
            raise atre

    @scaler.setter
    def scaler(self, scaler):
        try:
            if not isinstance(scaler, TransformerMixin) or scaler is None:
                raise TypeError("Scikit-learn Transformer-like object or None")
            self._scaler = scaler
            self.pca_algorithm = clone(self.pca_algorithm, safe=True)
            self.modelParameters = None
            self.loadings = None
            self.scores = None
            self.cvParameters = None
            return None

        except AttributeError as atre:
            raise atre
        except TypeError as typerr:
            raise typerr

    def cross_validation(self, x,  method=KFold(7, True), outputdist=False, bro_press=True, **crossval_kwargs):
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

            # Check if global model is fitted... and if not, fit it using all of X
            if self._isfitted is False:
                self.fit(x)
            cv_pipeline = self
            # Initialise predictive residual sum of squares variable
            press = 0
            # Calculate Sum of Squares SS
            ss = np.sum((x - np.mean(x, 0))**2)
            # Initialise list for loadings and for the VarianceExplained in the test set values

            if hasattr(self.pca_algorithm, 'components_'):
                loadings = []
            cv_varexplained = []

            for xtrain, xtest in method.split(x):
                cv_pipeline.fit(x[xtrain, :])
                # Calculate R2/Variance Explained in test set

                testset_scores = cv_pipeline.transform(x[xtest, :])
                rss = np.sum((x[xtest, :] - cv_pipeline.inverse_transform(testset_scores))**2)
                tss = np.sum((x[xtest, :] - np.mean(x[xtest, :], 0))**2)

                # Append the var explained in test set for this round and loadings
                cv_varexplained.append(cv_pipeline.pca_algorithm.explained_variance_)
                if hasattr(self.pca_algorithm, 'components_'):
                    loadings.append(cv_pipeline.loadings)

                if bro_press is True:
                    for column in range(0, x[xtest, :].shape[1]):
                        xpred = cv_pipeline._press_impute(x[xtest, :], column)
                        press += np.sum((x[xtest, column] - xpred[:, column])**2)
                else:
                    pred_scores = cv_pipeline.fit_transform(x[xtest, :])
                    pred_x = cv_pipeline.inverse_transform(pred_scores)
                    press += np.sum((x[xtest, :] - pred_x)**2)

            # Create matrices for each component loading containing the cv values in each round
            # nrows = nrounds, ncolumns = n_variables

            # Check that the PCA model has loadings
            if hasattr(self.pca_algorithm, 'components_'):
                cv_loads = []
                for comp in range(0, self.ncomps):
                    cv_loads.append(np.array([x[comp] for x in loadings]))

                # Align loadings due to sign indeterminacy.
                for cvround in range(0, method.n_splits):
                    for currload in range(0, self.ncomps):
                        choice = np.argmin(np.array([np.sum(np.abs(self.loadings - cv_loads[currload][cvround, :])), np.sum(np.abs(self.loadings - cv_loads[currload][cvround,: ] * -1))]))
                        if choice == 1:
                            cv_loads[currload][cvround, :] = -1*cv_loads[currload][cvround, :]

            # Calculate total sum of squares
            # Q^2X
            q_squared = 1 - (press/ss)
            # Assemble the stuff in the end

            self.cvParameters = {'Mean_VarianceExplained': 1, 'Stdev_VarianceExplained': 1,
                                'Q2': q_squared}

            if outputdist:
                self.cvParameters['CV_VarianceExplained'] = cv_varexplained
            # Check that the PCA model has loadings
            if hasattr(self.pca_algorithm, 'components_'):
                self.cvParameters['Mean_Loadings'] = 1
                self.cvParameters['Stdev_Loadings'] = 1
                if outputdist:
                    self.cvParameters['CV_Loadings'] = cv_loads

            return None

        except TypeError as terp:
            raise terp

    def permute_test(self, nperms = 1000, crossVal=KFold(7, True)):
        #permuted
        #for perm in range(0, nperms):
        return NotImplementedError

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

    def loadings_plot(self, lv=1, coeffs='weightscv'):
        """

        :param lv:
        :param coeffs:
        :return:
        """
        return None
