from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA as skPCA
from sklearn.decomposition.base import _BasePCA
from sklearn.model_selection import BaseCrossValidator, KFold
from sklearn.model_selection._split import BaseShuffleSplit
import numpy as np
from sklearn.base import clone
from .ChemometricsScaler import ChemometricsScaler
import copy
from copy import deepcopy
__author__ = 'gd2212'


class ChemometricsPCA(_BasePCA):
    """

    ChemometricsPCA object - Wrapper for sklearn.decomposition PCA algorithms, with tailored methods
    for Chemometric Data analysis.

    :param ncomps: Number of PCA components desired.
    :type ncomps: int
    :param sklearn.decomposition._BasePCA pca_algorithm: scikit-learn PCA models (inheriting from _BasePCA).
    :param scaler: The object which will handle data scaling.
    :type scaler: ChemometricsScaler object, scaling/preprocessing objects from scikit-learn or None
    :param kwargs pca_type_kwargs: Keyword arguments to be passed during initialization of pca_algorithm.
    :raise TypeError: If the pca_algorithm or scaler objects are not of the right class.
    """

    # Constant usage of kwargs might look excessive but ensures that most things from scikit-learn can be used directly
    # no matter what PCA algorithm is used
    def __init__(self, ncomps=2, pca_algorithm=skPCA, scaler=ChemometricsScaler(), **pca_type_kwargs):

        try:
            # Perform the check with is instance but avoid abstract base class runs. PCA needs number of comps anyway!
            pca_algorithm = pca_algorithm(n_components=ncomps, **pca_type_kwargs)
            if not isinstance(pca_algorithm, (_BasePCA, BaseEstimator, TransformerMixin)):
                raise TypeError("Scikit-learn model please")
            if not (isinstance(scaler, TransformerMixin) or scaler is None):
                raise TypeError("Scikit-learn Transformer-like object or None")
            if scaler is None:
                scaler = ChemometricsScaler(0, with_std=False)
            # Add a check for partial fit methods? As in deploy partial fit child class if PCA is incremental??
            # By default it will work, but having the partial_fit function acessible might be usefull
            #types.MethodType(self, partial_fit)
            #def partial_fit():
            #    returnx

            # The kwargs provided for the model are exactly the same as those
            # go and check for these examples the correct exception to throw when kwarg is not valid
            # TO DO: Set the sklearn params for PCA to be a junction of the custom ones and the "core" params of model
            self.pca_algorithm = pca_algorithm

            # Most initialized as None, before object is fitted.
            self.scores = None
            self.loadings = None
            self._ncomps = ncomps
            self._scaler = scaler
            self.cvParameters = None
            self.modelParameters = None
            self._isfitted = False

        except TypeError as terp:
            print(terp.args[0])
            raise terp

    def fit(self, x, **fit_params):
        """

        Perform model fitting on the provided x data matrix and calculate basic goodness-of-fit metrics.
        Equivalent to scikit-learn's default BaseEstimator method.

        :param x: Data matrix to fit the PCA model.
        :type x: numpy.ndarray, shape [n_samples, n_features].
        :param kwargs fit_params: Keyword arguments to be passed to the .fit() method of the core sklearn model.
        :raise ValueError: If any problem occurs during fitting.
        """

        try:
            # This scaling check is always performed to ensure running model with scaling or with scaling == None
            # always give consistent results (same type of data scale expected for fitting,
            # returned by inverse_transform, etc
            if self.scaler is not None:
                xscaled = self.scaler.fit_transform(x)
                self.pca_algorithm.fit(xscaled, **fit_params)
                self.scores = self.pca_algorithm.transform(xscaled)
                ss = np.sum((xscaled - np.mean(xscaled, 0)) ** 2)
                predicted = self.pca_algorithm.inverse_transform(self.scores)
                rss = np.sum((xscaled - predicted) ** 2)
                # variance explained from scikit-learn stored as well
                self.modelParameters = {'R2X': 1 - (rss/ss), 'VarExpRatio': self.pca_algorithm.explained_variance_ratio_,
                                        'VarExp': self.pca_algorithm.explained_variance_}
            else:
                self.pca_algorithm.fit(x, **fit_params)
                self.scores = self.pca_algorithm.transform(x)
                ss = np.sum((x - np.mean(x, 0)) ** 2)
                predicted = self.pca_algorithm.inverse_transform(self.scores)
                rss = np.sum((x - predicted) ** 2)
                self.modelParameters = {'R2X': 1 - (rss/ss), 'VarExp': self.pca_algorithm.explained_variance_,
                                        'VarExpRatio': self.pca_algorithm.explained_variance_ratio_}
            # Kernel PCA and other non-linear methods might not have explicit loadings - safeguard against this
            if hasattr(self.pca_algorithm, 'components_'):
                self.loadings = self.pca_algorithm.components_
            self._isfitted = True

        except ValueError as verr:
            raise verr

    def _partial_fit(self, x):
        """
        under construction...
        :param x:
        :return:
        """
        return NotImplementedError

    def fit_transform(self, x, **fit_params):
        """

        Fit a model and return the scores, as per the scikit-learn's TransformerMixin method.

        :param x: Data matrix to fit and project.
        :type x: numpy.ndarray, shape [n_samples, n_features]
        :param kwargs fit_params: Optional keyword arguments to be passed to the fit method.
        :return: PCA projections (scores) corresponding to the samples in X.
        :rtype: numpy.ndarray, shape [n_samples, n_comps]
        :raise ValueError: If there are problems with the input or during model fitting.
        """

        try:
            self.fit(x, **fit_params)
            return self.transform(x)
        except ValueError as exp:
            raise exp

    def transform(self, x):
        """

        Calculate the projections (scores) of the x data matrix. Similar to scikit-learn's TransformerMixin method.

        :param x: Data matrix to fit and project.
        :type x: numpy.ndarray, shape [n_samples, n_features]
        :param kwargs transform_params: Optional keyword arguments to be passed to the transform method.
        :return: PCA projections (scores) corresponding to the samples in X.
        :rtype: numpy.ndarray, shape [n_samples, n_comps]
        :raise ValueError: If there are problems with the input or during model fitting.
        """
        try:
            if self.scaler is not None:
                xscaled = self.scaler.transform(x)
                return self.pca_algorithm.transform(xscaled)
            else:
                return self.pca_algorithm.transform(x)
        except ValueError as verr:
            raise verr

    def score(self, x, sample_weight=None):
        """

        Return the average log-likelihood of all samples. Same as the underlying score method from the scikit-learn
        PCA objects.

        :param x: Data matrix to score model on.
        :type x: numpy.ndarray, shape [n_samples, n_features]
        :param numpy.ndarray sample_weight: Optional sample weights during scoring.
        :return: Average log-likelihood over all samples.
        :rtype: float
        :raises ValueError: if the data matrix x provided is invalid.
        """
        try:
            # Not all sklearn pca objects have a "score" method...
            score_method = getattr(self.pca_algorithm, "score", None)
            if not callable(score_method):
                raise NotImplementedError
            # Scaling check for consistency
            if self.scaler is not None:
                xscaled = self.scaler.transform(x)
                return self.pca_algorithm.score(xscaled, sample_weight)
            else:
                return self.pca_algorithm.score(x, sample_weight)
        except ValueError as verr:
            raise verr

    def inverse_transform(self, scores):
        """

        Transform scores to the original data space using the principal component loadings.
        Similar to scikit-learn's default TransformerMixin method.

        :param scores: The projections (scores) to be converted back to the original data space.
        :type scores: numpy.ndarray, shape [n_samples, n_comps]
        :return: Data matrix in the original data space.
        :rtype: numpy.ndarray, shape [n_samples, n_features]
        :raises ValueError: If the dimensions of score mismatch the number of components in the model.
        """
        # Scaling check for consistency
        if self.scaler is not None:
            xinv_prescaled = self.pca_algorithm.inverse_transform(scores)
            xinv = self.scaler.inverse_transform(xinv_prescaled)
            return xinv
        else:
            return self.pca_algorithm.inverse_transform(scores)

    def _press_impute_pinv(self, x, var_to_pred):
        """

        Single value imputation method, essential to use in the cross-validation.
        In theory can also be used to do missing data imputation.
        Based on the Eigenvector_PRESS calculation as described in:
        1) Bro et al, Cross-validation of component models: A critical look at current methods,
        Analytical and Bioanalytical Chemistry 2008 - doi: 10.1007/s00216-007-1790-1
        2) amoeba's answer on CrossValidated: http://stats.stackexchange.com/a/115477

        :param x: Data matrix in the original data space.
        :type x: numpy.ndarray, shape [n_samples, n_comps]
        :param int var_to_pred: which variable is to be imputed from the others.
        :return: Imputed X matrix.
        :rtype: numpy.ndarray, shape [n_samples, n_features]
        :raise ValueError: If there is any error during the imputation process.
        """

        try:
            # Scaling check for consistency
            if self.scaler is not None:
                xscaled = self.scaler.transform(x)
            else:
                xscaled = x
            # Following from
            to_pred = np.delete(xscaled, var_to_pred, axis=1)
            topred_loads = np.delete(self.loadings.T, var_to_pred, axis=0)
            imputed_x = np.dot(np.dot(to_pred, np.linalg.pinv(topred_loads).T), self.loadings)
            if self.scaler is not None:
                imputed_x = self.scaler.inverse_transform(imputed_x)
            return imputed_x
        except ValueError as verr:
            raise verr

    def _press_impute_transpose(self, x, var_to_pred):
        """
        NOT READY needs finishing.

        Single value imputation method, essential to use in the cross-validation
        In theory can also be used to do missing data imputation.
        Based on the approximation described in amoeba's answer
        on CrossValidated: http://stats.stackexchange.com/a/115477

        :param x: Data matrix in the original data space.
        :type x: numpy.ndarray, shape [n_samples, n_comps]
        :param int var_to_pred: which variable is to be imputed from the others.
        :return: Imputed X matrix.
        :rtype: numpy.ndarray, shape [n_samples, n_features]
        :raise ValueError: If there is any error during the imputation process.
        """
        try:
            # Scaling check for consistency
            if self.scaler is not None:
                xscaled = self.scaler.transform(x)
            else:
                xscaled = x
            # Following from
            to_pred = np.delete(xscaled, var_to_pred, axis=1)
            topred_loads = np.delete(self.loadings.T, var_to_pred, axis=0)
            imputed_x = np.dot(np.dot(to_pred, np.linalg.pinv(topred_loads).T), self.loadings)
            if self.scaler is not None:
                imputed_x = self.scaler.inverse_transform(imputed_x)

            return imputed_x
        except ValueError as verr:
            raise verr

    @property
    def ncomps(self):
        try:
            return self._ncomps
        except AttributeError as atre:
            raise atre

    @ncomps.setter
    def ncomps(self, ncomps=1):
        """

        Setter for number of components.

        :param int ncomps: Number of components to use in the model.
        :raise AttributeError: If there is a problem changing the number of components and resetting the model.
        """
        # To ensure changing number of components effectively resets the model
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
        """

        Setter for the model scaler.

        :param scaler: The object which will handle data scaling.
        :type scaler: ChemometricsScaler object, scaling/preprocessing objects from scikit-learn or None
        :raise AttributeError: If there is a problem changing the scaler and resetting the model.
        :raise TypeError: If the new scaler provided is not a valid object.
        """
        try:
            if not (isinstance(scaler, TransformerMixin) or scaler is None):
                raise TypeError("Scikit-learn Transformer-like object or None")
            if scaler is None:
                scaler = ChemometricsScaler(0, with_std=False)

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

    def hotelling_T2(self, comps):
        """

        Obtain the parameters for the Hotelling T2 ellipse at the desired significance level.

        :param list comps:
        :return:
        :rtype:
        :raise ValueError: If the dimensions request
        """
        try:
            self.scores[:, comps]
            return None
        except AttributeError as atre:
            raise atre
        except ValueError as valerr:
            raise valerr
        except TypeError as typerr:
            raise typerr

    def dModX(self, x=None):
        """
        :return:
        """
        return NotImplementedError

    def leverages(self):
        """
        Calculate the leverages for each observation
        :return:
        :rtype:
        """
        return NotImplementedError

    def cross_validation(self, x,  cv_method=KFold(7, True), outputdist=False, press_impute=True, testset_scale=False):
        """

        Cross-validation method for the model. Calculates Q2 and cross-validated estimates for all model parameters.

        :param x: Data matrix.
        :type x: numpy.ndarray, shape [n_samples, n_feeatures]
        :param cv_method: An instance of a scikit-learn CrossValidator object.
        :type cv_method: BaseCrossValidator
        :param bool outputdist: Output the whole distribution for. Useful when ShuffleSplit or CrossValidators other than KFold.
        :param bool press_impute: Use imputation of test set observations instead of row wise cross-validation. Slower but more reliable.
        :param bool testset_scale: Scale the test using its own mean and standard deviation instead of the scaler fitted on training set.
        :return:
        :rtype: dict
        :raise TypeError: If the cv_method passed is not a scikit-learn CrossValidator object.
        :raise ValueError: If the x data matrix is invalid.
        """

        try:

            if not (isinstance(cv_method, BaseCrossValidator) or isinstance(cv_method, BaseShuffleSplit)):
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
            ss = np.sum((cv_pipeline.scaler.transform(x)) ** 2)
            # Initialise list for loadings and for the VarianceExplained in the test set values
            # Check if model has loadings, as in case of kernelPCA these are not available
            if hasattr(self.pca_algorithm, 'components_'):
                loadings = []

            # CV_varexplained_training is a list containg lists with the SingularValue/Variance explained as obtained
            # in the training set during fitting. cv_varexplained_test is a single R2X measure obtained from using the
            # model fitted with the training set in the test set.
            cv_varexplained_training = []
            cv_varexplained_test = []

            for xtrain, xtest in cv_method.split(x):
                cv_pipeline.fit(x[xtrain, :])
                # Calculate R2/Variance Explained in test set
                # To calculate an R2X in the test set
                if testset_scale:
                    xtest_scaled = cv_pipeline.scaler.fit_transform(x[xtest, :])
                else:
                    xtest_scaled = cv_pipeline.scaler.transform(x[xtest, :])
                tss = np.sum((xtest_scaled)**2)
                # Append the var explained in training set for this round and loadings for this round
                cv_varexplained_training.append(cv_pipeline.pca_algorithm.explained_variance_ratio_)
                if hasattr(self.pca_algorithm, 'components_'):
                    loadings.append(cv_pipeline.loadings)

                if press_impute is True:
                    press_testset = 0
                    for column in range(0, x[xtest, :].shape[1]):
                        xpred = cv_pipeline.scaler.transform(cv_pipeline._press_impute_pinv(x[xtest, :], column))
                        press_testset += np.sum((xtest_scaled[:, column] - xpred[:, column]) ** 2)
                    cv_varexplained_test.append(1 - (press_testset / tss))
                    total_press += press_testset
                else:
                    # RSS for row wise cross-validation
                    pred_scores = cv_pipeline.transform(x[xtest, :])
                    pred_x = cv_pipeline.scaler.transform(cv_pipeline.inverse_transform(pred_scores))
                    rss = np.sum((xtest_scaled - pred_x) ** 2)
                    total_press += rss
                    cv_varexplained_test.append(1 - (rss/tss))

            # Create matrices for each component loading containing the cv values in each round
            # nrows = nrounds, ncolumns = n_variables
            # Check that the PCA model has loadings
            if hasattr(self.pca_algorithm, 'components_'):
                cv_loads = []
                for comp in range(0, self.ncomps):
                    cv_loads.append(np.array([x[comp] for x in loadings]))

                # Align loadings due to sign indeterminacy.
                # Solution provided is to select the sign that gives a more similar profile to the
                # Loadings calculated with the whole data.
                for cvround in range(0, cv_method.n_splits):
                    for currload in range(0, self.ncomps):
                        choice = np.argmin(np.array([np.sum(np.abs(self.loadings - cv_loads[currload][cvround, :])), np.sum(np.abs(self.loadings - cv_loads[currload][cvround,: ] * -1))]))
                        if choice == 1:
                            cv_loads[currload][cvround, :] = -1*cv_loads[currload][cvround, :]

            # Calculate total sum of squares
            # Q^2X
            q_squared = 1 - (total_press/ss)
            # Assemble the dictionary and data matrices

            self.cvParameters = {'Mean_VarExpRatio_Training': np.array(cv_varexplained_training).mean(axis=0),
                                 'Stdev_VarExpRatio_Training': np.array(cv_varexplained_training).mean(axis=0),
                                'Mean_VarExp_Test': np.mean(cv_varexplained_test),
                                 'Stdev_VarExp_Test': np.std(cv_varexplained_test),
                                 'Q2': q_squared}

            if outputdist is True:
                self.cvParameters['CV_VarExpRatio_Training'] = cv_varexplained_training
                self.cvParameters['CV_VarExp_Test'] = cv_varexplained_test
            # Check that the PCA model has loadings
            if hasattr(self.pca_algorithm, 'components_'):
                self.cvParameters['Mean_Loadings'] = [np.mean(x, 0) for x in cv_loads]
                self.cvParameters['Stdev_Loadings'] = [np.std(x, 0) for x in cv_loads]
                if outputdist is True:
                    self.cvParameters['CV_Loadings'] = cv_loads

            return None

        except TypeError as terp:
            raise terp
        except ValueError as verr:
            raise verr

    def permutationtest_loadings(self, x, nperms=1000):
        """

        Permutation test to assess significance of magnitude of value for variable in component loading vector.
        Can be used to test importance of variable to the loading vector.

        :param x: Data matrix.
        :type x: numpy.ndarray, shape [n_samples, n_features]
        :param int nperms: Number of permutations.
        :return: Permuted null distribution for vector values.
        :rtype: numpy.ndarray, shape [ncomps, n_perms, n_features]
        :raise ValueError: If there is a problem with the input x data or during the procedure.
        """
        try:
            # Check if global model is fitted... and if not, fit it using all of X
            if self._isfitted is False or self.loadings is None:
                self.fit(x)
            # Make a copy of the object, to ensure the internal state doesn't come out differently from the
            # cross validation method call...
            permute_class = copy.deepcopy(self)
            # Initalise list for loading distribution
            permuted_loads = [np.zeros((nperms, x.shape[1]))] * permute_class.ncomps
            for permutation in range(0, nperms):
                for var in range(0, x.shape[1]):
                    # Copy original column order, shuffle array in place...
                    orig = np.copy(x[:, var])
                    np.random.shuffle(x[:, var])
                    # ... Fit model and replace original data
                    permute_class.fit(x)
                    x[:, var] = orig
                    # Store the loadings for each permutation component-wise
                    for loading in range(0, permute_class.ncomps):
                        permuted_loads[loading][permutation, var] = permute_class.loadings[loading][var]

            # Align loadings due to sign indeterminacy.
            # Solution provided is to select the sign that gives a more similar profile to the
            # Loadings calculated with the whole data.
            for perm_n in range(0, nperms):
                for currload in range(0, permute_class.ncomps):
                    choice = np.argmin(np.array([np.sum(np.abs(self.loadings - permuted_loads[currload][perm_n, :])),
                                                 np.sum(np.abs(self.loadings - permuted_loads[currload][perm_n, :] * -1))]))
                    if choice == 1:
                        permuted_loads[currload][perm_n, :] = -1 * permuted_loads[currload][perm_n, :]
            return permuted_loads
        except ValueError as verr:
            raise verr

    def permutationtest_components(self, x, nperms=1000):
        """
        Under Construction...
        Permutation test for a whole component. Also outputs permuted null distributions for the loadings.

        :param x: Data matrix.
        :type x: numpy.ndarray, shape [n_samples, n_features]
        :param int nperms: Number of permutations.
        :return: Permuted null distribution for the component metrics (VarianceExplained and R2).
        :rtype: numpy.ndarray, shape [ncomps, n_perms, n_features]
        :raise ValueError: If there is a problem with the input data.
        """
        try:
            # Check if global model is fitted... and if not, fit it using all of X
            if self._isfitted is False:
                self.fit(x)
            # Make a copy of the object, to ensure the internal state doesn't come out differently from the
            # cross validation method call...
            permute_class = copy.deepcopy(self)
            # Initalise list for loading distribution
            permuted_varExp = []
            for permutation in range(0, nperms):
                # Copy original column order, shuffle array in place...
                orig = np.copy(x)
                #np.random.shuffle(x.T)
                # ... Fit model and replace original data
                permute_class.fit(x)
                x = orig
                permuted_varExp.append(permute_class.ModelParameters['VarExpRatio'])
            return permuted_varExp

        except ValueError as verr:
            raise verr

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

