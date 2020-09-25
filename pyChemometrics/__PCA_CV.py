from sklearn.model_selection import KFold, BaseCrossValidator
from sklearn.model_selection._split import BaseShuffleSplit


def PCA_CV(PCA_model, x, cv_method=KFold(7, shuffle=True), store_distribution=False, press_impute=False,
           which_params=list()):

    """

    Cross-validation method for the PCA. Calculates cross-validated estimates for Q2X and other
    model parameters using row-wise cross validation.

    :param x: Data matrix.
    :type x: numpy.ndarray, shape [n_samples, n_features]
    :param cv_method: An instance of a scikit-learn CrossValidator object.
    :type cv_method: BaseCrossValidator
    :param bool outputdist: Output the whole distribution for the cross validated parameters.
    Useful when using ShuffleSplit or CrossValidators other than KFold.
    :param bool press_impute: Use imputation of test set observations instead of row wise cross-validation.
    Slower but more reliable.
    :return: Adds a dictionary cvParameters to the object, containing the cross validation results
    :rtype: dict
    :raise TypeError: If the cv_method passed is not a scikit-learn CrossValidator object.
    :raise ValueError: If the x data matrix is invalid.
    """

    try:

        if not (isinstance(cv_method, BaseCrossValidator) or isinstance(cv_method, BaseShuffleSplit)):
            raise TypeError("Scikit-learn cross-validation object please")

        # Check if global model is fitted... and if not, fit it using all of X
        if PCA_model.isfitted is False:
            PCA_model.fit(x)
        # Make a copy of the object, to ensure the internal state doesn't come out differently from the
        # cross validation method call...
        cv_pipeline = deepcopy(self)

        # Initialise predictive residual sum of squares variable (for whole CV routine)
        total_press = 0
        # Calculate Sum of Squares SS in whole dataset
        ss = np.sum((cv_pipeline.scaler.transform(x)) ** 2)
        # Initialise list for loadings and for the VarianceExplained in the test set values
        # Check if model has loadings, as in case of kernelPCA these are not available
        if hasattr(self.pca_algorithm, 'components_'):
            loadings = []

        # cv_varexplained_training is a list containing lists with the SingularValue/Variance Explained metric
        # as obtained in the training set during fitting.
        # cv_varexplained_test is a single R2X measure obtained from using the
        # model fitted with the training set in the test set.
        cv_varexplained_training = []
        cv_varexplained_test = []

        # Default version (press_impute = False) will perform
        #  Row/Observation-Wise CV - Faster computationally, but has some limitations
        # See Bro R. et al, Cross-validation of component models: A critical look at current methods,
        # Analytical and Bioanalytical Chemistry 2008
        # press_impute method requires computational optimization, and is under construction
        for xtrain, xtest in cv_method.split(x):
            cv_pipeline.fit(x[xtrain, :])
            # Calculate R2/Variance Explained in test set
            # To calculate an R2X in the test set

            xtest_scaled = cv_pipeline.scaler.transform(x[xtest, :])

            tss = np.sum((xtest_scaled) ** 2)
            # Append the var explained in training set for this round and loadings for this round
            cv_varexplained_training.append(cv_pipeline.pca_algorithm.explained_variance_ratio_)
            if hasattr(self.pca_algorithm, 'components_'):
                loadings.append(cv_pipeline.loadings)

            if press_impute is True:
                press_testset = 0
                for column in range(0, x[xtest, :].shape[1]):
                    xpred = cv_pipeline.scaler.transform(cv_pipeline._press_impute_pinv(x[xtest, :], column))
                    press_testset += np.sum(np.square(xtest_scaled[:, column] - xpred[:, column]))
                cv_varexplained_test.append(1 - (press_testset / tss))
                total_press += press_testset
            else:
                # RSS for row wise cross-validation
                pred_scores = cv_pipeline.transform(x[xtest, :])
                pred_x = cv_pipeline.scaler.transform(cv_pipeline.inverse_transform(pred_scores))
                rss = np.sum(np.square(xtest_scaled - pred_x))
                total_press += rss
                cv_varexplained_test.append(1 - (rss / tss))

        # Create matrices for each component loading containing the cv values in each round
        # nrows = nrounds, ncolumns = n_variables
        # Check that the PCA model has loadings
        if hasattr(self.pca_algorithm, 'components_'):
            cv_loads = []
            for comp in range(0, self.ncomps):
                cv_loads.append(np.array([x[comp] for x in loadings]))

            # Align loadings due to sign indeterminacy.
            # The solution followed here is to select the sign that gives a more similar profile to the
            # Loadings calculated with the whole data.
            # TODO add scores for CV scores, but still need to check the best way to do it properly
            # Don't want to enforce the common "just average everything" and interpret score plot behaviour...
            for cvround in range(0, cv_method.n_splits):
                for currload in range(0, self.ncomps):
                    choice = np.argmin(np.array([np.sum(np.abs(self.loadings - cv_loads[currload][cvround, :])),
                                                 np.sum(
                                                     np.abs(self.loadings - cv_loads[currload][cvround, :] * -1))]))
                    if choice == 1:
                        cv_loads[currload][cvround, :] = -1 * cv_loads[currload][cvround, :]

        # Calculate total sum of squares
        # Q^2X
        q_squared = 1 - (total_press / ss)
        # Assemble the dictionary and data matrices
        if self.cvParameters is not None:
            self.cvParameters['Mean_VarExpRatio_Training'] = np.array(cv_varexplained_training).mean(axis=0)
            self.cvParameters['Stdev_VarExpRatio_Training'] = np.array(cv_varexplained_training).std(axis=0)
            self.cvParameters['Mean_VarExp_Test'] = np.mean(cv_varexplained_test)
            self.cvParameters['Stdev_VarExp_Test'] = np.std(cv_varexplained_test)
            self.cvParameters['Q2X'] = q_squared
        else:
            self.cvParameters = {'Mean_VarExpRatio_Training': np.array(cv_varexplained_training).mean(axis=0),
                                 'Stdev_VarExpRatio_Training': np.array(cv_varexplained_training).std(axis=0),
                                 'Mean_VarExp_Test': np.mean(cv_varexplained_test),
                                 'Stdev_VarExp_Test': np.std(cv_varexplained_test),
                                 'Q2X': q_squared}
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
    :return: Permuted null distribution for loading vector values.
    :rtype: numpy.ndarray, shape [ncomps, n_perms, n_features]
    :raise ValueError: If there is a problem with the input x data or during the procedure.
    """
    # TODO: Work in progress, more as a curiosity
    try:
        # Check if global model is fitted... and if not, fit it using all of X
        if self._isfitted is False or self.loadings is None:
            self.fit(x)
        # Make a copy of the object, to ensure the internal state doesn't come out differently from the
        # cross validation method call...
        permute_class = deepcopy(self)
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
                                             np.sum(np.abs(
                                                 self.loadings - permuted_loads[currload][perm_n, :] * -1))]))
                if choice == 1:
                    permuted_loads[currload][perm_n, :] = -1 * permuted_loads[currload][perm_n, :]
        return permuted_loads
    except ValueError as verr:
        raise verr


def permutationtest_components(self, x, nperms=1000):
    """
    Unfinished
    Permutation test for a whole component. Also outputs permuted null distributions for the loadings.

    :param x: Data matrix.
    :type x: numpy.ndarray, shape [n_samples, n_features]
    :param int nperms: Number of permutations.
    :return: Permuted null distribution for the component metrics (VarianceExplained and R2).
    :rtype: numpy.ndarray, shape [ncomps, n_perms, n_features]
    :raise ValueError: If there is a problem with the input data.
    """
    # TODO: Work in progress, more out of curiosity
    try:
        # Check if global model is fitted... and if not, fit it using all of X
        if self._isfitted is False:
            self.fit(x)
        # Make a copy of the object, to ensure the internal state doesn't come out differently from the
        # cross validation method call...
        permute_class = deepcopy(self)
        # Initalise list for loading distribution
        permuted_varExp = []
        for permutation in range(0, nperms):
            # Copy original column order, shuffle array in place...
            orig = np.copy(x)
            # np.random.shuffle(x.T)
            # ... Fit model and replace original data
            permute_class.fit(x)
            x = orig
            permuted_varExp.append(permute_class.ModelParameters['VarExpRatio'])
        return permuted_varExp

    except ValueError as verr:
        raise verr

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

    # TODO Double check improved algorithms and methods for PRESS estimation for PCA in general
    # TODO Implement Camacho et al, column - erfk to increase computational efficiency
    # TODO check bi-cross validation
    try:
        # Scaling check for consistency
        if self.scaler is not None:
            xscaled = self.scaler.transform(x)
        else:
            xscaled = x
        # Following from reference 1
        to_pred = np.delete(xscaled, var_to_pred, axis=1)
        topred_loads = np.delete(self.loadings.T, var_to_pred, axis=0)
        imputed_x = np.dot(np.dot(to_pred, np.linalg.pinv(topred_loads).T), self.loadings)
        if self.scaler is not None:
            imputed_x = self.scaler.inverse_transform(imputed_x)
        return imputed_x
    except ValueError as verr:
        raise verr