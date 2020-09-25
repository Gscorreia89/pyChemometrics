def cross_validation(self, x, cv_method=KFold(7, True), outputdist=False, press_impute=False):
    """

    Cross-validation method for the model. Calculates cross-validated estimates for Q2X and other
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
        if self._isfitted is False or self.loadings is None:
            self.fit(x)
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


# @staticmethod
# def stop_cond(model, x):
#    stop_check = getattr(model, modelParameters)
#    if stop_check > 0:
#        return True
#    else:
#        return False

def _screecv_optimize_ncomps(self, x, total_comps=5, cv_method=KFold(7, True), stopping_condition=None):
    """

    Routine to optimize number of components quickly using Cross validation and stabilization of Q2X.

    :param numpy.ndarray x: Data
    :param int total_comps:
    :param sklearn.BaseCrossValidator cv_method:
    :param None or float stopping_condition:
    :return:
    """
    models = list()

    for ncomps in range(1, total_comps + 1):

        currmodel = deepcopy(self)
        currmodel.ncomps = ncomps
        currmodel.fit(x)
        currmodel.cross_validation(x, outputdist=False, cv_method=cv_method, press_impute=False)
        models.append(currmodel)

        # Stopping condition on Q2, assuming stopping_condition is a float encoding percentage of increase from
        # previous Q2X
        # Exclude first component since there is nothing to compare with...
        if isinstance(stopping_condition, float) and ncomps > 1:
            previous_q2 = models[ncomps - 2].cvParameters['Q2X']
            current_q2 = models[ncomps - 1].cvParameters['Q2X']

            if (current_q2 - previous_q2) / abs(previous_q2) < stopping_condition:
                # Stop the loop
                models.pop()
                break
        # Flexible case to be implemented, to allow many other stopping conditions
        elif callable(stopping_condition):
            pass

    q2 = np.array([x.cvParameters['Q2X'] for x in models])
    r2 = np.array([x.modelParameters['R2X'] for x in models])

    results_dict = {'R2X_Scree': r2, 'Q2X_Scree': q2, 'Scree_n_components': len(r2)}
    # If cross-validation has been called
    if self.cvParameters is not None:
        self.cvParameters['R2X_Scree'] = r2
        self.cvParameters['Q2X_Scree'] = q2
        self.cvParameters['Scree_n_components'] = len(r2)
    # In case cross_validation wasn't called before.
    else:
        self.cvParameters = {'R2X_Scree': r2, 'Q2X_Scree': q2, 'Scree_n_components': len(r2)}

    return results_dict