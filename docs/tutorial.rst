Using the pyChemometrics objects
--------------------------------
pyChemometrics is a python 3.5 library for multivariate chemometric data analysis.

The main objects ChemometricsPCA, ChemometricsPLS and ChemometricsPLSDA consist of wrappers for scikit-learn
Principal Component Analysis and Partial Least Squares Regression objects. They have been made to mimic as much as possible
scikit-learn classifiers, from their internal properties, and therefore can be interfaced with other
components of scikit-learn, such as the sklearn::`Pipeline`.

These wrappers contain implementations of various routines and metrics commonly seen in the Chemometric and metabonomic literature.
PRESS and Q2Y estimation, permutation testing, Hotelling T2 for outlier detection of scores, VIP scores for variable importance.
Pareto and Unit-Variance scaling.

Each of these objects uses ChemometricsScaler objects to automatically handle the scaling of the X and Y data matrices.

Scaling
=======

The :py:class:`~pyChemometrics.ChemometricsScaler` object handles the scaling of the data matrices. The main s
The data is always. The choice of the power determines the type of scaling. For example, scaling_power = 0 performs column centering
only, scaling_power = 1/2 Pareto scaling and scaling_power = 1 UV (Unit Variance scaling or standardisation).

:py:class:`~pyChemometrics.ChemometricsPCA` object. The scaler parameter expects a :py:class:`~pyChemometrics.ChemometricsScaler`
with the default options and Unit-Variance scaling

    pca_model = pyChemometrics.ChemometricsPCA(...)

The pyChemometrics objects contain methods similar to the ones defined in the scikit-learn Transformer, Classifier
and Regressor Mixins, for example, .fit, .transform , .predict and .score.


    pca_model.fit(X)
    # Obtain the scores (T), the lower dimensional representation of data.
    t_scores  = pca_model.transform(X)
    # Obtain the reconstructed dataset from the T scores.
    pca_model.inverse_transform(scores)


Principal Component Analysis
============================
Principal Component Analysis is provided by the :py:class:`~pyChemometrics.ChemometricsPCA` object.

:py:class:`~pyChemometrics.ChemometricsPCA` object. The scaler parameter expects a :py:class:`~pyChemometrics.ChemometricsScaler`
with the default options and Unit-Variance scaling

    pca_model = pyChemometrics.ChemometricsPCA(...)

    pca_model.fit(X)

    t_scores pca_model.transform(X)

    pca_model.inverse_transform(scores)

The scores and loadings obtained for each component upon calling the .fit method are set as atributes of the model.

The modelParameters dictionary contains the following keys:
    - VarExp: Total variance explained by the model, per component
    - VarExpRatio: % of variance explained, per component
    - R2X: The variance explained by the model in the fitting/training set. Calculated using the model residuals.
    - S0: The denominator for calculation of the Normalised DmodX score.

Performing model cross_validation using the :py:class:`~pyChemometrics.ChemometricsPCA.cross_validation()` method
generates another dictionary atribute, cvParameters. These contain the mean and standard deviation values obtained
from the multiple folds or sampling repeats performed the cross-validation, and if cross_validation method was called
with outputdist = True, also the whole distribution obtained by CV for each parameter.

The cvParameters dictionary contains these keys:
    - Mean_Loadings: Average loading vectors during cross-validation
    - Stdev_Loadings: Standard deviation of the loading vectors

If the outputdist option is set to True when performing cross validation, cvParameters will contain extra keys with
numpy.ndarrays containing all the model parameters (scores, loadings, goodness of fit metrics, etc) obtained for each model fitted
during CV.

The main
The methods provided by these objects
The pyChemometrics objects follow a similar logic Similarly to scikit-learn:

Partial Least Squares Regression
================================
The standard Partial Least Squares object

The scores and loadings obtained for each component upon calling the .fit method are set as atributes of the model.

    - scores_t:
    - scores_u:
    - weights_w:
    - weights_c:
    - loadings_p:
    - loadings_q:
    - rotations_ws:
    - rotations_cs:
    - b_u:
    - b_t:
    - beta_coeffs:
    - logistic_coefs:
    - n_classes:

The modelParameters dictionary contains the following keys:
    - R2Y: Total variance explained by the model, per component
    - R2X: % of variance explained, per component
    - SSX:
    - SSY:
    - SSXcomp: The variance explained by the model in the fitting/training set. Calculated using the model residuals.
    - SSYcomp: The denominator for calculation of the Normalised DmodX score.

Performing model cross_validation using the :py:class:`~pyChemometrics.ChemometricsPLS.cross_validation()` method
generates another dictionary atribute, cvParameters. These contain the mean and standard deviation values obtained
from the multiple folds or sampling repeats performed the cross-validation, and if cross_validation method was called
with outputdist = True, also the whole distribution obtained by CV for each parameter.

The cvParameters dictionary contains these keys:
    - Mean_Loadings: Average loading vectors during cross-validation
    - Stdev_Loadings: Standard deviation of the loading vectors

If the outputdist option is set to True when performing cross validation, cvParameters will contain extra keys with
numpy.ndarrays containing all the model parameters (scores, loadings, goodness of fit metrics, etc) obtained for each model fitted
during CV.


:py:class:`ChemometricsPLS`

Partial Least Squares - Discriminant Analysis
=============================================

The :py:class:`~pyChemometrics.ChemometricsPLSDA` object shares many features with the :py:class:`ChemometricsPLS` object.

Calling the fit method will fill in these

    - scores_t:
    - scores_u:
    - weights_w:
    - weights_c:
    - loadings_p:
    - loadings_q:
    - rotations_ws:
    - rotations_cs:
    - b_u:
    - b_t:
    - beta_coeffs:
    - logistic_coefs:
    - n_classes:

However, this object expects either a singly Y vector containing, or a dummy matrix. The singly Y vector encoding class membership
is re-coded as a dummy matrix of dimensions [n observations x m classes] as part of the algorithm.

The scores and loadings obtained for each component upon calling the .fit method are set as atributes of the model.

The modelParameters dictionary attributes are contains the following keys:
    The 'PLS' subdictionary contains all the values pertaining to the PLS regression algorithm.
    - R2Y: Total variance explained by the model, per component
    - R2X: % of variance explained, per component
    - SSX:
    - SSY:
    - SSXcomp: The variance explained by the model in the fitting/training set. Calculated using the model residuals.
    - SSYcomp: The denominator for calculation of the Normalised DmodX score.
    The 'DA' subdictionary contains the classification metrics obtained by scoring the class predictions with the known truth.
    - Balanced accuracy:
    - F1 measure:
    - Precision:
    - Recall:
    - ROC curve:
    - AUC:
    - 01-Loss:
    - MCC:

Performing model cross_validation using the :py:class:`~pyChemometrics.ChemometricsPLS.cross_validation()` method
generates another dictionary atribute, cvParameters. These contain the mean and standard deviation values obtained
from the multiple folds or sampling repeats performed the cross-validation, and if cross_validation method was called
with outputdist = True, also the whole distribution obtained by CV for each parameter.

The cvParameters dictionary contains these keys:
    - Mean_Loadings: Average loading vectors during cross-validation
    - Stdev_Loadings: Standard deviation of the loading vectors

Additionaly, the discriminant analysis also contains the mean and standard deviation parameters for the DA component.
    - Mean_Accuracy:
    - Stdev_Accuracy:

If the outputdist option is set to True when performing cross validation, cvParameters will contain extra keys with
numpy.ndarrays containing all the model parameters (scores, loadings, goodness of fit metrics, etc) obtained for each model fitted
during CV.

Partial Least Squares - Logistic Regression
===========================================

The :py:class:`~pyChemometrics.ChemometricsPLS_Logistic` object shares many features with the :py:class:`ChemometricsPLS` object.

    - scores_t:
    - scores_u:
    - weights_w:
    - weights_c:
    - loadings_p:
    - loadings_q:
    - rotations_ws:
    - rotations_cs:
    - b_u:
    - b_t:
    - beta_coeffs:
    - logistic_coefs:
    - n_classes:

Calling the fit method will fill in these

Partial Least Squares - Linear Discriminant Analysis
====================================================

The :py:class:`~pyChemometrics.ChemometricsPLS_LDA` object shares many features with the :py:class:`ChemometricsPLS_LDA` object.

    - scores_t:
    - scores_u:
    - weights_w:
    - weights_c:
    - loadings_p:
    - loadings_q:
    - rotations_ws:
    - rotations_cs:
    - b_u:
    - b_t:
    - beta_coeffs:
    - logistic_coefs:
    - n_classes:

Calling the fit method will fill in these