# pyChemometrics

### Description
The pyChemometrics package provides implementations of PCA, PLS regression and PLS discriminant 
analysis (PLS-DA) tailored for the analysis of spectroscopy and metabonomic datasets
(especially from Nuclear magnetic resonance spectroscopy and mass spectrometry). 
Some of the common validation metrics and procedures seen in the chemometric and metabonomic literature 
are provided, including different matrix scaling options (mean centring, Pareto and Unit-Variance), 
Leave-one-out-cross-validation (LOOCV), K-Fold cross validation and Monte Carlo CV, calculation of the 
Q2Y and Q2X measures from LOOCV and K-Fold CV, permutation tests and the variable importance for prediction 
(VIP) metric.

## Table of contents
The functionality in this package is provided by a set of objects

 - ChemometricsScaler: Handles the scaling of the data matrices
 - ChemometricsPCA: Principal Component analysis
 - ChemometricsPLS: Partial Least Squares regression
 - ChemometricsPLSDA: Partial Least Squares - Discriminant Analysis
 - ChemometricsPLS_Logistic: Partial Least Squares - Logistic regression using the PLS scores as predictors
 - ChemometricsPLS_LDA: Partial Least Squares - Quadratic discriminant analysis, using the PLS scores as predictors
 
The main objects in this package wrap pre-existing scitkit-learn Principal Component Analysis 
(PCA) and Partial Least Squares (PLS) algorithms, 
and make use of the cross-validation and model selection functionality from scikit-learn.

### Instalation
To install, simply navigate to the main package folder and run

    python setup.py install
    
### License
All code is provided under a BSD-3 license.
