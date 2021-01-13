# pyChemometrics

[![DOI](https://zenodo.org/badge/61749327.svg)](https://zenodo.org/badge/latestdoi/61749327) ![Python38](https://img.shields.io/badge/python-3.8-blue.svg) [![Build Status](https://travis-ci.org/Gscorreia89/pyChemometrics.svg?branch=master)](https://travis-ci.org/Gscorreia89/pyChemometrics) [![codecov](https://codecov.io/gh/gscorreia89/pyChemometrics/branch/master/graph/badge.svg)](https://codecov.io/gh/gscorreia89/pyChemometrics)

### Description
The pyChemometrics package provides implementations of PCA, PLS regression and PLS discriminant 
analysis (PLS-DA) tailored for the analysis of spectroscopy and metabonomic datasets
(especially from Nuclear magnetic resonance spectroscopy and mass spectrometry). 
Some of the common validation metrics and procedures seen in the chemometric and metabonomic literature 
are provided, including different matrix scaling options (mean centring, Pareto and Unit-Variance), 
Leave-one-out-cross-validation (LOOCV), K-Fold cross validation and Monte Carlo CV, calculation of the 
Q2Y and Q2X measures from LOOCV and K-Fold CV, permutation tests and the variable importance for prediction 
(VIP) metric. 

### Documentation and Tutorials
Documentation and information on running the tutorials (Still in progress) can be found in [readthedocs](http://pychemometrics.readthedocs.io/en/stable/).

### Table of contents
The main objects in the package are:

 - ChemometricsScaler: Handles the scaling of the data matrices
 - ChemometricsPCA: Principal Component analysis
 - ChemometricsPLS: Partial Least Squares regression
 - ChemometricsPLSDA: Partial Least Squares - Discriminant Analysis
 - ChemometricsPLS_Logistic: Partial Least Squares - Logistic regression using the PLS scores as predictors
 - ChemometricsPLS_LDA: Partial Least Squares - Quadratic discriminant analysis, using the PLS scores as predictors
 
The main objects in this package wrap pre-existing scikit-learn [Principal Component Analysis 
(PCA)](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) 
and [Partial Least Squares (PLS) algorithms](http://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html), 
and make use of the [cross-validation and model selection functionality from scikit-learn](http://scikit-learn.org/stable/modules/cross_validation.html#cross-validation).

### Instalation
To install, simply navigate to the main package folder and run:

    python setup.py install

Alternatively, using pip (from source):

    pip install /pyChemometricsDirectory/

Or from Pypi:

    pip install pyChemometrics


Installation with pip allows the usage of the uninstall command

    pip uninstall pyChemometrics
    
### License
All code is provided under a BSD 3-clause license. See LICENSE file for more information.
