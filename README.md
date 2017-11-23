# pyChemometrics


### Description
The pyChemometrics package provides implementations of PCA, PLS regression and PLS discriminant 
analysis (PLS-DA) tailored for the analysis of spectroscopy and metabonomic datasets
(Nuclear magnetic resonance spectroscopy and mass spectrometry). Some of the common validation metrics 
and procedures seen in the chemometric and metabonomic literature 
are provided. For example, Pareto and unit variance scaling options, Q2Y measure for cross-validation, 
permutation tests, K-fold cross-validation schemes, VIP variable importance metric, for example).

The main objects in this package wrap pre-existing scitkit-learn Principal Component Analysis 
(PCA) and Partial Least Squares (PLS) algorithms, 
and make use of the cross-validation and model selection functionality from scikit-learn.

### Instalation
To install, simply navigate to the main package folder and run

    python setup.py install
    
### License
All code is provided under a BSD-3 license.
