from pyChemometrics.ChemometricsPCA import ChemometricsPCA
from pyChemometrics.ChemometricsPLS import ChemometricsPLS
from pyChemometrics.ChemometricsScaler import ChemometricsScaler
from pyChemometrics.ChemometricsPLS_Logistic import ChemometricsPLS_Logistic
from pyChemometrics.ChemometricsPLSDA import ChemometricsPLSDA
from pyChemometrics.ChemometricsPLS_LDA import ChemometricsPLS_LDA

__version__ = '0.13.4'

__all__ = ['ChemometricsScaler', 'ChemometricsPCA', 'ChemometricsPLS',
           'ChemometricsPLS_Logistic', 'ChemometricsPLSDA', 'ChemometricsPLS_LDA']

"""
The pyChemometrics provides objects which wrap pre-existing scikit-learn PCA and PLS algorithms and adds 
model assessment metrics and functions common in the Chemometrics literature.

ChemometricsScaler - Scaler object used in all objects, capable of performing a wide data scaling options seen in the
chemometric and metabonomic literature (mean centering, pareto and unit variance scaling).

ChemometricsPCA - PCA analysis object.

ChemometricsPLS - Object for Partial Least Squares regression analysis and regression quality metrics.

Chemometrics PLSDA - Object for Partial Least Squares Discriminant analysis (PLS followed by transformation of the 
Y prediction into class membership). Supports both 1vs1 and Multinomial classification schemes (although ROC curves
and quality control metrics for Multinomial are still work in progress).

ChemometricsPLS_Logistic - Object for Partial Least Squares followed by logistic regression on the scores. Alternative 
for PLS-DA, where PLS is used as a pre-processing method and Logistic regression to obtain a more formal class 
prediction method.

ChemometricsPLS_LDA - Partial Least Squares followed by Quadratic discriminant analysis on the scores. Similar to
ChemometricsPLS_Logistic.
"""
