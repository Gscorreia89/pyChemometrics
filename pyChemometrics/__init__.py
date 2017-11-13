from pyChemometrics.ChemometricsPCA import ChemometricsPCA
from pyChemometrics.ChemometricsPLS import ChemometricsPLS
from pyChemometrics.ChemometricsScaler import ChemometricsScaler
from pyChemometrics.ChemometricsPLS_Logistic import ChemometricsPLS_Logistic
from pyChemometrics.ChemometricsPLSDA import ChemometricsPLSDA
from pyChemometrics.ChemometricsPLS_LDA import ChemometricsPLS_LDA

__version__ = '0.1'

__all__ = ['ChemometricsScaler', 'ChemometricsPCA', 'ChemometricsPLS',
           'ChemometricsPLS_Logistic', 'ChemometricsPLSDA', 'ChemometricsPLS_LDA']

"""
The pyChemometrics provides objects which wrap pre-existing scikit-learn PCA and PLS algorithms and adds model assessment metrics and
functions common in the Chemometrics literature.
"""
