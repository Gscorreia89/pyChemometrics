from pyChemometrics.ChemometricsPCA import ChemometricsPCA
from pyChemometrics.ChemometricsPLS import ChemometricsPLS
from pyChemometrics.ChemometricsScaler import ChemometricsScaler

__version__ = '0.1'

__all__ = ['ChemometricsScaler', 'ChemometricsPCA', 'ChemometricsPLS']

"""
The pyChemometrics provides objects which wrap pre-existing scikit-learn PCA and PLS algorithms and adds model assessment metrics and
functions common in the Chemometrics literature.
"""
