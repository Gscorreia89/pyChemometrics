pyChemometrics objects
----------------------
ChemometricsPCA and ChemometricsPLS are the main objects and are meant to be used directly. ChemometricsScaler objects are
used inside the PCA and PLS objects, but its also made available to the user. The ChemometricsPCA and ChemometricsPLS objects
wrap pre-existing scikit-learn implementations of these classifiers, but they provide extra methods for cross-validation,
permutation testing, and calculation of Hotelling T2, VIP's metrics which are commonly used in the Chemometrics Literature.
Since they also inherit from and mimic the scikit-learn model method convention, they can be used and chain within a sklearn::`Pipeline`.

.. automodule:: pyChemometrics

ChemometricsPCA
===============

.. autoclass:: pyChemometrics.ChemometricsPCA
  :members:

ChemometricsPLS
===============

.. autoclass:: pyChemometrics.ChemometricsPLS
  :members:

ChemometricsScaler
==================

.. autoclass:: pyChemometrics.ChemometricsScaler
  :members:
