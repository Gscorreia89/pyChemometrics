pyChemometrics objects
----------------------

The main objects ChemometricsPCA, ChemometricsPLS and ChemometricsPLSDA consist of wrappers for scikit-learn
Principal Component Analysis and Partial Least Squares Regression objects. They have been made to mimic as much as possible
scikit-learn classifiers, from their internal properties, and therefore can be interfaced with other
components of scikit-learn, such as the a klearn::`Pipeline`.

These wrappers contain implementations of various routines and metrics commonly seen in the Chemometric and metabonomic literature.
PRESS and Q2Y estimation, permutation testing, Hotelling T2 for outlier detection of scores, VIP scores for variable importance.
Pareto and Unit-Variance scaling.

Each of these objects uses ChemometricsScaler objects to automatically handle the scaling of the X and Y data matrices.

.. automodule:: pyChemometrics

ChemometricsPCA
===============

.. autoclass:: pyChemometrics.ChemometricsPCA
  :members:

ChemometricsPLS
===============

.. autoclass:: pyChemometrics.ChemometricsPLS
  :members:

ChemometricsPLS
===============

.. autoclass:: pyChemometrics.ChemometricsPLSDA
  :members:

ChemometricsScaler
==================

.. autoclass:: pyChemometrics.ChemometricsScaler
  :members:
