__author__ = 'gd2212'
# general file for testing, to get started

import pandas as pds
#from ChemometricsPLS import ChemometricsPLS as chempls
from sklearn.cross_decomposition import PLSRegression, PLSCanonical, PLSSVD
import matplotlib.pyplot as plt
from sklearn.base import clone
import numpy as np
from ChemometricsScaler import ChemometricsScaler as chemsc
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, BayesianRidge


# Read lipofit data, perfect example for CV testing since it has classes and nested data
matrix = pds.read_csv('ExampleFile(TRACElipofitmat).csv')
matrix = matrix[matrix['Outlier?'] != True]
# the X matrix
xmat = matrix.iloc[:, 9::]
y_cont = matrix['TPTG in mg/dL']
y_dis = matrix['Sex']
y_dis = pds.Categorical(y_dis).codes


#scaler = chemsc(1)

ples = PLSCanonical(1, scale=False)

y = y_cont.values
yc = y - np.mean(y)

x = xmat.values
xc = xmat.values - np.mean(xmat.values, 0)

ples.fit(xc, yc)

yp = ples.predict(xc).squeeze()
rssy = np.sum((yc - yp)**2)

xp = np.dot(ples.x_scores_, ples.x_loadings_.T)
rssx = np.sum((xc - xp)**2)
tssy = np.sum(yc**2)
tssx = np.sum(xc**2)

r2x = 1 - (rssx/tssx)
r2y = 1 - (rssy/tssy)


xtypred = np.sum(np.dot(xp.T, yp))
xtyor = np.sum(np.dot(xc.T, yc))