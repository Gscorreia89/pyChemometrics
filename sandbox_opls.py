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
xmat = np.c_[matrix.iloc[:, 9::].values, matrix.iloc[:, 8]]
y_cont = matrix['TPTG in mg/dL']
#y_cont2 = matrix.iloc[:, 8:10]

#y_dis = matrix['Sex']
#y_dis2 = matrix[]
#y_dis = pds.Categorical(y_dis).codes

ples = PLSRegression(2, scale=False)

y = y_cont.values
yc = y - np.mean(y, 0)

x = xmat
xc = xmat - np.mean(xmat, 0)

ples.fit(xc, yc)

wo = np.c_[ples.x_weights_[:, 1::], ples.x_weights_[:, 0]]
co = np.c_[ples.y_weights_[:, 1::], ples.y_weights_[:, 0]]
Uo = np.c_[ples.y_scores_[:, 1::], ples.y_scores_[:, 0]]
Qo = np.c_[ples.y_loadings_[:, 1::], ples.y_loadings_[:, 0]]
cso = ples.y_rotations_

To, Ro = np.linalg.qr(np.dot(xc, wo))
Po = np.dot(xc.T, To)

Wostar = np.linalg.lstsq(wo.T, Ro.T)
