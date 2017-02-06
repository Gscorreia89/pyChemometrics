__author__ = 'gd2212'
# general file for testing, to get started

import pandas as pds
#from ChemometricsPLS import ChemometricsPLS as chempls
from sklearn.cross_decomposition import PLSRegression
import matplotlib.pyplot as plt
from sklearn.base import clone
import numpy as np
from ChemometricsScaler import ChemometricsScaler as chemsc
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, BayesianRidge

ols = LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)
ols.fit()

# Read lipofit data, perfect example for CV testing since it has classes and nested data
matrix = pds.read_csv('ExampleFile(TRACElipofitmat).csv')
# the X matrix
xmat = matrix.iloc[:, 9::]
y_cont = matrix['TPTG in mg/dL']
y_dis = matrix['Sex']
y_dis = pds.Categorical(y_dis).codes

scaler = chemsc(1)

ols = LinearRegression(fit_intercept=False, normalize=True, copy_X=True, n_jobs=1)
ols.fit(xmat.values, y_cont.values)

rdg = Ridge(alpha=0.1, fit_intercept=False, normalize=True)
rdg.fit(xmat.values, y_cont.values)

brdg = BayesianRidge(normalize=True)
brdg.fit(xmat.values, y_cont.values)

ples = PLSRegression(5, scale=False)
ples.fit(scaler.fit_transform(xmat.values), y_cont.values)

ples_shrink = list()

for ncomp in range(1, 51):
    ples = PLSRegression(ncomp, scale=True)
    ples.fit(scaler.fit_transform(xmat.values), y_cont.values)
    ples_shrink.append(ples.coef_)

ples_shrink = np.array(ples_shrink).squeeze()