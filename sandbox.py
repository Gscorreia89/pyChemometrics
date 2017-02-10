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


# Read lipofit data, perfect example for CV testing since it has classes and nested data
matrix = pds.read_csv('ExampleFile(TRACElipofitmat).csv')
# the X matrix
xmat = matrix.iloc[:, 8::]
y_cont = matrix['TPTG in mg/dL']
y_dis = matrix['Sex']
y_dis = pds.Categorical(y_dis).codes


scaler = chemsc(1)

ples = PLSRegression(3, scale=False)

y = y_cont.values
yc = y - np.mean(y)

x = xmat.values
xc = xmat.values - np.mean(xmat.values, 0)

ples.fit(xc, yc)

yp = ples.predict(xc).squeeze()

rssy = np.sum((yc - yp)**2)

tssy = np.sum(yc**2)
tssx = np.sum(xc**2)

