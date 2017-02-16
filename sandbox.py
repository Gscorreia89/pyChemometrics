import pandas as pds
from ChemometricsPLS import ChemometricsPLS as chempls
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
xmat = matrix.iloc[:, 10::]
y_cont = matrix['TPTG in mg/dL']
y_cont2 = matrix.iloc[:, 8:10]

y_dis = matrix['Sex']
y_dis = pds.Categorical(y_dis).codes

scaler = chemsc(1)

ples = chempls(2, xscaler=None, yscaler=None)

y = y_cont.values

x = xmat.values

ples.fit(x, y)

