__author__ = 'gd2212'
# general file for testing, to get started

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

#scaler = chemsc(1)

ples = PLSSVD(2, scale=False)
ples2 = PLSCanonical(2, scale=False)
ples3 = PLSRegression(2, scale=False)

y = y_cont.values
yc = y - np.mean(y, 0)

x = xmat.values
xc = xmat.values - np.mean(xmat.values, 0)

ples.fit(xc, yc)
ples2.fit(xc, yc)
ples3.fit(xc, yc)

#yp = ples3.predict(xc).squeeze()
#rssy = np.sum((yc - yp)**2)

bt = np.dot(ples.y_scores_.T, ples.x_scores_)/np.dot(ples.x_scores_.T, ples.x_scores_)
bu = np.dot(ples.x_scores_.T, ples.y_scores_)/np.dot(ples.y_scores_.T, ples.y_scores_)

newbu = np.dot(np.dot(np.linalg.pinv(np.dot(ples.y_scores_.T, ples.y_scores_)), ples.y_scores_.T), ples.x_scores_)
newbt = np.dot(np.dot(np.linalg.pinv(np.dot(ples.x_scores_.T, ples.x_scores_)), ples.x_scores_.T), ples.y_scores_)

bt2 = np.dot(ples2.y_scores_.T, ples2.x_scores_)/np.dot(ples2.x_scores_.T, ples2.x_scores_)
bu2 = np.dot(ples2.x_scores_.T, ples2.y_scores_)/np.dot(ples2.y_scores_.T, ples2.y_scores_)
newbu2 = np.dot(np.dot(np.linalg.pinv(np.dot(ples2.y_scores_.T, ples2.y_scores_)), ples2.y_scores_.T), ples2.x_scores_)
newbt2 = np.dot(np.dot(np.linalg.pinv(np.dot(ples2.x_scores_.T, ples2.x_scores_)), ples2.x_scores_.T), ples2.y_scores_)

bu3 = np.dot(ples3.x_scores_.T, ples3.y_scores_)/np.dot(ples3.y_scores_.T, ples3.y_scores_)
bt3 = np.dot(ples3.y_scores_.T, ples3.x_scores_)/np.dot(ples3.x_scores_.T, ples3.x_scores_)
newbu3 = np.dot(np.dot(np.linalg.pinv(np.dot(ples3.y_scores_.T, ples3.y_scores_)), ples3.y_scores_.T), ples3.x_scores_)
newbt3 = np.dot(np.dot(np.linalg.pinv(np.dot(ples3.x_scores_.T, ples3.x_scores_)), ples3.x_scores_.T), ples3.y_scores_)

xp = np.dot(np.dot(ples.x_scores_, newbu), ples.x_weights_.T)
xp3 = np.dot(np.dot(ples3.y_scores_, newbu3), ples3.x_loadings_.T)

xp2 = np.dot(np.dot(ples2.x_scores_, newbu2), ples2.x_weights_.T)

yp = np.dot(np.dot(ples.x_scores_, newbt), ples.y_weights_.T).squeeze()
yp2 = np.dot(np.dot(ples2.x_scores_, newbt2), ples2.y_weights_.T).squeeze()
yp3 = np.dot(np.dot(ples3.x_scores_, newbt3), ples3.y_weights_.T).squeeze()


# Modelled R2X
#

# Modelled R2Y
# G


rssx = np.sum((xc - xp)**2)
rssx2 = np.sum((xc - xp2)**2)
rssx3 = np.sum((xc - xp3)**2)

rssy = np.sum((yc - yp)**2)
rssy2 = np.sum((yc - yp2)**2)
rssy3 = np.sum((yc - yp3)**2)

tssy = np.sum(yc**2)
tssx = np.sum(xc**2)

r2x = 1 - (rssx/tssx)
r2x2 = 1 - (rssx2/tssx)
r2x3 = 1 - (rssx3/tssx)

r2y = 1 - (rssy/tssy)
r2y2 = 1 - (rssy2/tssy)
r2y3 = 1 - (rssy3/tssy)

tn = ples3.x_scores_
pn = ples3.x_weights_
     #p.linalg.norm(ples3.x_loadings_, axis=0)

def2 = np.dot(tn[:, 1][:, None], pn[:, 1][None, :])
def1 = np.dot(tn[:, 0][:, None], pn[:, 0][None, :])
