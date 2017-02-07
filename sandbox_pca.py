
import pandas as pds
from ChemometricsPCA import ChemometricsPCA as chempca
import matplotlib.pyplot as plt
from sklearn.base import clone
import numpy as np
from ChemometricsScaler import ChemometricsScaler as chemsc

# Read lipofit data, perfect example for CV testing since it has classes and nested data
#matrix = pds.read_csv('ExampleFile(TRACElipofitmat).csv')
matrix = pds.read_csv('CSVD S LNEG_intensityData.csv')
# the X matrix
#xmat = matrix.iloc[:, 8::]
#y_cont = matrix['TPTG in mg/dL']

xmat = matrix


#r2x = list()
#q2 = list()
#for ncomp in range(1, 3):
#    pca = chempca(ncomp)
#    pca.fit(xmat.values)
#    pca.cross_validation(xmat.values, bro_press=False, outputdist=True)
#    r2x.append(pca.modelParameters['R2X'])
#    q2.append(pca.cvParameters['Q2'])

from sklearn.utils import resample

bt_means = list()
bt_vars = list()
for boot in range(0, 2000):#
    resamp = resample(xmat.values, n_samples=615)
    bt_means.append(np.mean(resamp, 0))
    bt_vars.append(np.std(resamp, 0))

bt_means = np.array(bt_means)
bt_vars = np.array(bt_vars)