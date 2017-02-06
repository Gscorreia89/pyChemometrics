
import pandas as pds
from ChemometricsPCA import ChemometricsPCA as chempca
import matplotlib.pyplot as plt
from sklearn.base import clone
import numpy as np
from ChemometricsScaler import ChemometricsScaler as chemsc

# Read lipofit data, perfect example for CV testing since it has classes and nested data
matrix = pds.read_csv('ExampleFile(TRACElipofitmat).csv')
# the X matrix
xmat = matrix.iloc[:, 8::]
y_cont = matrix['TPTG in mg/dL']


r2x = list()


for ncomp in range(1, 15):
    pca = chempca(ncomp)
    pca.fit(xmat.values)
    r2x.append(pca.modelParameters['R2X'])

