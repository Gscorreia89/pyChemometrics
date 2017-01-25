__author__ = 'gd2212'
# general file for testing, to get started

import pandas as pds
from PCA import ChemometricsPCA as chempca
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.base import clone

# Read lipofit data, perfect example for CV testing since it has classes and nested data
matrix = pds.read_csv('ExampleFile(TRACElipofitmat).csv')
# the X matrix
xmat = matrix.iloc[:, 8::]

# Kernel model, including kw

a = chempca(10)
a.fit(xmat)
a.cross_validation(xmat.values, bro_press=True)

qsquareds = []
for ncomps in range(1, 10):
    pca = chempca(ncomps=ncomps)
    pca.cross_validation(x=xmat.values, bro_press=True)
    qsquareds.append(pca.cvParameters['Q2'])

#bfkernelPCA = PCA(pca_algorithm=KernelPCA, **{'kernel': 'rbf'})

#pcascores = regularpca.fit(xmat)
#rbfscores = rbfkernelPCA.fit_transform(xmat)



# for now improvised scoreplot...
#plt.figure()
#plt.scatter(pcascores[:, 0], pcascores[:, 1], c='b')
#plt.title('"Classic" PCA')

#plt.figure()
#plt.scatter(rbfscores[:, 0], rbfscores[:, 1], c='r')
#plt.title('Kernel PCA')


