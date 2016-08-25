__author__ = 'gd2212'
# general file for testing, to get started

import pandas as pds
from PCA import PCA
from sklearn.decomposition import KernelPCA
import matplotlib.pyplot as plt

# Read lipofit data, perfect example for CV testing since it has classes and nested data
matrix = pds.read_csv('ExampleFile(TRACElipofitmat).csv')
# the X matrix
xmat = matrix.iloc[:, 8:]

# Kernel model, including kw

regularpca = PCA()
rbfkernelPCA = PCA(pca_algorithm=KernelPCA, **{'kernel': 'rbf'})

pcascores = regularpca.fit_transform(xmat)
rbfscores = rbfkernelPCA.fit_transform(xmat)


# for now improvised scoreplot...
plt.figure()
plt.scatter(pcascores[:, 0], pcascores[:, 1], c='b')
plt.title('"Classic" PCA')

plt.figure()
plt.scatter(rbfscores[:, 0], rbfscores[:, 1], c='r')
plt.title('Kernel PCA')


