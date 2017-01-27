__author__ = 'gd2212'
# general file for testing, to get started

import pandas as pds
from ChemometricsPCA import ChemometricsPCA as chempca
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.base import clone
import matplotlib.pyplot as plt
import numpy as np

# Read lipofit data, perfect example for CV testing since it has classes and nested data
matrix = pds.read_csv('ExampleFile(TRACElipofitmat).csv')
# the X matrix
xmat = matrix.iloc[:, 8::]

# Kernel model, including kw

a = chempca(3)
a.fit(xmat)
a.cross_validation(xmat.values, bro_press=True)


# Bro_press comparison
ncomps = 3

qsquareds = []
r2test = []
for curr_ncomp in range(1, ncomps+1):
    pca = chempca(ncomps=curr_ncomp)
    pca.cross_validation(x=xmat.values, bro_press=True)
    qsquareds.append(pca.cvParameters['Q2'])
    r2test.append(pca.cvParameters['Mean_VarianceExplained_Train'])

qsquareds_nobro= []
r2test_nobro = []
for curr_ncomp in range(1, ncomps+1):
    pca = chempca(ncomps=curr_ncomp)
    pca.cross_validation(x=xmat.values, bro_press=False)
    qsquareds_nobro.append(pca.cvParameters['Q2'])
    r2test_nobro.append(pca.cvParameters['Mean_VarianceExplained_Train'])

%matplotlib qt
width = 0.35
fig, ax = plt.subplots()
left = np.arange(1, ncomps+1)
ax.bar(left , qsquareds_nobro, width/4)
ax.bar(left + 0.25*width, qsquareds, width/4, color='r', alpha=0.5)
ax.bar(left + 0.5*width, r2test_nobro, width/4, color='g')
ax.bar(left + 0.75*width, r2test, width/4, color='k')

plt.show()


# for now improvised scoreplot...
#plt.figure()
#plt.scatter(pcascores[:, 0], pcascores[:, 1], c='b')
#plt.title('"Classic" PCA')

#plt.figure()
#plt.scatter(rbfscores[:, 0], rbfscores[:, 1], c='r')
#plt.title('Kernel PCA')


