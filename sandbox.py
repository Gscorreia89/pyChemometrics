__author__ = 'gd2212'
# general file for testing, to get started

import pandas as pds
import os
from PCA import PCA
from sklearn.decomposition import KernelPCA

matrix = pds.read_csv('ExampleFile(TRACElipofitmat).csv')

xmat = matrix.iloc[:, 8:]

ples = PCA(pca_algorithm=KernelPCA)


