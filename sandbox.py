__author__ = 'gd2212'
# general file for testing, to get started

import pandas as pds
import os
import PCA

os.chdir(r'C:\Users\Goncalo\PycharmProjects\Chemometrics')
matrix = pds.read_csv('ExampleFile(TRACElipofitmat).csv')

xmat = matrix.iloc[:, 8:]


ples = PCA.PCA()


