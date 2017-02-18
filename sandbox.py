import pandas as pds
from ChemometricsPLS import ChemometricsPLS as chempls
import matplotlib.pyplot as plt
import numpy as np
from ChemometricsScaler import ChemometricsScaler


# Read lipofit data, perfect example for CV testing since it has classes and nested data
matrix = pds.read_csv('ExampleFile(TRACElipofitmat).csv')
matrix = matrix[matrix['Outlier?'] != True]
# the X matrix
xmat = matrix.iloc[:, 9::]
y_cont = matrix['TPTG in mg/dL']
y_cont2 = matrix.iloc[:, 8:10]

y_dis = matrix['Sex']
y_dis = pds.Categorical(y_dis).codes

#scaler = chemsc(1)

ples = chempls(1, xscaler=ChemometricsScaler(1), yscaler=ChemometricsScaler(1))

y = y_cont.values

x = xmat.values

ples.fit(x, y)

ples.cross_validation(x, y, testset_scale=True)
