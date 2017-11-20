from sklearn.datasets import make_classification
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from pyChemometrics.ChemometricsPLSDA import ChemometricsPLSDA
from pyChemometrics.ChemometricsPLS import ChemometricsPLS
from pyChemometrics.ChemometricsScaler import ChemometricsScaler
from pyChemometrics.ChemometricsPLS_Logistic import ChemometricsPLS_Logistic
import pandas as pds
from pyChemometrics.ChemometricsPCA import ChemometricsPCA

fake_data = make_classification(n_samples=100, n_features=200, n_informative=25,
                                              n_redundant=5, n_repeated=0, n_classes=2,
                                              n_clusters_per_class=2, weights=None,
                                              flip_y=0.01, class_sep=1.5, hypercube=True,
                                              shift=0.0, scale=1.0, shuffle=True,
                                              random_state=35624)

fake_x = fake_data[0]
fake_y = fake_data[1]

#ples = ChemometricsPLS(ncomps=1, yscaler=ChemometricsScaler(scale_power=0, with_mean=True))
#ples.fit(fake_x, fake_y)

ples = ChemometricsPLSDA(ncomps=2)
ples.fit(fake_x, fake_y)
ples.cross_validation(fake_x, fake_y)
ples.screePlot(fake_x, fake_y)

permt = ples.permutation_test(fake_x, fake_y, 150)

ples = ChemometricsPCA(ncomps=2)
ples.fit(fake_x)

permt = ples.permutation_test(fake_x, fake_y, 150)