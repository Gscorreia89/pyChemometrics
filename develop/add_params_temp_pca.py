from pyChemometrics import ChemometricsScaler, ChemometricsPCA
import numpy as np


import pandas as pds

t_dset = pds.read_csv('./tests/test_data/classification_twoclass.csv')
xmat = t_dset.iloc[:, 1::].values


x_scaler = ChemometricsScaler(1)
pcamodel = ChemometricsPCA(ncomps=3, scaler=x_scaler)

pcamodel.fit(xmat)

#pcamodel._screecv_optimize_ncomps(xmat, 10, stopping_condition=0.05)

np.random.seed(0)

pcamodel.cross_validation(xmat)

pcamodel._screecv_optimize_ncomps(xmat, 10, stopping_condition=0.05)


np.savetxt('./tests/test_data/pca_loadings.csv', pcamodel.loadings, fmt='%.18e', delimiter=',', newline='\n',
           header='', footer='', comments='#')

np.savetxt('./tests/test_data/pca_scores.csv', pcamodel.scores, fmt='%.18e',
           delimiter=',', newline='\n', header='', footer='', comments='#')

np.savetxt('./tests/test_data/pca_dmodx.csv', pcamodel.dmodx(xmat), fmt='%.18e',
           delimiter=',', newline='\n', header='', footer='', comments='#')

mean_loadings = np.array(pcamodel.cvParameters['Mean_Loadings'])
stdev_loadings = np.array(pcamodel.cvParameters['Stdev_Loadings'])

cvloads = np.r_[mean_loadings, stdev_loadings]

np.savetxt('./tests/test_data/pca_cvloads.csv', cvloads, fmt='%.18e',
           delimiter=',', newline='\n', header='', footer='', comments='#')

x_scaler_par = ChemometricsScaler(1/2)
pcamodel_par = ChemometricsPCA(ncomps=3, scaler=x_scaler_par)

pcamodel_par.fit(xmat)

np.savetxt('./tests/test_data/pca_scores_par.csv', pcamodel_par.scores, fmt='%.18e', delimiter=',',
           newline='\n', header='', footer='', comments='#')
np.savetxt('./tests/test_data/pca_loadings_par.csv', pcamodel_par.loadings, fmt='%.18e',
           delimiter=',', newline='\n', header='', footer='', comments='#')

x_scaler_mc = ChemometricsScaler(0)
pcamodel_mc = ChemometricsPCA(ncomps=3, scaler=x_scaler_mc)

pcamodel_mc.fit(xmat)

np.savetxt('./tests/test_data/pca_scores_mc.csv', pcamodel_mc.scores, fmt='%.18e',
           delimiter=',', newline='\n', header='', footer='', comments='#')
np.savetxt('./tests/test_data/pca_loadings_mc.csv', pcamodel_mc.loadings, fmt='%.18e',
           delimiter=',', newline='\n', header='', footer='', comments='#')
