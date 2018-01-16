from pyChemometrics import ChemometricsScaler, ChemometricsPLS
import numpy as np


import pandas as pds
# Use the standard datasets
t_dset = pds.read_csv('./tests/test_data/regression.csv')
xmat = t_dset.iloc[:, 1::].values
y = t_dset.iloc[:, 0].values

x_scaler = ChemometricsScaler(1)
y_scaler = ChemometricsScaler(1)

plsmodel = ChemometricsPLS(ncomps=3, xscaler=x_scaler, yscaler=y_scaler)

plsmodel.fit(xmat, y)

np.random.seed(0)

plsmodel.cross_validation(xmat, y)

np.savetxt('./tests/test_data/pls_loadings_p.csv', plsmodel.loadings_p, fmt='%.18e', delimiter=',', newline='\n',
           header='', footer='', comments='#')
np.savetxt('./tests/test_data/pls_scores_t.csv', plsmodel.scores_t, fmt='%.18e',
           delimiter=',', newline='\n', header='', footer='', comments='#')

np.savetxt('./tests/test_data/pls_weights_w.csv', plsmodel.weights_w, fmt='%.18e', delimiter=',', newline='\n',
           header='', footer='', comments='#')

np.savetxt('./tests/test_data/pls_weights_c.csv', plsmodel.weights_c, fmt='%.18e', delimiter=',', newline='\n',
           header='', footer='', comments='#')

np.savetxt('./tests/test_data/pls_loadings_q.csv', plsmodel.loadings_q, fmt='%.18e', delimiter=',', newline='\n',
           header='', footer='', comments='#')
np.savetxt('./tests/test_data/pls_scores_u.csv', plsmodel.scores_u, fmt='%.18e',
           delimiter=',', newline='\n', header='', footer='', comments='#')

np.savetxt('./tests/test_data/pls_dmodx.csv', plsmodel.dmodx(xmat), fmt='%.18e',
           delimiter=',', newline='\n', header='', footer='', comments='#')

np.savetxt('./tests/test_data/pls_betas.csv', plsmodel.beta_coeffs, fmt='%.18e',
           delimiter=',', newline='\n', header='', footer='', comments='#')

np.savetxt('./tests/test_data/pls_vip.csv', plsmodel.VIP(), fmt='%.18e',
           delimiter=',', newline='\n', header='', footer='', comments='#')

mean_weights = np.array(plsmodel.cvParameters['Mean_Weights_w'])
stdev_weights = np.array(plsmodel.cvParameters['Stdev_Weights_w'])

cvweights = np.r_[mean_weights, stdev_weights]

np.savetxt('./tests/test_data/pls_cvweights.csv', cvweights, fmt='%.18e',
           delimiter=',', newline='\n', header='', footer='', comments='#')

x_scaler_par = ChemometricsScaler(1/2)
y_scaler_par = ChemometricsScaler(1/2)
plsmodel_par = ChemometricsPLS(ncomps=3, xscaler=x_scaler_par, yscaler=y_scaler_par)

plsmodel_par.fit(xmat, y)

np.savetxt('./tests/test_data/pls_scores_t_par.csv', plsmodel_par.scores_t, fmt='%.18e', delimiter=',',
           newline='\n', header='', footer='', comments='#')
np.savetxt('./tests/test_data/pls_betas_par.csv', plsmodel_par.beta_coeffs, fmt='%.18e',
           delimiter=',', newline='\n', header='', footer='', comments='#')
np.savetxt('./tests/test_data/pls_vip_par.csv', plsmodel_par.VIP(), fmt='%.18e',
           delimiter=',', newline='\n', header='', footer='', comments='#')

x_scaler_mc = ChemometricsScaler(0)
y_scaler_mc = ChemometricsScaler(0)
plsmodel_mc = ChemometricsPLS(ncomps=3, xscaler=x_scaler_mc, yscaler=y_scaler_mc)

plsmodel_mc.fit(xmat, y)

np.savetxt('./tests/test_data/pls_scores_t_mc.csv', plsmodel_mc.scores_t, fmt='%.18e',
           delimiter=',', newline='\n', header='', footer='', comments='#')
np.savetxt('./tests/test_data/pls_betas_mc.csv', plsmodel_mc.beta_coeffs, fmt='%.18e',
           delimiter=',', newline='\n', header='', footer='', comments='#')
np.savetxt('./tests/test_data/pls_vip_mc.csv', plsmodel_mc.VIP(), fmt='%.18e',
           delimiter=',', newline='\n', header='', footer='', comments='#')

# Multi-block

x_scaler_par = ChemometricsScaler(1/2)
y_scaler_par = ChemometricsScaler(1/2)
plsmodel_par = ChemometricsPLS(ncomps=3, xscaler=x_scaler_par, yscaler=y_scaler_par)

plsmodel_par.fit(xmat, y)

np.savetxt('./tests/test_data/pls_scores_t_par.csv', plsmodel_par.scores_t, fmt='%.18e', delimiter=',',
           newline='\n', header='', footer='', comments='#')
np.savetxt('./tests/test_data/pls_betas_par.csv', plsmodel_par.beta_coeffs, fmt='%.18e',
           delimiter=',', newline='\n', header='', footer='', comments='#')
np.savetxt('./tests/test_data/pls_vip_par.csv', plsmodel_par.VIP(), fmt='%.18e',
           delimiter=',', newline='\n', header='', footer='', comments='#')

x_scaler_mc = ChemometricsScaler(0)
y_scaler_mc = ChemometricsScaler(0)
plsmodel_mc = ChemometricsPLS(ncomps=3, xscaler=x_scaler_mc, yscaler=y_scaler_mc)

plsmodel_mc.fit(xmat, y)

np.savetxt('./tests/test_data/pls_scores_t_mc.csv', plsmodel_mc.scores_t, fmt='%.18e',
           delimiter=',', newline='\n', header='', footer='', comments='#')
np.savetxt('./tests/test_data/pls_betas_mc.csv', plsmodel_mc.beta_coeffs, fmt='%.18e',
           delimiter=',', newline='\n', header='', footer='', comments='#')
np.savetxt('./tests/test_data/pls_vip_mc.csv', plsmodel_mc.VIP(), fmt='%.18e',
           delimiter=',', newline='\n', header='', footer='', comments='#')