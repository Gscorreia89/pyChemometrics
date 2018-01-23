from pyChemometrics import ChemometricsScaler, ChemometricsPLS
import numpy as np

np.random.seed(0)

import pandas as pds
# Use the standard datasets
t_dset = pds.read_csv('./tests/test_data/regression.csv')
xmat = t_dset.iloc[:, 1:4].values
y = t_dset.iloc[:, 0].values

y = y[np.newaxis].T

mc_scaler = ChemometricsScaler(0)
uv_scaler = ChemometricsScaler(1)
par_scaler = ChemometricsScaler(1/2)

xmat_mc = mc_scaler.fit_transform(xmat)
y_mc = mc_scaler.fit_transform(y)

xmat_uv = uv_scaler.fit_transform(xmat)
y_uv = uv_scaler.fit_transform(y)

xmat_par = par_scaler.fit_transform(xmat)
y_par = par_scaler.fit_transform(y)


np.savetxt('./tests/test_data/scaler_xmat_mc.csv', xmat_mc, fmt='%.18e', delimiter=',', newline='\n',
           header='', footer='', comments='#')

np.savetxt('./tests/test_data/scaler_xmat_uv.csv', xmat_uv, fmt='%.18e', delimiter=',', newline='\n',
           header='', footer='', comments='#')

np.savetxt('./tests/test_data/scaler_xmat_par.csv', xmat_par, fmt='%.18e', delimiter=',', newline='\n',
           header='', footer='', comments='#')

np.savetxt('./tests/test_data/scaler_y_mc.csv', y_mc, fmt='%.18e', delimiter=',', newline='\n',
           header='', footer='', comments='#')

np.savetxt('./tests/test_data/scaler_y_uv.csv', y_uv, fmt='%.18e', delimiter=',', newline='\n',
           header='', footer='', comments='#')

np.savetxt('./tests/test_data/scaler_y_par.csv', y_par, fmt='%.18e', delimiter=',', newline='\n',
           header='', footer='', comments='#')
