from pyChemometrics import ChemometricsScaler, ChemometricsPLS
import numpy as np
import pandas as pds

# Use the standard datasets
t_dset = pds.read_csv('./tests/test_data/classification_twoclass.csv')
xmat = t_dset.iloc[:, 1::].values
y = t_dset.iloc[:, 0].values

x_scaler = ChemometricsScaler(1)
y_scaler = ChemometricsScaler(1)

plsmodel = ChemometricsPLS(ncomps=3, xscaler=x_scaler, yscaler=y_scaler)

plsmodel.fit(xmat, y)

# Random seed
np.random.seed(0)

plsmodel.cross_validation(xmat, y)


np.savetxt('./tests/test_data/pls_loadings_p.csv', plsmodel.modelParameters, fmt='%.18e', delimiter=',', newline='\n',
           header='', footer='', comments='#')
np.savetxt('./tests/test_data/pls_scores_t.csv', plsmodel.modelParameters, fmt='%.18e',
           delimiter=',', newline='\n', header='', footer='', comments='#')

np.savetxt('./tests/test_data/pls_weights_w.csv', plsmodel.modelParameters, fmt='%.18e', delimiter=',', newline='\n',
           header='', footer='', comments='#')

np.savetxt('./tests/test_data/pls_loadings_p.csv', plsmodel.cvParameters, fmt='%.18e', delimiter=',', newline='\n',
           header='', footer='', comments='#')
np.savetxt('./tests/test_data/pls_scores_t.csv', plsmodel.cvParameters, fmt='%.18e',
           delimiter=',', newline='\n', header='', footer='', comments='#')

np.savetxt('./tests/test_data/pls_weights_w.csv', plsmodel.cvParameters, fmt='%.18e', delimiter=',', newline='\n',
           header='', footer='', comments='#')