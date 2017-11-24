import unittest
import os
import pandas as pds
import numpy as np
from numpy.testing import assert_allclose
from sklearn.model_selection import KFold
from pyChemometrics import ChemometricsScaler, ChemometricsPCA

"""
Suite of tests to assess coherence and functionality of the PCA object.

"""


class TestPCA(unittest.TestCase):
    """

    Verify outputs of the ChemometricsPCA object

    """

    def setUp(self):

        try:
            # Generate a fake classification dataset
            t_dset = pds.read_csv('./test_data/classification_twoclass.csv')
            self.xmat = t_dset.iloc[:, 1::].values

        except (IOError, OSError) as ioerr:
            os.system('python gen_synthetic_datasets.py')
            t_dset = pds.read_csv('./test_data/classification_twoclass.csv')
            self.xmat = t_dset.iloc[:, 1::].values

        self.expected_modelParameters = {'R2X': 0.12913056143673818,
                                          'S0': 0.9803124001345157,
                                         'VarExp': np.array([9.44045066, 8.79710591, 8.11561924]),
                                          'VarExpRatio': np.array([0.04625821, 0.04310582, 0.03976653])}

        self.expected_scores = np.loadtxt('./test_data/pca_scores.csv', delimiter=',')
        self.expected_loadings = np.loadtxt('./test_data/pca_loadings.csv', delimiter=',')

        self.x_scaler = ChemometricsScaler(1)
        self.pcamodel = ChemometricsPCA(ncomps=3, scaler=self.x_scaler)

    def test_fit(self):

        self.pcamodel.fit(self.xmat)

        for key, item in self.expected_modelParameters.items():
            assert_allclose(self.pcamodel.modelParameters[key], item, rtol=1e-05)

        assert_allclose(self.pcamodel.scores, self.expected_scores)
        assert_allclose(self.pcamodel.loadings, self.expected_loadings)

    def test_transform(self):

        self.pcamodel.fit(self.xmat)
        assert_allclose(self.pcamodel.transform(self.xmat), self.expected_scores)

    def test_fit_transform(self):
        assert_allclose(self.pcamodel.fit_transform(self.xmat), self.expected_scores)

    def test_cv(self):
        # Restart the seed and perform cross validation
        np.random.seed(0)
        self.pcamodel.cross_validation(self.xmat, cv_method=KFold(7, True))

        assert_allclose(self.pcamodel.cvParameters['Q2X'], self.expectedcvParams['Q2X'])
        assert_allclose(self.pcamodel.cvParameters['Mean_R2XTrain'], self.expectedcvParams['Mean_T2XTrain'])
        assert_allclose(self.pcamodel.cvParameters['Stdev_R2XTrain'], self.expectedcvParams['Stdev_R2XTrain'])
        assert_allclose(self.pcamodel.cvParameters['Mean_R2XTest'], self.expectedcvParams['Mean_R2XTest'])
        assert_allclose(self.pcamodel.cvParameters['Stdev_R2XTest'], self.expectedcvParams['Stdev_R2XTest'])

    def HotellingT2(self):
        pass

    def test_dmodx(self):
        pass

    def test_scalers(self):
        pass

    def test_outliers(self):
        outliers_t2 = self.pcamodel.outlier()
        outliers_dmodx = self.pcamodel.outlier(self.xmat)
        self.assertAlmostEqual(self.pcamodel.cvParameters, self.outliers)
        return None


if __name__ == '__main__':
    unittest.main()
