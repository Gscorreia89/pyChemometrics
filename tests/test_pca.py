import unittest
import os
import pandas as pds
import numpy as np

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

        self.x_scaler = ChemometricsScaler(1)
        self.pcamodel = ChemometricsPCA(n_comps=3, xscaler=self.x_scaler)

    def test_fit(self):
        self.pcamodel.fit(self.xmat)
        self.assertAlmostEqual(self.pcamodel.modelParameters, self.expected_modelParameters)
        self.assertAlmostEqual(self.pcamodel.scores, self.expected_scores)
        self.assertAlmostEqual(self.pcamodel.loadings, self.expected_loadings)

    def test_transform(self):
        self.assertAlmostEqual(self.pcammodel.transform(self.xmat), self.expected_scores)

    def test_fit_transform(self):
        self.assertAlmostEqual(self.pcammodel.fit_transform(self.xmat), self.expected_scores)

    def test_cv(self):
        self.pcamodel.cross_validation(self.xmat)
        self.assertAlmostEqual(self.pcammodel.cvParameters, self.expectedcvParams)

    def HotellingT2(self):
        pass

    def test_dmodx(self):
        pass

    def test_scalers(self):
        pass

    def test_outliers(self):
        outliers_t2 = self.pcamodel.outlier()
        outliers_dmodx = self.pcamodel.outlier(self.xmat)
        self.assertAlmostEqual(self.pcammodel.cvParameters, self.outliers)
        return None


if __name__ == '__main__':
    unittest.main()
