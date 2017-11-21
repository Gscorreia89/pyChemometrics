import unittest
import os
import pandas as pds
import numpy as np
from sklearn.datasets import make_classification

from pyChemometrics import ChemometricsScaler, ChemometricsPCA

"""
Suite of tests to assess correctness of the PCA object.
Cross - checked with R's pcamethods.
"""

dataset = {'X': np.array([]), 'Y': np.array([])}
expected_scores = {'t': []}
expect_R2 = {'X': []}
expected_loadings = {'p': []}
expected_prediction = {'x': []}


class test_pcamodel(unittest.TestCase):
    """

    Verify outputs of the ChemometricsPCA object

    """

    def setUp(self):

        try:
            # Generate a fake classification dataset
            t_dset = pds.read_csv('./test_data/classification_twoclass.csv')
            self.xmat = t_dset.values

        except (IOError, OSError) as ioerr:
            os.system('python gen_synthetic_datasets.py')
            t_dset = pds.read_csv('./test_data/classification_twoclass.csv')
            self.xmat = t_dset.values

        self.x_scaler = ChemometricsScaler(1)
        self.pcamodel = ChemometricsPCA(n_comps=3, xscaler=self.x_scaler)

    def test_fit(self):
        self.pcamodel.fit(self.xmat)
        self.assertAlmostEqual(self.pcamodel.modelParameters['R2X'], expected)
        self.assertAlmostEqual()

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

    def test_outliers(self):
        outliers_t2 = self.pcamodel.outlier()
        outliers_dmodx = self.pcamodel.outlier(self.xmat)
        self.assertAlmostEqual(self.pcammodel.cvParameters, self.expectedcvParams)
        return None


if __name__ == '__main__':
    unittest.main()
