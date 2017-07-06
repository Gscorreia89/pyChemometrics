import unittest

import numpy as np
from sklearn.datasets import make_regression

from pyChemometrics import ChemometricsScaler, ChemometricsPLS

"""
Suite of tests to assess correctness of the PLS Regression object.
Cross - checked with R's pls, and SIMCA P.
"""

dataset = {'X': np.array([]), 'Y':np.array([])}
expected_weights = {'w': [], 'c': []}
expected_scores = {'t': [], 'u': []}
expect_R2 = {'Y': [], 'X': []}
expected_loadings = {'p': [], 'q': []}
expected_rotations = {'ws': [], 'cs': []}
expected_regression_coefs = {'beta': []}
expected_vip = {'vip_w': []}
expected_prediction = {'y': [], 'x': []}
expected_bs = {'b_u': [], 'b_t': []}


class test_pls_regression(unittest.TestCase):
    """
    Verify agreement of PLS algorithms under different objects and conditions
    """

    def setUp(self):

        # Generate 2 fake classification datasets, one with 2 classes and another with 3
        self.twoclass_dataset = make_regression(40, n_features=100, n_informative=5)
        x_scaler = ChemometricsScaler(1)
        y_scaler = ChemometricsScaler(1, with_mean=True, with_std=False)
        self.plsreg = ChemometricsPLS(n_comps=3, xscaler=x_scaler, y_scaler=y_scaler)

    def test_single_y(self):
        self.plsreg.fit(self.twoclass_dataset[0], self.twoclass_dataset([1]))
        self.plslog.fit(self.twoclass_dataset[0], self.twoclass_dataset([1]))
        self.assertEqual(self.plsreg.rotations_cs, self.plslog.rotations_cs)

    def test_multi_y(self):
        self.plsreg.fit(self.threeclass_dataset[0], self.threeclass_dataset([1]))
        self.plslog.fit(self.threeclass_dataset[0], self.threeclass_dataset([1]))
        self.assertEqual(self.data.noFeatures, self.noFeat)

    def test_(self):
        self.assertEqual()

if __name__ == '__main__':
    unittest.main()


