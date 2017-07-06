import unittest
import numpy as np
from numpy.testing import assert_array_equal
from sklearn.datasets import make_classification

from pyChemometrics import ChemometricsScaler, ChemometricsPLS, ChemometricsPLS_Logistic

"""
Suite of tests to ensure that all PLS objects are consistent among each other. 
For example, the ChemometricsPLS object in Regression mode needs to give the same results (coefficients, 
loadings, scores R2s, etc) as the PLS-Classifier objects (ChemometricPLS_Logistic and ChemometricsPLS_QDA), 
provided that we account for the differences in data input and class vector do dummy conversions.
"""


class test_plsobjconsistency(unittest.TestCase):
    """
    Verify agreement of PLS algorithms under different objects and conditions
    """

    def setUp(self):

        # Generate 2 fake classification datasets, one with 2 classes and another with 3
        self.twoclass_dataset = make_classification(40, n_features=100, n_informative=5, n_redundant=5, n_classes=2)
        self.three_classdataset = make_classification(40, n_features=100, n_informative=5, n_redundant=5, n_classes=3)
        # Set up the same y_scalers
        y_scaler = ChemometricsScaler(with_mean=False, with_std=False)
        self.plsreg = ChemometricsPLS(ncomps=3, yscaler=y_scaler)
        self.plslog = ChemometricsPLS_Logistic(ncomps=3)

        # Generate the dummy matrix so we can run the pls regression objects in the same conditions as
        # the discriminant ones

        dummy_matrix = np.zeros((len(self.three_classdataset[0]), 3))
        for col in range(3):
            dummy_matrix[np.where(self.three_classdataset[1] == col), col] = 1
        self.dummy_y = dummy_matrix

    def test_single_y(self):

        self.plsreg.fit(self.twoclass_dataset[0], self.twoclass_dataset[1])
        self.plslog.fit(self.twoclass_dataset[0], self.twoclass_dataset[1])

        assert_array_equal(self.plsreg.scores_t, self.plslog.scores_t)
        assert_array_equal(self.plsreg.scores_u, self.plslog.scores_u)
        assert_array_equal(self.plsreg.rotations_cs, self.plslog.rotations_cs)
        assert_array_equal(self.plsreg.rotations_ws, self.plslog.rotations_ws)
        assert_array_equal(self.plsreg.weights_w, self.plslog.weights_w)
        assert_array_equal(self.plsreg.weights_c, self.plslog.weights_c)
        assert_array_equal(self.plsreg.loadings_p, self.plslog.loadings_p)
        assert_array_equal(self.plsreg.loadings_q, self.plslog.loadings_q)
        assert_array_equal(self.plsreg.beta_coeffs, self.plslog.beta_coeffs)
        assert_array_equal(self.plsreg.modelParameters['R2Y'], self.plslog.modelParameters['PLS']['R2Y'])
        assert_array_equal(self.plsreg.modelParameters['R2X'], self.plslog.modelParameters['PLS']['R2X'])
        assert_array_equal(self.plsreg.modelParameters['SSX'], self.plslog.modelParameters['PLS']['SSX'])
        assert_array_equal(self.plsreg.modelParameters['SSY'], self.plslog.modelParameters['PLS']['SSY'])
        assert_array_equal(self.plsreg.modelParameters['SSXcomp'], self.plslog.modelParameters['PLS']['SSXcomp'])
        assert_array_equal(self.plsreg.modelParameters['SSYcomp'], self.plslog.modelParameters['PLS']['SSYcomp'])

    def test_multi_y(self):
        self.plsreg.fit(self.three_classdataset[0], self.dummy_y)
        self.plslog.fit(self.three_classdataset[0], self.three_classdataset[1])

        assert_array_equal(self.plsreg.scores_t, self.plslog.scores_t)
        assert_array_equal(self.plsreg.scores_u, self.plslog.scores_u)
        assert_array_equal(self.plsreg.scores_u, self.plslog.scores_u)
        assert_array_equal(self.plsreg.loadings_p, self.plslog.loadings_p)
        assert_array_equal(self.plsreg.rotations_cs, self.plslog.rotations_cs)
        assert_array_equal(self.plsreg.rotations_ws, self.plslog.rotations_ws)
        assert_array_equal(self.plsreg.weights_w, self.plslog.weights_w)
        assert_array_equal(self.plsreg.weights_c, self.plslog.weights_c)
        assert_array_equal(self.plsreg.loadings_p, self.plslog.loadings_p)
        assert_array_equal(self.plsreg.loadings_q, self.plslog.loadings_q)
        assert_array_equal(self.plsreg.beta_coeffs, self.plslog.beta_coeffs)
        assert_array_equal(self.plsreg.modelParameters['R2Y'], self.plslog.modelParameters['PLS']['R2Y'])
        assert_array_equal(self.plsreg.modelParameters['R2X'], self.plslog.modelParameters['PLS']['R2X'])
        assert_array_equal(self.plsreg.modelParameters['SSX'], self.plslog.modelParameters['PLS']['SSX'])
        assert_array_equal(self.plsreg.modelParameters['SSY'], self.plslog.modelParameters['PLS']['SSY'])
        assert_array_equal(self.plsreg.modelParameters['SSXcomp'], self.plslog.modelParameters['PLS']['SSXcomp'])
        assert_array_equal(self.plsreg.modelParameters['SSYcomp'], self.plslog.modelParameters['PLS']['SSYcomp'])


if __name__ == '__main__':
    unittest.main()

