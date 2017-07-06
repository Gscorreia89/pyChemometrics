import unittest

import numpy as np
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
        self.plsreg = ChemometricsPLS(n_comps=3, yscaler=y_scaler)
        self.plslog = ChemometricsPLS_Logistic(n_comps=3)

        # Generate the dummy matrix so we can run the pls regression objects in the same conditions as
        # the discriminant ones

        dummy_matrix = np.zeros((len(self.three_classdataset[0]), 3))
        for col in range(3):
            dummy_matrix[np.where(self.three_classdataset[1] == col), col] = 1
        self.dummy_y = dummy_matrix

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

