import unittest
import numpy as np
from sklearn.datasets import make_classification
from pyChemometrics import ChemometricsScaler, ChemometricsPLS, ChemometricsPLS_Logistic

"""
Suite of tests to ensure that all PLS objects are consistent among each other: 
For example, the ChemometricsPLS object in Regression mode needs to give the same results (coefficients, 
loadings, scores R2s, etc) as the PLS-Classifier ojects (ChemometricPLS_Logistic and ChemometricsPLS_QDA), 
provided we account for the differences in data input and vector to dummy matrix conversion.
"""


class test_ChemometricsScaler(unittest.TestCase):
    """

    Use a made up dataset on the fly...

    """

    def setUp(self):
        self.ples = 1

    def test_scaleVector(self):
        """
        Check that scaling works with arbitrary value between 0 and 1 as expected on a single vector.
        """
        # Modify here - pull a random number between 0 and 1
        scaling_factor = 1
        scaledData = self.scaler.transform()

        expected_scaledData = (self.dataset - np.mean(self.dataset)) / (np.std(self.dataset)) ** scaling_factor

    def test_scaleMatrix(self):
        """
        Check that scaling works with arbitrary value between 0 and 1 as expected on a matrix of m samples by n features.
        """

        np.testing.assert_array_almost_equal()

    def scale_back(self):
        """
        Check back_transformations of the scaler (inverse_transform)
        """


if __name__ == '__main__':
    unittest.main()
