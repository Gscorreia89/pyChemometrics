import unittest

from numpy.testing import assert_allclose
import numpy as np
from pyChemometrics import ChemometricsScaler

"""

Tests for the ChemometricsScaler object

"""


class TestScalerObject(unittest.TestCase):
    """

    Use a made up dataset on the fly...

    """

    def setUp(self):
        self.scaler = ChemometricsScaler()

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

        assert_allclose()

    def scale_back(self):
        """
        Check back_transformations of the scaler (inverse_transform)
        """


if __name__ == '__main__':
    unittest.main()
