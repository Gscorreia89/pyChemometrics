import unittest
from numpy.testing import assert_allclose
import numpy as np
import pandas as pds
import os
from pyChemometrics import ChemometricsScaler

"""

Tests for the ChemometricsScaler object

"""


class TestScalerObject(unittest.TestCase):
    """

    Use a made up dataset

    """

    def setUp(self):
        try:
            regression_problem = pds.read_csv(os.path.join(os.path.dirname(__file__), './test_data/regression.csv'))
            multiblock_regression_problem = pds.read_csv(os.path.join(os.path.dirname(__file__), './test_data/regression_multiblock.csv'))
        except (IOError, OSError) as ioerr:
            #os.system("python gen_synthetic_datasets.py")
            import tests.gen_synthetic_datasets

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

    def scale_back_vector(self):
        """
        Check back_transformations of the scaler (inverse_transform)
        """

    def scale_back_matrix(self):
        """
        Check back_transformations of the scaler (inverse_transform)
        """


if __name__ == '__main__':
    unittest.main()
