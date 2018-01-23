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

        except (IOError, OSError) as ioerr:
            import tests.gen_synthetic_datasets
            regression_problem = pds.read_csv(os.path.join(os.path.dirname(__file__), './test_data/regression.csv'))

        self.mc_scaler = ChemometricsScaler(0)
        self.uv_scaler = ChemometricsScaler(1)
        self.par_scaler = ChemometricsScaler(1 / 2)
        self.y = regression_problem.values[:, 0][np.newaxis].T
        self.xmat = regression_problem.values[:, 1:4]

        self.xmat_mc = np.loadtxt(os.path.join(os.path.dirname(__file__), './test_data/scaler_xmat_mc.csv'),
                                  delimiter=',')
        self.xmat_uv = np.loadtxt(os.path.join(os.path.dirname(__file__), './test_data/scaler_xmat_uv.csv'),
                                  delimiter=',')
        self.xmat_par = np.loadtxt(os.path.join(os.path.dirname(__file__), './test_data/scaler_xmat_par.csv'),
                                   delimiter=',')

        self.y_mc = np.loadtxt(os.path.join(os.path.dirname(__file__), './test_data/scaler_y_mc.csv'), delimiter=',')
        self.y_uv = np.loadtxt(os.path.join(os.path.dirname(__file__), './test_data/scaler_y_uv.csv'), delimiter=',')
        self.y_par = np.loadtxt(os.path.join(os.path.dirname(__file__), './test_data/scaler_y_par.csv'), delimiter=',')

    def test_scaleVector(self):
        """
        Check that scaling works with arbitrary value between 0 and 1 as expected on a single vector.
        """

        assert_allclose(self.mc_scaler.fit_transform(self.y).squeeze(), self.y_mc)
        assert_allclose(self.uv_scaler.fit_transform(self.y).squeeze(), self.y_uv)
        assert_allclose(self.par_scaler.fit_transform(self.y).squeeze(), self.y_par)

    def test_scaleMatrix(self):
        """
        Check that scaling works with arbitrary value between 0 and 1 as expected on a matrix of m samples by n features.
        """

        assert_allclose(self.mc_scaler.fit_transform(self.xmat), self.xmat_mc)
        assert_allclose(self.uv_scaler.fit_transform(self.xmat), self.xmat_uv)
        assert_allclose(self.par_scaler.fit_transform(self.xmat), self.xmat_par)

    def test_inverseTransformVector(self):
        """
        Test inverse transform of a vector
        """

        self.mc_scaler.fit(self.y)
        self.uv_scaler.fit(self.y)
        self.par_scaler.fit(self.y)

        assert_allclose(self.mc_scaler.inverse_transform(self.y_mc), self.y.squeeze())
        assert_allclose(self.uv_scaler.inverse_transform(self.y_uv), self.y.squeeze())
        assert_allclose(self.par_scaler.inverse_transform(self.y_par), self.y.squeeze())

    def test_inverseTransformMatrix(self):
        """
        Test inverse transform of a matrix
        """

        self.mc_scaler.fit(self.xmat)
        self.uv_scaler.fit(self.xmat)
        self.par_scaler.fit(self.xmat)

        assert_allclose(self.mc_scaler.inverse_transform(self.xmat_mc), self.xmat)
        assert_allclose(self.uv_scaler.inverse_transform(self.xmat_uv), self.xmat)
        assert_allclose(self.par_scaler.inverse_transform(self.xmat_par), self.xmat)


if __name__ == '__main__':
    unittest.main()
