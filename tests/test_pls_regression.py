import os
import unittest
import pandas as pds
import numpy as np

from pyChemometrics import ChemometricsScaler, ChemometricsPLS

"""

Suite of tests to assess coherence and functionality of the PLS regression object.

"""


class TestPLS(unittest.TestCase):
    """

    Verify agreement of PLS algorithms under different objects and conditions

    """

    def setUp(self):

        try:
            regression_problem = pds.read_csv('./test_data/regression.csv')
            multiblock_regression_problem = pds.read_csv('./test_data/regression_multiblock.csv')
        except (IOError, OSError) as ioerr:
            os.system("python gen_synthetic_datasets.py")
            regression_problem = pds.read_csv('./test_data/regression.csv')
            multiblock_regression_problem = pds.read_csv('./test_data/regression_multiblock.csv')
        finally:
            # Load expected values for a PLS regression against a Y vector
            self.expected_loadings_p = pds.read_csv('./test_data/pls_reg_loadings_p.csv')
            self.expected_weights_w = pds.read_csv('./test_data/pls_reg_weights_w.csv')
            self.expected_scores_t = pds.read_csv('./test_data/pls_reg_scores_t.csv')
            self.expected_scores_u = pds.read_csv('./test_data/pls_reg_scores_u.csv')
            self.expected_weights_c = pds.read_csv('./test_data/pls_reg_weights_c.csv')
            self.expected_loadings_q = pds.read_csv('./test_data/pls_reg_loadings_q.csv')
            self.expected_betacoefs = pds.read_csv('./test_data/pls_reg_betacoefs.csv')
            self.expected_vipsw = pds.read_csv('./test_data/pls_reg_vipsw.csv')

            # Load expected values for a PLS regression model against a Y matrix
            self.expected_loadings_p_yblock = pds.read_csv('./test_data/pls_reg_loadings_p.csv')
            self.expected_weights_w_yblock = pds.read_csv('./test_data/pls_reg_weights_w.csv')
            self.expected_scores_t_yblock = pds.read_csv('./test_data/pls_reg_scores_t.csv')
            self.expected_scores_u_yblock = pds.read_csv('./test_data/pls_reg_scores_u.csv')
            self.expected_weights_c_yblock = pds.read_csv('./test_data/pls_reg_weights_c.csv')
            self.expected_loadings_q_yblock = pds.read_csv('./test_data/pls_reg_loadings_q.csv')
            self.expected_betacoefs_yblock = pds.read_csv('./test_data/pls_reg_betacoefs.csv')
            self.expected_vipsw_yblock = pds.read_csv('./test_data/pls_reg_vipsw.csv')

            # check this
            self.regression_yvector = regression_problem.values
            self.regression_ymat = regression_problem.values
            self.regression_xmat = regression_problem.values

        x_scaler = ChemometricsScaler(1)
        y_scaler = ChemometricsScaler(1)
        self.plsreg = ChemometricsPLS(n_comps=3, xscaler=x_scaler, y_scaler=y_scaler)
        self.plsreg_multiblock = ChemometricsPLS(n_comps=3, xscaler=x_scaler, y_scaler=y_scaler)

    def test_single_y(self):
        self.plsreg.fit(self.regression[0], self.twoclass_dataset([1]))
        self.plsreg_multiblock.fit(self.regression_xmat, self.regression_ymat)
        self.assertAlmostEqual(self.plsreg.loadings_p, self.expected_loadings_p_yblock)
        self.assertAlmostEqual(self.plsreg.loadings_q, self.expected_loadings_q_yblock)
        self.assertAlmostEqual(self.plsreg.weights_w, self.expected_weights_w_yblock)
        self.assertAlmostEqual(self.plsreg.weights_c, self.expected_weights_c_yblock)
        self.assertAlmostEqual(self.plsreg.scores_t, self.expected_scores_t_yblock)
        self.assertAlmostEqual(self.plsreg.scores_u, self.expected_scores_u_yblock)
        self.assertAlmostEqual(self.plsreg_mul.beta_coeffs, self.expected_betacoefs_yblock)
        self.assertAlmostEqual(self.plsreg_multiblock.VIP(), self.expected_vipsw_yblock)

    def test_multi_y(self):
        self.plsreg_multiblock.fit(self.regression_xmat, self.regression_ymat)
        # Assert equality of main model parameters
        self.assertAlmostEqual(self.plsreg_multiblock.loadings_p, self.expected_loadings_p_yblock)
        self.assertAlmostEqual(self.plsreg_multiblock.loadings_q, self.expected_loadings_q_yblock)
        self.assertAlmostEqual(self.plsreg_multiblock.weights_w, self.expected_weights_w_yblock)
        self.assertAlmostEqual(self.plsreg_multiblock.weights_c, self.expected_weights_c_yblock)
        self.assertAlmostEqual(self.plsreg_multiblock.scores_t, self.expected_scores_t_yblock)
        self.assertAlmostEqual(self.plsreg_multiblock.scores_u, self.expected_scores_u_yblock)
        self.assertAlmostEqual(self.plsreg_multiblock.beta_coeffs, self.expected_betacoefs_yblock)
        self.assertAlmostEqual(self.plsreg_multiblock.VIP(), self.expected_vipsw_yblock)
        self.assertAlmostEqual(self.plsreg_multiblock.VIP(), self.expected_vipsw_yblock)

    def test_scalers(self):
        # Same stuff with pareto, and mc in Y and X for a change
        pass

    def test_cv(self):
        self.plsreg.cross_validation(self.xmat, self.regres)
        self.plsreg_multiblock.cross_validation(self.xmat)
        self.plsreg_multiblock
        self.assertAlmostEqual()

    def test_permutation(self):
        pass

    def hotellingt2(self):
        pass


if __name__ == '__main__':
    unittest.main()


