import os
import unittest
import pandas as pds
import numpy as np

from pyChemometrics import ChemometricsScaler, ChemometricsPLS

"""

Suite of tests to assess correctness of the PLS Regression object.

"""


class test_pls_da(unittest.TestCase):
    """

    Verify agreement of PLS algorithms under different objects and conditions

    """

    def setUp(self):

        try:
            two_class = pds.read_csv('./test_data/da.csv')
            multiclass = pds.read_csv('./test_data/da_multi.csv')

        except OSError as exp:
            os.system("python gen_synthetic_datasets.py")
            two_class = pds.read_csv('./test_data/da.csv')
            multiclass = pds.read_csv('./test_data/da_multi.csv')

        finally:
            # Load expected values for a PLS regression against a Y vector
            self.expected_loadings_p = pds.read_csv('./test_data/pls_da_loadings_p.csv')
            self.expected_weights_w = pds.read_csv('./test_data/pls_da_weights_w.csv')
            self.expected_scores_t = pds.read_csv('./test_data/pls_da_scores_t.csv')
            self.expected_scores_u = pds.read_csv('./test_data/pls_da_scores_u.csv')
            self.expected_weights_c = pds.read_csv('./test_data/pls_da_weights_c.csv')
            self.expected_loadings_q = pds.read_csv('./test_data/pls_da_loadings_q.csv')
            self.expected_betacoefs = pds.read_csv('./test_data/pls_da_betacoefs.csv')
            self.expected_vipsw = pds.read_csv('./test_data/pls_da_vipsw.csv')

            # Load expected values for a PLS regression model against a Y matrix
            self.expected_loadings_p_multiclass = pds.read_csv('./test_data/pls_da_multi_loadings_p.csv')
            self.expected_weights_w_multiclass = pds.read_csv('./test_data/pls_da_multi_weights_w.csv')
            self.expected_scores_t_multiclass = pds.read_csv('./test_data/pls_da_multi_scores_t.csv')
            self.expected_scores_u_multiclass = pds.read_csv('./test_data/pls_da_multi_scores_u.csv')
            self.expected_weights_c_multiclass = pds.read_csv('./test_data/pls_da_multi_weights_c.csv')
            self.expected_loadings_q_multiclass = pds.read_csv('./test_data/pls_da_multi_loadings_q.csv')
            self.expected_betacoefs_multiclass = pds.read_csv('./test_data/pls_da_multi_betacoefs.csv')
            self.expected_vipsw_multiclass = pds.read_csv('./test_data/pls_da_multi_vipsw.csv')

            # check this
            self.da_yvector = two_class.values
            self.da_ymat = multiclass.values
            self.xmat = two_class.values
            self.xmat_multi = multiclass.values

        x_scaler = ChemometricsScaler(1)
        y_scaler = ChemometricsScaler(1, with_mean=True, with_std=False)
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

        self.assertAlmostEqual()

    def test_permutation(self):
        pass

    def hotellingt2(self):
        pass


if __name__ == '__main__':
    unittest.main()


