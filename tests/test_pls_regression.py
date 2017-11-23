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
        x_scaler_par = ChemometricsScaler(1 / 2)
        y_scaler_par = ChemometricsScaler(1 / 2)
        x_scaler_mc = ChemometricsScaler(0)
        y_scaler_mc = ChemometricsScaler(0)

        pareto_model = ChemometricsPLS(n_comps=3, xscaler=x_scaler_par, y_scaler=y_scaler_par)
        pareto_model_multiy = ChemometricsPLS(n_comps=3, xscaler=x_scaler_par, y_scaler=y_scaler_par)
        mc_model = ChemometricsPLS(n_comps=3, xscaler=x_scaler_mc, yscaler=y_scaler_mc)
        mc_model_multiy = ChemometricsPLS(n_comps=3, xscaler=x_scaler_mc, yscaler=y_scaler_mc)

        pareto_model.fit(self.xmat, self.da)
        pareto_model_multiy.fit(self.xmat_multi, self.da_mat)
        mc_model.fit(self.xmat, self.da)
        mc_model_multiy.fit(self.xmat_multi, self.da_mat)

        self.assertAlmostEqual(self.plsda_multiy.loadings_p, self.expected_loadings_p_yblock)
        self.assertAlmostEqual(self.plsda_multiy.loadings_q, self.expected_loadings_q_yblock)
        self.assertAlmostEqual(self.plsda_multiy.weights_w, self.expected_weights_w_yblock)
        self.assertAlmostEqual(self.plsda_multiy.weights_c, self.expected_weights_c_yblock)
        self.assertAlmostEqual(self.plsda_multiy.scores_t, self.expected_scores_t_yblock)
        self.assertAlmostEqual(self.plsda_multiy.scores_u, self.expected_scores_u_yblock)
        self.assertAlmostEqual(self.plsda_multiy.beta_coeffs, self.expected_betacoefs_yblock)
        self.assertAlmostEqual(self.plsda_multiy.VIP(), self.expected_vipsw_yblock)

    def test_cv(self):
        # Fix the seed for the permutation test and cross_validation
        np.random.seed(0)
        self.plsda.cross_validation(self.xmat, self.da)
        self.plsda_multiy.cross_validation(self.xmat_multi, self.da_mat)
        self.assertAlmostEqual(self.plsda.cvParameters, self.expected_cvParams)
        self.assertAlmostEqual(self.plsda_multiy.cvParameters, self.expected_cvParams_multi)

    def test_permutation(self):
        # Fix the seed for the permutation test and cross_validation
        np.random.seed(0)
        permutation_results = self.plsda.permutation_test(self.xmat, self.da, nperms=5)
        self.assertAlmostEqual()

    def hotellingt2(self):
        t2 = self.plsda.hotelling_T2(comps=None)
        t2_multi = self.plsda_multiy.hotelling_T2(comps=None)
        self.assertAlmostEqual(t2, self.expected_t2)
        self.assertAlmostEqual(t2_multi, self.expected_t2_multi)

    def Dmodx(self):
        dmodx = self.plsda.dmodx(self.xmat)
        dmodx_multi = self.plsda_multiy.dmodx(self.xmat_multi)
        self.assertAlmostEqual(dmodx, self.expected_dmodx)
        self.assertAlmostEqual(dmodx_multi, self.expected_dmodx_multi)

    def test_outliers(self):
        outliers_t2 = self.pcamodel.outlier(self.xmat)
        outliers_dmodx = self.pcamodel.outlier(self.xmat)
        self.assertAlmostEqual(outliers_t2, self.expected_outliers_t2)
        self.assertAlmostEqual(outliers_dmodx, self.expected_outliers_dmodx)

        outliers_t2_multi = self.plsda_multiy.outlier(self.xmat_multi)
        outliers_dmodx_multi = self.plsda_multiy.outlier(self.xmat_multi)
        self.assertAlmostEqual(outliers_dmodx_multi, self.expected_outliers_dmodx_multi)
        self.assertAlmostEqual(outliers_t2_multi, self.expected_outliers_t2_multi)



if __name__ == '__main__':
    unittest.main()


