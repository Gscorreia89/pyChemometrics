import os
import unittest
import pandas as pds
import numpy as np

from pyChemometrics import ChemometricsScaler, ChemometricsPLSDA

"""

Suite of tests to assess coherence and functionality of the PLS-DA object.

"""


class TestPLSDA(unittest.TestCase):
    """

    Verify agreement of PLS algorithms under different objects and conditions

    """

    def setUp(self):

        try:
            two_class = pds.read_csv('./test_data/classification_twoclass.csv')
            multiclass = pds.read_csv('./test_data/classification_multiclass.csv')

        except OSError as exp:
            os.system("python gen_synthetic_datasets.py")
            two_class = pds.read_csv('./test_data/classification_twoclass.csv')
            multiclass = pds.read_csv('./test_data/classification_multiclass.csv')

        finally:
            # Load expected values for a PLS da with 2 classes
            self.expected_loadings_p = pds.read_csv('./test_data/pls_da_loadings_p.csv')
            self.expected_weights_w = pds.read_csv('./test_data/pls_da_weights_w.csv')
            self.expected_scores_t = pds.read_csv('./test_data/pls_da_scores_t.csv')
            self.expected_scores_u = pds.read_csv('./test_data/pls_da_scores_u.csv')
            self.expected_weights_c = pds.read_csv('./test_data/pls_da_weights_c.csv')
            self.expected_loadings_q = pds.read_csv('./test_data/pls_da_loadings_q.csv')
            self.expected_betacoefs = pds.read_csv('./test_data/pls_da_betacoefs.csv')
            self.expected_vipsw = pds.read_csv('./test_data/pls_da_vipsw.csv')
            self.expected_cvParams = pds.read_csv('./test_data/pls_da_cvoarams.csv')

            # Make a single file out of this
            self.expected_t2 = pds.read_csv('./test_data/pls_da_cvParams.csv')
            self.expected_dmodx = pds.read_csv('./test_data/pls_da_cvParams.csv')
            self.expected_outliers_dmodx = pds.read_csv('./test_data/pls_da_cvParams.csv')
            self.expected_outliers_t2 = pds.read_csv('./test_data/pls_da_cvparams.csv')

            # Load expected values for a PLS da model with multiple classes
            self.expected_loadings_p_multiclass = pds.read_csv('./test_data/pls_da_multi_loadings_p.csv')
            self.expected_weights_w_multiclass = pds.read_csv('./test_data/pls_da_multi_weights_w.csv')
            self.expected_scores_t_multiclass = pds.read_csv('./test_data/pls_da_multi_scores_t.csv')
            self.expected_scores_u_multiclass = pds.read_csv('./test_data/pls_da_multi_scores_u.csv')
            self.expected_weights_c_multiclass = pds.read_csv('./test_data/pls_da_multi_weights_c.csv')
            self.expected_loadings_q_multiclass = pds.read_csv('./test_data/pls_da_multi_loadings_q.csv')
            self.expected_betacoefs_multiclass = pds.read_csv('./test_data/pls_da_multi_betacoefs.csv')
            self.expected_vipsw_multiclass = pds.read_csv('./test_data/pls_da_multi_vipsw.csv')

            # Make a single file out of this
            self.expected_t2_multi = pds.read_csv('./test_data/pls_da_multi_cvParams.csv')
            self.expected_dmodx_multi = pds.read_csv('./test_data/pls_da_multi_cvParams.csv')
            self.expected_outliers_dmodx_multi = pds.read_csv('./test_data/pls_da_multi_cvParams.csv')
            self.expected_outliers_t2_multi = pds.read_csv('./test_data/pls_da_multi_cvParams.csv')

            # check this
            self.da_mat = multiclass['Class_Vector'].values
            self.da = two_class['Class'].values
            self.xmat_multi = multiclass.iloc[:, 5::].values
            self.xmat = two_class.iloc[:, 1::].values

        x_scaler = ChemometricsScaler(1)
        y_scaler = ChemometricsScaler(1, with_mean=True, with_std=False)
        self.plsda = ChemometricsPLSDA(n_comps=3, xscaler=x_scaler, y_scaler=y_scaler)
        self.plsda_multiy = ChemometricsPLSDA(n_comps=3, xscaler=x_scaler, y_scaler=y_scaler)

    def test_single_y(self):
        self.plsda.fit(self.xmat, self.da)
        self.assertAlmostEqual(self.plsda.loadings_p, self.expected_loadings_p)
        self.assertAlmostEqual(self.plsda.loadings_q, self.expected_loadings_q)
        self.assertAlmostEqual(self.plsda.weights_w, self.expected_weights_w)
        self.assertAlmostEqual(self.plsda.weights_c, self.expected_weights_c)
        self.assertAlmostEqual(self.plsda.scores_t, self.expected_scores_t)
        self.assertAlmostEqual(self.plsda.scores_u, self.expected_scores_u)
        self.assertAlmostEqual(self.plsda.beta_coeffs, self.expected_betacoefs)
        self.assertAlmostEqual(self.plsda.VIP(), self.expected_vipsw)

    def test_multi_y(self):
        self.plsda_multiy.fit(self.xmat_multi, self.da_mat)
        self.assertAlmostEqual(self.plsda_multiy.loadings_p, self.expected_loadings_p_yblock)
        self.assertAlmostEqual(self.plsda_multiy.loadings_q, self.expected_loadings_q_yblock)
        self.assertAlmostEqual(self.plsda_multiy.weights_w, self.expected_weights_w_yblock)
        self.assertAlmostEqual(self.plsda_multiy.weights_c, self.expected_weights_c_yblock)
        self.assertAlmostEqual(self.plsda_multiy.scores_t, self.expected_scores_t_yblock)
        self.assertAlmostEqual(self.plsda_multiy.scores_u, self.expected_scores_u_yblock)
        self.assertAlmostEqual(self.plsda_multiy.beta_coeffs, self.expected_betacoefs_yblock)
        self.assertAlmostEqual(self.plsda_multiy.VIP(), self.expected_vipsw_yblock)

    def test_scalers(self):
        x_scaler_par = ChemometricsScaler(1 / 2)
        y_scaler_par = ChemometricsScaler(1 / 2)
        x_scaler_mc = ChemometricsScaler(0)
        y_scaler_mc = ChemometricsScaler(0)

        pareto_model = ChemometricsPLSDA(n_comps=3, xscaler=x_scaler_par, y_scaler=y_scaler_par)
        pareto_model_multiy = ChemometricsPLSDA(n_comps=3, xscaler=x_scaler_par, y_scaler=y_scaler_par)
        mc_model = ChemometricsPLSDA(n_comps=3, xscaler=x_scaler_mc, yscaler=y_scaler_mc)
        mc_model_multiy = ChemometricsPLSDA(n_comps=3, xscaler=x_scaler_mc, yscaler=y_scaler_mc)

        pareto_model.fit(self.xmat, self.da)
        pareto_model_multiy.fit(self.xmat_multi, self.da_mat)
        mc_model.fit(self.xmat, self.da)
        mc_model_multiy.fit(self.xmat_multi, self.da_mat)

        self.assertAlmostEqual(pareto_model.loadings_p, self.expected_loadings_p_par)
        self.assertAlmostEqual(pareto_model.loadings_q, self.expected_loadings_q_par)
        self.assertAlmostEqual(pareto_model.weights_w, self.expected_weights_w_par)
        self.assertAlmostEqual(pareto_model.weights_c, self.expected_weights_c_par)
        self.assertAlmostEqual(pareto_model.scores_t, self.expected_scores_t_par)
        self.assertAlmostEqual(pareto_model.scores_u, self.expected_scores_u_par)
        self.assertAlmostEqual(pareto_model.beta_coeffs, self.expected_betacoefs_par)
        self.assertAlmostEqual(pareto_model.VIP(), self.expected_vipsw_par)

        self.assertAlmostEqual(pareto_model_multiy.loadings_p, self.expected_loadings_p_yblock_par)
        self.assertAlmostEqual(pareto_model_multiy.loadings_q, self.expected_loadings_q_yblock_par)
        self.assertAlmostEqual(pareto_model_multiy.weights_w, self.expected_weights_w_yblock_par)
        self.assertAlmostEqual(pareto_model_multiy.weights_c, self.expected_weights_c_yblock_par)
        self.assertAlmostEqual(pareto_model_multiy.scores_t, self.expected_scores_t_yblock_par)
        self.assertAlmostEqual(pareto_model_multiy.scores_u, self.expected_scores_u_yblock_par)
        self.assertAlmostEqual(pareto_model_multiy.beta_coeffs, self.expected_betacoefs_yblock_par)
        self.assertAlmostEqual(pareto_model_multiy.VIP(), self.expected_vipsw_yblock_par)

        self.assertAlmostEqual(mc_model.loadings_p, self.expected_loadings_p_mc)
        self.assertAlmostEqual(mc_model.loadings_q, self.expected_loadings_q_mc)
        self.assertAlmostEqual(mc_model.weights_w, self.expected_weights_w_mc)
        self.assertAlmostEqual(mc_model.weights_c, self.expected_weights_c_mc)
        self.assertAlmostEqual(mc_model.scores_t, self.expected_scores_t_mc)
        self.assertAlmostEqual(mc_model.scores_u, self.expected_scores_mc)
        self.assertAlmostEqual(mc_model.beta_coeffs, self.expected_betacoefs_mc)
        self.assertAlmostEqual(mc_model.VIP(), self.expected_vipsw_mc)

        self.assertAlmostEqual(mc_model_multiy.loadings_p, self.expected_loadings_p_yblock_mc)
        self.assertAlmostEqual(mc_model_multiy.loadings_q, self.expected_loadings_q_yblock_mc)
        self.assertAlmostEqual(mc_model_multiy.weights_w, self.expected_weights_w_yblock_mc)
        self.assertAlmostEqual(mc_model_multiy.weights_c, self.expected_weights_c_yblock_mc)
        self.assertAlmostEqual(mc_model_multiy.scores_t, self.expected_scores_t_yblock_mc)
        self.assertAlmostEqual(mc_model_multiy.scores_u, self.expected_scores_u_yblock_mc)
        self.assertAlmostEqual(mc_model_multiy.beta_coeffs, self.expected_betacoefs_yblock_mc)
        self.assertAlmostEqual(mc_model_multiy.VIP(), self.expected_vipsw_yblock_mc)

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
        self.plsda.cross_validation(self.xmat, self.da)
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