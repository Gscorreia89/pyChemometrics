import os
import unittest
import pandas as pds
import numpy as np
from numpy.testing import assert_allclose

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
            regression_problem = pds.read_csv(os.path.join(os.path.dirname(__file__), './test_data/regression.csv'))
            multiblock_regression_problem = pds.read_csv(os.path.join(os.path.dirname(__file__), './test_data/regression_multiblock.csv'))

        except (IOError, OSError) as ioerr:
            #os.system("python gen_synthetic_datasets.py")
            import tests.gen_synthetic_datasets
            regression_problem = pds.read_csv(os.path.join(os.path.dirname(__file__), './test_data/regression.csv'))
            multiblock_regression_problem = pds.read_csv(os.path.join(os.path.dirname(__file__), './test_data/regression_multiblock.csv'))

        finally:
            # Load expected values for a PLS regression against a Y vector
            self.expected_loadings_p = np.loadtxt(os.path.join(os.path.dirname(__file__), './test_data/pls_loadings_p.csv'), delimiter=',')
            self.expected_loadings_q = np.loadtxt(os.path.join(os.path.dirname(__file__), './test_data/pls_loadings_q.csv'), delimiter=',')[np.newaxis, :]
            self.expected_weights_w = np.loadtxt(os.path.join(os.path.dirname(__file__), './test_data/pls_weights_w.csv'), delimiter=',')
            self.expected_weights_c = np.loadtxt(os.path.join(os.path.dirname(__file__), './test_data/pls_weights_c.csv'), delimiter=',')[np.newaxis, :]
            self.expected_scores_t = np.loadtxt(os.path.join(os.path.dirname(__file__), './test_data/pls_scores_t.csv'), delimiter=',')
            self.expected_scores_u = np.loadtxt(os.path.join(os.path.dirname(__file__), './test_data/pls_scores_u.csv'), delimiter=',')
            self.expected_betacoefs = np.loadtxt(os.path.join(os.path.dirname(__file__), './test_data/pls_betas.csv'), delimiter=',')[:, np.newaxis]
            self.expected_vips = np.loadtxt(os.path.join(os.path.dirname(__file__), './test_data/pls_vip.csv'), delimiter=',')
            self.expected_dmodx = np.loadtxt(os.path.join(os.path.dirname(__file__), './test_data/pls_dmodx.csv'), delimiter=',')

            # Load expected values for a PLS regression model against a Y matrix
            #self.expected_loadings_p_yblock = np.loadtxt('./test_data/pls_reg_yblock_loadings_p.csv', delimiter=',')
            #self.expected_weights_w_yblock = np.loadtxt('./test_data/pls_reg_yblock_weights_w.csv', delimiter=',')
            #self.expected_scores_t_yblock = np.loadtxt('./test_data/pls_reg_yblock_scores_t.csv', delimiter=',')
            #self.expected_scores_u_yblock = np.loadtxt('./test_data/pls_reg_yblock_scores_u.csv', delimiter=',')
            #self.expected_weights_c_yblock = np.loadtxt('./test_data/pls_reg_yblock_weights_c.csv', delimiter=',')
            #self.expected_loadings_q_yblock = np.loadtxt('./test_data/pls_reg_yblock_loadings_q.csv', delimiter=',')
            #self.expected_betacoefs_yblock = np.loadtxt('./test_data/pls_reg_yblock_betacoefs.csv', delimiter=',')

            self.expected_modelParameters = {'R2Y': 0.99442967438303576, 'R2X': 0.022903901163376705,
                                         'SSYcomp': np.array([5.42418672,  1.20742786,  0.27851628]),
                                         'SSXcomp': np.array([9750.59475071, 9779.57249348, 9770.96098837])}

            self.expected_cvParameters = {'Q2Y': 0.069284226071602006, 'Q2X': -0.12391667143436425,
                                      'MeanR2X_Training': 0.025896665665079883,
                                      'MeanR2Y_Training': 0.99636477396947942,
                                      'StdevR2Y_Training': 0.00091660538957527582,
                                      'StdevR2X_Training': 0.0010098198504153058,
                                      'StdevR2X_Test': 0.02386260538832127,
                                      'StdevR2Y_Test': 0.25034195769401973,
                                      'MeanR2X_Test': -0.022542842216950101,
                                      'MeanR2Y_Test': 0.096991536519031446}

            self.expected_t2 = np.array([7.00212848,  6.63400492,  5.6325462])
            self.expected_outliers_t2 = np.array([5, 33])
            self.expected_outliers_dmodx = np.array([])

            self.expected_scores_t_par = np.loadtxt(os.path.join(os.path.dirname(__file__), './test_data/pls_scores_t_par.csv'), delimiter=',')
            self.expected_betas_par = np.loadtxt(os.path.join(os.path.dirname(__file__), './test_data/pls_betas_par.csv'), delimiter=',')[:, np.newaxis]
            self.expected_scores_t_mc = np.loadtxt(os.path.join(os.path.dirname(__file__), './test_data/pls_scores_t_mc.csv'), delimiter=',')
            self.expected_betas_mc = np.loadtxt(os.path.join(os.path.dirname(__file__), './test_data/pls_betas_mc.csv'), delimiter=',')[:, np.newaxis]

            self.expected_vip_mc = np.loadtxt(os.path.join(os.path.dirname(__file__), './test_data/pls_vip_mc.csv'), delimiter=',')
            self.expected_vip_par = np.loadtxt(os.path.join(os.path.dirname(__file__), './test_data/pls_vip_par.csv'), delimiter=',')

            # check this
            self.y = regression_problem.iloc[:, 0].values
            self.ymat = multiblock_regression_problem.values
            self.xmat = regression_problem.iloc[:, 1::].values
            self.xmat_multiy = multiblock_regression_problem.values

            self.expected_permutation = {}

        x_scaler = ChemometricsScaler(1)
        y_scaler = ChemometricsScaler(1)
        self.plsreg = ChemometricsPLS(ncomps=3, xscaler=x_scaler, yscaler=y_scaler)
        self.plsreg_multiblock = ChemometricsPLS(ncomps=3, xscaler=x_scaler, yscaler=y_scaler)

    def test_single_y(self):
        """

        :return:
        """
        self.plsreg.fit(self.xmat, self.y)

        # Test model coefficients , scores and goodness of fit
        assert_allclose(self.plsreg.loadings_p, self.expected_loadings_p)
        assert_allclose(self.plsreg.loadings_q, self.expected_loadings_q)
        assert_allclose(self.plsreg.weights_w, self.expected_weights_w)
        assert_allclose(self.plsreg.weights_c, self.expected_weights_c)
        assert_allclose(self.plsreg.scores_t, self.expected_scores_t)
        assert_allclose(self.plsreg.scores_u, self.expected_scores_u)
        assert_allclose(self.plsreg.beta_coeffs, self.expected_betacoefs)
        assert_allclose(self.plsreg.modelParameters['R2Y'], self.expected_modelParameters['R2Y'])
        assert_allclose(self.plsreg.modelParameters['R2X'], self.expected_modelParameters['R2X'])
        assert_allclose(self.plsreg.modelParameters['SSXcomp'], self.expected_modelParameters['SSXcomp'])
        assert_allclose(self.plsreg.modelParameters['SSYcomp'], self.expected_modelParameters['SSYcomp'])
        assert_allclose(self.plsreg.VIP(), self.expected_vips)

    #def test_multi_y(self):
    #    self.plsreg_multiblock.fit(self.xmat_multiy, self.ymat)
    #    # Assert equality of main model parameters
    #    assert_allclose(self.plsreg_multiblock.loadings_p, self.expected_loadings_p_yblock)
    #    assert_allclose(self.plsreg_multiblock.loadings_q, self.expected_loadings_q_yblock)
    #    assert_allclose(self.plsreg_multiblock.weights_w, self.expected_weights_w_yblock)
    #    assert_allclose(self.plsreg_multiblock.weights_c, self.expected_weights_c_yblock)
    #    assert_allclose(self.plsreg_multiblock.scores_t, self.expected_scores_t_yblock)
    #    assert_allclose(self.plsreg_multiblock.scores_u, self.expected_scores_u_yblock)
    #    assert_allclose(self.plsreg_multiblock.beta_coeffs, self.expected_betacoefs_yblock)
    #    assert_allclose(self.plsreg_multiblock.VIP(), self.expected_vipsw_yblock)
    #    assert_allclose(self.plsreg.modelParameters, self.expected_modelParameters)

    def test_scalers(self):
        """

        :return:
        """
        x_scaler_par = ChemometricsScaler(1 / 2)
        y_scaler_par = ChemometricsScaler(1 / 2)
        x_scaler_mc = ChemometricsScaler(0)
        y_scaler_mc = ChemometricsScaler(0)

        pareto_model = ChemometricsPLS(ncomps=3, xscaler=x_scaler_par, yscaler=y_scaler_par)
        pareto_model_multiy = ChemometricsPLS(ncomps=3, xscaler=x_scaler_par, yscaler=y_scaler_par)
        mc_model = ChemometricsPLS(ncomps=3, xscaler=x_scaler_mc, yscaler=y_scaler_mc)
        mc_model_multiy = ChemometricsPLS(ncomps=3, xscaler=x_scaler_mc, yscaler=y_scaler_mc)

        pareto_model.fit(self.xmat, self.y)
        pareto_model_multiy.fit(self.xmat_multiy, self.ymat)
        mc_model.fit(self.xmat, self.y)
        mc_model_multiy.fit(self.xmat_multiy, self.ymat)

        assert_allclose(pareto_model.scores_t, self.expected_scores_t_par)
        assert_allclose(pareto_model.beta_coeffs, self.expected_betas_par)
        assert_allclose(pareto_model.VIP(), self.expected_vip_par)

        #assert_allclose(pareto_model_multiy.scores_t, self.expected_scores_t_yblock_par)
        #assert_allclose(pareto_model_multiy.beta_coeffs, self.expected_betacoefs_yblock_par)

        assert_allclose(mc_model.scores_t, self.expected_scores_t_mc)
        assert_allclose(mc_model.beta_coeffs, self.expected_betas_mc)
        assert_allclose(mc_model.VIP(), self.expected_vip_mc)

        #assert_allclose(mc_model_multiy.scores_t, self.expected_scores_t_yblock_mc)
        #assert_allclose(mc_model_multiy.beta_coeffs, self.expected_betacoefs_yblock_mc)

    def test_cv_single_y(self):
        """

        :return:
        """
        # Fix the seed for the permutation test and cross_validation
        np.random.seed(0)
        self.plsreg.cross_validation(self.xmat, self.y)

        assert_allclose(self.plsreg.cvParameters['Q2Y'], self.expected_cvParameters['Q2Y'])
        assert_allclose(self.plsreg.cvParameters['Q2X'], self.expected_cvParameters['Q2X'])
        assert_allclose(self.plsreg.cvParameters['MeanR2X_Training'], self.expected_cvParameters['MeanR2X_Training'])
        assert_allclose(self.plsreg.cvParameters['MeanR2Y_Training'], self.expected_cvParameters['MeanR2Y_Training'])
        assert_allclose(self.plsreg.cvParameters['MeanR2X_Test'], self.expected_cvParameters['MeanR2X_Test'])
        assert_allclose(self.plsreg.cvParameters['MeanR2Y_Test'], self.expected_cvParameters['MeanR2Y_Test'])
        assert_allclose(self.plsreg.cvParameters['StdevR2X_Training'], self.expected_cvParameters['StdevR2X_Training'])
        assert_allclose(self.plsreg.cvParameters['StdevR2Y_Training'], self.expected_cvParameters['StdevR2Y_Training'])
        assert_allclose(self.plsreg.cvParameters['StdevR2X_Test'], self.expected_cvParameters['StdevR2X_Test'])
        assert_allclose(self.plsreg.cvParameters['StdevR2Y_Test'], self.expected_cvParameters['StdevR2Y_Test'])

    #def test_cv_multi_y(self):
    #    # Fix the seed for the permutation test and cross_validation
    #    np.random.seed(0)
    #    self.plsreg_multiy.cross_validation(self.xmat_multi, self.da_mat)
    #    assert_allclose(self.plsreg_multiblock.cvParameters, self.expected_cvParams_multi)

    def test_permutation(self):
        """

        :return:
        """
        self.plsreg.fit(self.xmat, self.y)
        # Fix the seed for the permutation test and cross_validation
        np.random.seed(0)
        self.plsreg.cross_validation(self.xmat, self.y)
        permutation_results = self.plsreg.permutation_test(self.xmat, self.da, nperms=5)
        assert_allclose(permutation_results[0], self.permutation_results)

    def test_hotellingt2(self):
        """

        :return:
        """
        self.plsreg.fit(self.xmat, self.y)
        t2 = self.plsreg.hotelling_T2(comps=None)
        assert_allclose(t2, self.expected_t2)

    def test_dmodx(self):
        """

        :return:
        """
        self.plsreg.fit(self.xmat, self.y)
        dmodx = self.plsreg.dmodx(self.xmat)
        assert_allclose(dmodx, self.expected_dmodx)

    def test_outliers(self):
        """

        :return:
        """
        self.plsreg.fit(self.xmat, self.y)
        outliers_t2 = self.plsreg.outlier(self.xmat)
        outliers_dmodx = self.plsreg.outlier(self.xmat, measure='DmodX')
        assert_allclose(outliers_t2, self.expected_outliers_t2)
        assert_allclose(outliers_dmodx, self.expected_outliers_dmodx)


if __name__ == '__main__':
    unittest.main()


