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
            two_class = pds.read_csv(os.path.join(os.path.dirname(__file__), './test_data/classification_twoclass.csv'))
            multiclass = pds.read_csv(os.path.join(os.path.dirname(__file__), './test_data/classification_multiclass.csv'))

        except OSError as exp:
            os.system("python gen_synthetic_datasets.py")
            two_class = pds.read_csv(os.path.join(os.path.dirname(__file__), './test_data/classification_twoclass.csv'))
            multiclass = pds.read_csv(os.path.join(os.path.dirname(__file__), './test_data/classification_multiclass.csv'))

        finally:
            # Load expected values for a PLS da with 2 classes
            self.expected_cvParams = pds.read_csv(os.path.join(os.path.dirname(__file__), './test_data/pls_da_cvoarams.csv'))

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


if __name__ == '__main__':
    unittest.main()