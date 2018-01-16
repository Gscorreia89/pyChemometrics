import unittest
import numpy as np
from numpy.testing import assert_allclose
import pandas as pds
import os

from pyChemometrics import ChemometricsScaler, ChemometricsPLS, ChemometricsPLSDA

"""

Suite of tests to ensure that all PLS objects are consistent among each other. 
For example, the ChemometricsPLS object for regression analysis needs to give the same results (coefficients, 
loadings, scores R2s, etc) as the PLS component of the PLS Classifier objects, 
provided that we account for the differences in data input and class vector do dummy conversions.

"""


class TestPLSObjectConsistency(unittest.TestCase):
    """

    Verify agreement of the PLS regression component between ChemometricsPLS and Discriminant analysis versions.

    """

    def setUp(self):

        try:
            multiclass = pds.read_csv(os.path.join(os.path.dirname(__file__), './test_data/classification_multiclass.csv'))
            twoclass = pds.read_csv(os.path.join(os.path.dirname(__file__), './test_data/classification_twoclass.csv'))
        except OSError as exp:
            os.system("python gen_synthetic_datasets.py")
            multiclass = pds.read_csv(os.path.join(os.path.dirname(__file__), './test_data/classification_multiclass.csv'))
            twoclass = pds.read_csv(os.path.join(os.path.dirname(__file__), './test_data/classification_twoclass.csv'))
        finally:
            # check this
            self.da_mat = multiclass['Class_Vector'].values
            self.da = twoclass['Class'].values
            self.xmat_multi = multiclass.iloc[:, 5::].values
            self.xmat = twoclass.iloc[:, 1::].values

        # Set up the same scalers
        y_scaler = ChemometricsScaler(0, with_std=False, with_mean=True)
        self.plsreg = ChemometricsPLS(ncomps=3, yscaler=y_scaler)
        self.plsda = ChemometricsPLSDA(ncomps=3)

        # Generate the dummy matrix so we can run the pls regression objects in the same conditions as
        # the discriminant ones
        self.dummy_y = pds.get_dummies(self.da_mat).values

    def test_single_y(self):

        self.plsreg.fit(self.xmat, self.da)
        self.plsda.fit(self.xmat, self.da)

        assert_allclose(self.plsreg.scores_t, self.plsda.scores_t)
        assert_allclose(self.plsreg.scores_u, self.plsda.scores_u)
        assert_allclose(self.plsreg.rotations_cs, self.plsda.rotations_cs)
        assert_allclose(self.plsreg.rotations_ws, self.plsda.rotations_ws)
        assert_allclose(self.plsreg.weights_w, self.plsda.weights_w)
        assert_allclose(self.plsreg.weights_c, self.plsda.weights_c)
        assert_allclose(self.plsreg.loadings_p, self.plsda.loadings_p)
        assert_allclose(self.plsreg.loadings_q, self.plsda.loadings_q)
        assert_allclose(self.plsreg.beta_coeffs, self.plsda.beta_coeffs)
        assert_allclose(self.plsreg.modelParameters['R2Y'], self.plsda.modelParameters['PLS']['R2Y'])
        assert_allclose(self.plsreg.modelParameters['R2X'], self.plsda.modelParameters['PLS']['R2X'])
        assert_allclose(self.plsreg.modelParameters['SSX'], self.plsda.modelParameters['PLS']['SSX'])
        assert_allclose(self.plsreg.modelParameters['SSY'], self.plsda.modelParameters['PLS']['SSY'])
        assert_allclose(self.plsreg.modelParameters['SSXcomp'], self.plsda.modelParameters['PLS']['SSXcomp'])
        assert_allclose(self.plsreg.modelParameters['SSYcomp'], self.plsda.modelParameters['PLS']['SSYcomp'])

    def test_multi_y(self):

        self.plsreg.fit(self.xmat_multi, self.dummy_y)
        self.plsda.fit(self.xmat_multi, self.da_mat)
        
        assert_allclose(self.plsreg.scores_t, self.plsda.scores_t)
        assert_allclose(self.plsreg.scores_u, self.plsda.scores_u)
        assert_allclose(self.plsreg.rotations_cs, self.plsda.rotations_cs)
        assert_allclose(self.plsreg.rotations_ws, self.plsda.rotations_ws)
        assert_allclose(self.plsreg.weights_w, self.plsda.weights_w)
        assert_allclose(self.plsreg.weights_c, self.plsda.weights_c)
        assert_allclose(self.plsreg.loadings_p, self.plsda.loadings_p)
        assert_allclose(self.plsreg.loadings_q, self.plsda.loadings_q)
        assert_allclose(self.plsreg.beta_coeffs, self.plsda.beta_coeffs)
        assert_allclose(self.plsreg.modelParameters['R2Y'], self.plsda.modelParameters['PLS']['R2Y'])
        assert_allclose(self.plsreg.modelParameters['R2X'], self.plsda.modelParameters['PLS']['R2X'])
        assert_allclose(self.plsreg.modelParameters['SSX'], self.plsda.modelParameters['PLS']['SSX'])
        assert_allclose(self.plsreg.modelParameters['SSY'], self.plsda.modelParameters['PLS']['SSY'])
        assert_allclose(self.plsreg.modelParameters['SSXcomp'], self.plsda.modelParameters['PLS']['SSXcomp'])
        assert_allclose(self.plsreg.modelParameters['SSYcomp'], self.plsda.modelParameters['PLS']['SSYcomp'])
        

if __name__ == '__main__':
    unittest.main()

