import unittest
from numpy.testing import assert_array_equal, assert_allclose

from sklearn.datasets import make_regression, make_classification

from pyChemometrics import ChemometricsScaler, ChemometricsPLS, ChemometricsPCA

"""

Suite of tests to ensure that the data flow through the scalers and objects is coherent.
The scale of the data after inverse_transform and pre

"""


class test_classif_and_scalers(unittest.TestCase):
    """
    Verify if scaling behaviour is working appropriately with classifiers
    """

    def setUp(self):

        # Generate 2 fake classification datasets, one with 2 classes and another with 3
        self.twoclass_dataset = make_classification(40, n_features=100, n_informative=5, n_redundant=5, n_classes=2)
        self.regression_dataset_single_y = make_regression(40, 100, 5, 1)
        #self.three_classdataset = make_classification(40, n_features=100, n_informative=5, n_redundant=5, n_classes=3)

        self.pls_mconly = ChemometricsPLS(xscaler=None)
        self.pls_uvscale = ChemometricsPLS(xscaler=ChemometricsScaler(1))
        self.pca_mconly = ChemometricsPCA(scaler=None)
        self.pca_uvscale = ChemometricsPCA(scaler=ChemometricsScaler(1))

        self.scaler_x = ChemometricsScaler(1)
        self.scaler_y = ChemometricsScaler(1)

        self.x_uvscaled_reg_single_y = self.scaler_x.fit_transform(self.regression_dataset_single_y[0])

    def test_scaler_pca(self):
        # Doesn't really matter which dataset is used here...
        self.pca_mconly.fit(self.x_uvscaled_reg_single_y)
        self.pca_uvscale.fit(self.regression_dataset_single_y[0])

        assert_allclose(self.pca_mconly.scores, self.pca_uvscale.scores)
        assert_allclose(self.pca_mconly.loadings, self.pca_uvscale.loadings)
        assert_allclose(self.pca_mconly.modelParameters['R2X'], self.pca_uvscale.modelParameters['R2X'])
        assert_allclose(self.pca_mconly.modelParameters['VarExp'], self.pca_uvscale.modelParameters['VarExp'])
        assert_allclose(self.pca_mconly.modelParameters['VarExpRatio'], self.pca_uvscale.modelParameters['VarExpRatio'])

"""
    def test_scaler_plsreg_singley(self):
        self.plsreg.fit(self.twoclass_dataset[0], self.twoclass_dataset([1]))
        self.plslog.fit(self.twoclass_dataset[0], self.twoclass_dataset([1]))

        assert_array_equal(self.plsreg.scores_t, self.plslog.scores_t)
        assert_array_equal(self.plsreg.scores_u, self.plslog.scores_u)
        assert_array_equal(self.plsreg.scores_u, self.plslog.scores_u)
        assert_array_equal(self.plsreg.loadings_p, self.plslog.loadings_p)
        assert_array_equal(self.plsreg.rotations_cs, self.plslog.rotations_cs)
        assert_array_equal(self.plsreg.rotations_ws, self.plslog.rotations_ws)
        assert_array_equal(self.plsreg.weights_w, self.plslog.weights_w)
        assert_array_equal(self.plsreg.weights_c, self.plslog.weights_c)
        assert_array_equal(self.plsreg.loadings_p, self.plslog.loadings_p)
        assert_array_equal(self.plsreg.loadings_q, self.plslog.loadings_q)
        assert_array_equal(self.plsreg.beta_coeffs, self.plslog.beta_coeffs)
        assert_array_equal(self.plsreg.modelParameters['R2Y'], self.plslog.modelParameters['PLS']['R2Y'])
        assert_array_equal(self.plsreg.modelParameters['R2X'], self.plslog.modelParameters['PLS']['R2X'])
        assert_array_equal(self.plsreg.modelParameters['SSX'], self.plslog.modelParameters['PLS']['SSX'])
        assert_array_equal(self.plsreg.modelParameters['SSY'], self.plslog.modelParameters['PLS']['SSY'])
        assert_array_equal(self.plsreg.modelParameters['SSXcomp'], self.plslog.modelParameters['PLS']['SSXcomp'])
        assert_array_equal(self.plsreg.modelParameters['SSYcomp'], self.plslog.modelParameters['PLS']['SSYcomp'])


    def test_scaler_plsreg_multiy(self):
        self.plsreg.fit(self.threeclass_dataset[0], self.threeclass_dataset([1]))
        self.plslog.fit(self.threeclass_dataset[0], self.threeclass_dataset([1]))
        assert_array_equal(self.data.noFeatures, self.noFeat)

    def test_scaler_and_plsda_binary(self):
        self.plsreg.fit(self.twoclass_dataset[0], self.twoclass_dataset([1]))
        self.plslog.fit(self.twoclass_dataset[0], self.twoclass_dataset([1]))
        assert_array_equal(self.plsreg.rotations_cs, self.plslog.rotations_cs)
        assert_array_equal(self.plsreg.rotations_cs, self.plslog.rotations_cs)
        assert_array_equal(self.plsreg.rotations_cs, self.plslog.rotations_cs)
        assert_array_equal(self.plsreg.rotations_cs, self.plslog.rotations_cs)
        assert_array_equal(self.plsreg.rotations_cs, self.plslog.rotations_cs)

    def test_scaler_and_plsda_reg(self):
        self.plsreg.fit(self.threeclass_dataset[0], self.threeclass_dataset([1]))
        self.plslog.fit(self.threeclass_dataset[0], self.threeclass_dataset([1]))
        assert_array_equal(self.data.noFeatures, self.noFeat)
"""

if __name__ == '__main__':
    unittest.main()
