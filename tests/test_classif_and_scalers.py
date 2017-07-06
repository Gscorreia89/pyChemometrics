import unittest

from sklearn.datasets import make_regression, make_classification

from pyChemometrics import ChemometricsScaler, ChemometricsPLS, ChemometricsPCA

"""

Suite of tests to ensure that the results of data flow through the scalers and objects is working properly

"""


class test_classif_and_scalers(unittest.TestCase):
    """
    Verify if scaling behaviour is working appropriately with classifiers
    """

    def setUp(self):

        # Generate 2 fake classification datasets, one with 2 classes and another with 3
        self.twoclass_dataset = make_classification(40, n_features=100, n_informative=5, n_redundant=5, n_classes=2)
        self.regression_dataset_y = make_regression(40, 100, 5, 1)
        #self.three_classdataset = make_classification(40, n_features=100, n_informative=5, n_redundant=5, n_classes=3)

        self.pls = ChemometricsPLS()
        self.pca = ChemometricsPCA()

        self.scaler_x = ChemometricsScaler()
        self.scaler_y = ChemometricsScaler()

    def test_scaler_plsreg_singley(self):
        self.plsreg.fit(self.twoclass_dataset[0], self.twoclass_dataset([1]))
        self.plslog.fit(self.twoclass_dataset[0], self.twoclass_dataset([1]))
        self.assertEqual(self.plsreg.rotations_cs, self.plslog.rotations_cs)
        self.assertEqual(self.plsreg.rotations_cs, self.plslog.rotations_cs)
        self.assertEqual(self.plsreg.rotations_cs, self.plslog.rotations_cs)
        self.assertEqual(self.plsreg.rotations_cs, self.plslog.rotations_cs)
        self.assertEqual(self.plsreg.rotations_cs, self.plslog.rotations_cs)

    def test_scaler_plsreg_multiy(self):
        self.plsreg.fit(self.threeclass_dataset[0], self.threeclass_dataset([1]))
        self.plslog.fit(self.threeclass_dataset[0], self.threeclass_dataset([1]))
        self.assertEqual(self.data.noFeatures, self.noFeat)

    def test_scaler_and_plsda_binary(self):
        self.plsreg.fit(self.twoclass_dataset[0], self.twoclass_dataset([1]))
        self.plslog.fit(self.twoclass_dataset[0], self.twoclass_dataset([1]))
        self.assertEqual(self.plsreg.rotations_cs, self.plslog.rotations_cs)
        self.assertEqual(self.plsreg.rotations_cs, self.plslog.rotations_cs)
        self.assertEqual(self.plsreg.rotations_cs, self.plslog.rotations_cs)
        self.assertEqual(self.plsreg.rotations_cs, self.plslog.rotations_cs)
        self.assertEqual(self.plsreg.rotations_cs, self.plslog.rotations_cs)

    def test_scaler_and_plsda_reg(self):
        self.plsreg.fit(self.threeclass_dataset[0], self.threeclass_dataset([1]))
        self.plslog.fit(self.threeclass_dataset[0], self.threeclass_dataset([1]))
        self.assertEqual(self.data.noFeatures, self.noFeat)

if __name__ == '__main__':
    unittest.main()
