import unittest

from sklearn.datasets import make_classification

from pyChemometrics import ChemometricsScaler, ChemometricsPLS, ChemometricsPLS_Logistic


# TODO: check this test the actual algorithm is fully tested
class test_plslogistic(unittest.TestCase):
    """

    Test the functionality of the PLS - Logistic code. Most of the tests focus on the DA model specifics and classification metrics.
    The underlying PLS regression functionality is tested in the 'test_plsobjconsistency' suite of tests.

    """

    def setUp(self):

        # Generate 2 fake classification datasets, one with 2 classes and another with 3
        self.twoclass_dataset = make_classification(40, n_features=100, n_informative=5, n_redundant=5, n_classes=2)
        self.three_classdataset = make_classification(40, n_features=100, n_informative=5, n_redundant=5, n_classes=3)
        y_scaler = ChemometricsScaler(with_mean=False, with_std=False)
        self.plsreg = ChemometricsPLS(n_comps=3, yscaler=y_scaler)
        self.plslog = ChemometricsPLS_Logistic(n_comps=3)

    def test_single_y(self):
        self.plsreg.fit(self.twoclass_dataset[0], self.twoclass_dataset([1]))
        self.plslog.fit(self.twoclass_dataset[0], self.twoclass_dataset([1]))
        self.assertEqual(self.plsreg.rotations_cs, self.plslog.rotations_cs)

    def test_multi_y(self):
        self.plsreg.fit(self.threeclass_dataset[0], self.threeclass_dataset([1]))
        self.plslog.fit(self.threeclass_dataset[0], self.threeclass_dataset([1]))
        self.assertEqual(self.data.noFeatures, self.noFeat)

    def test_(self):
        self.assertEqual()

if __name__ == '__main__':
    unittest.main()
