import unittest
import os
import pandas as pds
import numpy as np
from numpy.testing import assert_allclose
from sklearn.model_selection import KFold
from pyChemometrics import ChemometricsScaler, ChemometricsPCA

import sys
sys

"""
Suite of tests to assess coherence and functionality of the PCA object.

"""


class TestPCA(unittest.TestCase):
    """

    Verify outputs of the ChemometricsPCA object

    """

    def setUp(self):

        try:
            # Generate a fake classification dataset
            t_dset = pds.read_csv(os.path.join(os.path.dirname(__file__), './test_data/classification_twoclass.csv'))
            self.xmat = t_dset.iloc[:, 1::].values

        except (IOError, OSError) as ioerr:
            os.system('python gen_synthetic_datasets.py')
            t_dset = pds.read_csv(os.path.join(os.path.dirname(__file__), './test_data/classification_twoclass.csv'))
            self.xmat = t_dset.iloc[:, 1::].values

        self.expected_modelParameters = {'R2X': 0.12913056143673818,
                                          'S0': 0.9803124001345157,
                                         'VarExp': np.array([9.44045066, 8.79710591, 8.11561924]),
                                          'VarExpRatio': np.array([0.04625821, 0.04310582, 0.03976653])}
        self.expected_cvParameters = {'Q2X': -0.10571035538454221, 'Mean_VarExp_Test': -0.0090083829247783621,
                                      'Stdev_VarExp_Test': 0.0037778709253728452,
                                      'Mean_VarExpRatio_Training': np.array([0.05108043,  0.04669199,  0.04380617]),
                                      'Stdev_VarExpRatio_Training': np.array([0.00130025,  0.00094489,  0.00044059])}

        self.expected_scores = np.loadtxt(os.path.join(os.path.dirname(__file__), './test_data/pca_scores.csv'), delimiter=',')
        self.expected_loadings = np.loadtxt(os.path.join(os.path.dirname(__file__), './test_data/pca_loadings.csv'), delimiter=',')

        self.expected_scores_mc = np.loadtxt(os.path.join(os.path.dirname(__file__), './test_data/pca_scores_mc.csv'), delimiter=',')
        self.expected_loadings_mc = np.loadtxt(os.path.join(os.path.dirname(__file__), './test_data/pca_loadings_mc.csv'), delimiter=',')

        self.expected_scores_par = np.loadtxt(os.path.join(os.path.dirname(__file__), './test_data/pca_scores_par.csv'), delimiter=',')
        self.expected_loadings_par = np.loadtxt(os.path.join(os.path.dirname(__file__), './test_data/pca_loadings_par.csv'), delimiter=',')

        self.expected_dmodx = np.loadtxt(os.path.join(os.path.dirname(__file__), './test_data/pca_dmodx.csv'), delimiter=',')
        cvloadings = np.loadtxt(os.path.join(os.path.dirname(__file__), './test_data/pca_cvloads.csv'), delimiter=',')
        self.expected_cv_meanloadings = cvloadings[0:3, :]
        self.expected_cv_stdevloadings = cvloadings[3::, :]

        self.expected_t2 = np.array([ 9.00313686,  8.69095296,  8.34753638])
        self.expected_outlier_dmodx = np.array([])
        self.expected_outlier_t2 = np.array([14])
        self.x_scaler = ChemometricsScaler(1)
        self.pcamodel = ChemometricsPCA(ncomps=3, scaler=self.x_scaler)

    def test_fit(self):

        self.pcamodel.fit(self.xmat)

        for key, item in self.expected_modelParameters.items():
            assert_allclose(self.pcamodel.modelParameters[key], item, rtol=1e-05)

        assert_allclose(self.pcamodel.scores, self.expected_scores)
        assert_allclose(self.pcamodel.loadings, self.expected_loadings)

    def test_transform(self):

        self.pcamodel.fit(self.xmat)
        assert_allclose(self.pcamodel.transform(self.xmat), self.expected_scores)

    def test_fit_transform(self):
        assert_allclose(self.pcamodel.fit_transform(self.xmat), self.expected_scores)

    def test_cv(self):
        # Restart the seed and perform cross validation
        np.random.seed(0)
        self.pcamodel.cross_validation(self.xmat, cv_method=KFold(7, True))

        assert_allclose(self.pcamodel.cvParameters['Q2X'], self.expected_cvParameters['Q2X'], rtol=1e-5)
        assert_allclose(self.pcamodel.cvParameters['Mean_VarExpRatio_Training'], self.expected_cvParameters['Mean_VarExpRatio_Training'], rtol=1e-5)
        assert_allclose(self.pcamodel.cvParameters['Stdev_VarExpRatio_Training'], self.expected_cvParameters['Stdev_VarExpRatio_Training'], rtol=1e-5)
        assert_allclose(self.pcamodel.cvParameters['Mean_VarExp_Test'], self.expected_cvParameters['Mean_VarExp_Test'], rtol=1e-5)
        assert_allclose(self.pcamodel.cvParameters['Stdev_VarExp_Test'], self.expected_cvParameters['Stdev_VarExp_Test'], rtol=1e-5)

        assert_allclose(np.array(self.pcamodel.cvParameters['Mean_Loadings']), self.expected_cv_meanloadings)
        assert_allclose(np.array(self.pcamodel.cvParameters['Stdev_Loadings']), self.expected_cv_stdevloadings)

    def test_hotellingT2(self):
        self.pcamodel.fit(self.xmat)
        assert_allclose(self.pcamodel.hotelling_T2(None, 0.05), self.expected_t2, rtol=1e-05)

    def test_dmodx(self):
        self.pcamodel.fit(self.xmat)
        assert_allclose(self.pcamodel.dmodx(self.xmat), self.expected_dmodx)

    def test_scalers(self):
        x_scaler_par = ChemometricsScaler(1 / 2)
        x_scaler_mc = ChemometricsScaler(0)

        pareto_model = ChemometricsPCA(ncomps=3, scaler=x_scaler_par)
        mc_model = ChemometricsPCA(ncomps=3, scaler=x_scaler_mc)

        pareto_model.fit(self.xmat)
        mc_model.fit(self.xmat)

        assert_allclose(pareto_model.loadings, self.expected_loadings_par)
        assert_allclose(pareto_model.scores, self.expected_scores_par)

        assert_allclose(mc_model.loadings, self.expected_loadings_mc)
        assert_allclose(mc_model.scores, self.expected_scores_mc)

    def test_outliers(self):
        self.pcamodel.fit(self.xmat)
        outliers_t2 = self.pcamodel.outlier(self.xmat)
        outliers_dmodx = self.pcamodel.outlier(self.xmat, measure='DmodX')
        assert_allclose(outliers_t2, self.expected_outlier_t2)
        assert_allclose(outliers_dmodx, self.expected_outlier_dmodx)
        return None


if __name__ == '__main__':
    unittest.main()
