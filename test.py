import unittest
import numpy as np
import stats_util

class TestStatsUtil(unittest.TestCase):

    def test_AIC_score(self):
        y = np.arange(5)
        y_pred = y + np.array([0.1, -0.2, 0.3, -0.4, 0.5])
        self.assertAlmostEqual(stats_util.AIC_score(y,y_pred,2), -7.03637456595)

    def test_BIC_score(self):
        y = np.arange(5)
        y_pred = y + np.array([0.1, -0.2, 0.3, -0.4, 0.5])
        self.assertAlmostEqual(stats_util.BIC_score(y,y_pred,2), -7.81749874108)
        
    def test_boxcox(self):    
        x = np.random.random(5)
        np.testing.assert_allclose(stats_util.boxcox(x, lmbda=0), np.log(x))
        
    def test_data_is_flat(self):
        x = np.arange(10); noise = 1e-1*(np.random.random(10)-0.5)
        y = 5*x + noise
        self.assertTrue(stats_util.data_is_flat(x, y))
    
    def test_discontinuous_idx(self):
        x = np.arange(100)
        y = -10*np.sign(x-10) + 10*np.sign(x-40) - 5
        np.testing.assert_allclose(stats_util.discontinuous_idx(y, std_width=1), np.array([9,10,39,40]))
        
    def test_exp_decay(self):
        x = np.arange(5)
        np.testing.assert_allclose(stats_util.exp_decay(x, 1, 0), np.ones(5))
        
    def test_exp_decay_fit(self):
        x = np.arange(100); noise = 1e-5*(np.random.random(100)-0.5)
        y = np.exp(-4*x) + noise
        popt, perr = stats_util.exp_decay_fit(x, y)
        np.testing.assert_allclose(popt, np.array([1,4]), atol=1e-3)
        np.testing.assert_allclose(perr, np.zeros(2), atol=1e-2)
        
    def test_fitting_residual(self):
        x = np.arange(100); noise = 1e-5*(np.random.random(100)-0.5)
        y = np.polyval([2,1,1], x) + noise
        f = lambda x, p0, p1, p2: p0*x**2 + p1*x + p2
        p = [2, 1, 1]
        np.testing.assert_allclose(stats_util.fitting_residual(x,y,f,p), noise, atol=1e-5)
        
    def test_flat_histogram(self):
        x = np.array([47, 47, 47, 47, 47, 47, 47, 47, 46, 45, 36])
        popt, perr = stats_util.flat_histogram(x)
        self.assertListEqual(popt, [8, 47, np.inf])
        self.assertListEqual(perr, [0, 0, 0])
        
    def test_gaussian_fit(self):
        x = 
        popt, perr = stats_util.gaussian_fit()
        self.assertListEqual(popt, [])
        self.assertListEqual(perr, [])
        
        
if __name__ == '__main__':
    unittest.main()
    
    