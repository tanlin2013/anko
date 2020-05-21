import unittest
import numpy as np
from anko import stats_util

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
        
    def test_data_is_linear(self):
        x = np.arange(10); noise = 1e-1*(np.random.random(10)-0.5)
        y = 5*x + noise
        self.assertTrue(stats_util.data_is_linear(x, y))
    
    def test_discontinuous_idx(self):
        x = np.arange(100)
        y = -10*np.sign(x-10) + 10*np.sign(x-40) - 5
        np.testing.assert_allclose(stats_util.discontinuous_idx(y, std_width=1), np.array([9,10,39,40]))
        
    def test_exp_decay(self):
        x = np.arange(5)
        np.testing.assert_allclose(stats_util.exp_decay(x, 1, 0), np.ones(5))
        
    def test_exp_decay_fit(self):
        x = np.arange(1,100+1); noise = 1e-3*(np.random.random(100)-0.5)
        y = np.exp(-4*x) * np.exp(noise)
        popt, perr = stats_util.exp_decay_fit(x, y)
        np.testing.assert_allclose(popt, np.array([1,4]), atol=1e-3)
        np.testing.assert_allclose(perr, np.zeros(2), atol=1e-2)
        
    def test_fitting_residual(self):
        x = np.arange(100); noise = 1e-5*(np.random.random(100)-0.5)
        y = np.polyval([2,1,1], x) + noise
        f = lambda x, p0, p1, p2: p0*x**2 + p1*x + p2
        p = [2, 1, 1]
        np.testing.assert_allclose(stats_util.fitting_residual(x,y,f,p,standardized=False),
                                   noise, atol=1e-5)
        
    def test_flat_histogram(self):
        x = np.array([47, 47, 47, 47, 47, 47, 47, 47, 46, 45, 36])
        popt, perr = stats_util.flat_histogram(x)
        np.testing.assert_allclose(popt, [8, 47, np.inf])
        np.testing.assert_allclose(perr, [0, 0, 0])
        
    def test_gaussian_fit(self):
        mean = 20; std = 3
        x = np.random.normal(mean, std, size=10000).astype(int)
        popt, perr = stats_util.gaussian_fit(x)
        np.testing.assert_allclose(popt[1:], [mean, std], rtol=1e-1)
        
# =============================================================================
#     def test_z_normalization(self):
#         x = [189, 188, 196, 196, 196, 193, 206, 203, 203, 214, 217, 217, 218, 218, 248, 247, 247, 252, 252, 253, 259, 259, 254, 252, 252, 252, 252, 247, 247, 247, 249, 249, 246, 243, 243, 242, 248, 255, 253, 251, 247, 251, 253, 253, 256, 260, 257, 258, 260, 260, 259, 258, 253, 249, 248, 251, 253, 253, 242, 286, 287, 292, 293, 293, 297, 296, 296, 288, 286, 286, 285, 285, 285, 283, 283, 283, 283, 283, 283, 283, 282, 282, 282, 282, 282, 282, 282, 278, 278, 278, 278, 274, 278, 276, 276, 276, 242, 276, 276, 276]
#         
# =============================================================================
    
if __name__ == '__main__':
    unittest.main()
    
    