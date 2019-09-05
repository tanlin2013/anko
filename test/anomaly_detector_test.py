#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import unittest
from anko.anomaly_detector import AnomalyDetector

class TestAnomalyDetector(unittest.TestCase):
    
    def test_build_statsdata(self):
        return
    
    def test_check_on_erf(self):
        t = np.arange(1, 100+1)
        series = 20*(np.sign(t-20)+2)
        
        agent = AnomalyDetector(t, series)
        agent.apply_policies["z_normalization"] = True
        statsdata = agent.check()
        
        self.assertEqual(statsdata['model'], 'increase_step_func')
        np.testing.assert_allclose(statsdata['popt'], [20, 60, 20], atol=1, rtol=1e-1)
        np.testing.assert_allclose(statsdata['perr'], [0.4629, 0.2242, 0], atol=1, rtol=1e-1)
        np.testing.assert_allclose(statsdata['anomalous_data'], [(20, 40)])
    
    def test_check_on_exp(self):
        t = np.arange(1, 100+1)
        series = 10 * np.exp(-3*t)
        
        agent = AnomalyDetector(t, series)
        agent.apply_policies["z_normalization"] = True
        statsdata = agent.check()
        
        self.assertEqual(statsdata['model'], 'exp_decay')
        np.testing.assert_allclose(statsdata['popt'], [10, 3], atol=1, rtol=1e-1)
        np.testing.assert_allclose(statsdata['perr'], 0, atol=1, rtol=1e-1)
        np.testing.assert_allclose(statsdata['anomalous_data'], [])
       
    @staticmethod
    def read_from_file():
        npzfile = np.load('./test_series.npz')
        series_data = [npzfile['arr_%i'%i] for i in range(len(npzfile.files))]
        npzfile.close()
        return np.array(series_data)
    
    def test_check_from_file(self):
        sample_idx = np.random.randint(1, 100, size=10)
        series = self.__class__.read_from_file()[sample_idx]
        for i in range(10):
            try:
                t = np.arange(1, len(series[i])+1)
                agent = AnomalyDetector(t, series[i])
                agent.apply_policies["z_normalization"] = True
                statsdata = agent.check()
                statsdata["series"] = series[i]
            except ValueError:
                pass
      
if __name__ == '__main__':
    unittest.main()
