#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import os
import unittest
from anko.anomaly_detector import AnomalyDetector
# =============================================================================
# import sys
# sys.path.append('../anko')
# from anomaly_detector import AnomalyDetector
# =============================================================================
class TestAnomalyDetector(unittest.TestCase):
    
    def test_check_on_linear(self):
        a = 6; b = 10
        t = np.arange(1, 100+1)
        series = a * t +b      
        agent = AnomalyDetector(t, series)
        statsdata = agent.check()        
        self.assertEqual(statsdata['model'], 'linear_regression')
        np.testing.assert_allclose(statsdata['popt'], [b, a], atol=1, rtol=1e-1)
    
    def test_check_on_erf(self):
        t = np.arange(1, 100+1)
        series = 20*(np.sign(t-20)+2)        
        agent = AnomalyDetector(t, series)
        statsdata = agent.check()        
        self.assertEqual(statsdata['model'], 'increase_step_func')
        np.testing.assert_allclose(statsdata['popt'], [20, 60, 20], atol=1, rtol=1e-1)
        np.testing.assert_allclose(statsdata['perr'], [0.4629, 0.2242, 0], atol=1, rtol=1e-1)
        np.testing.assert_allclose(statsdata['anomalous_data'], [(20, 40)])
    
    def test_check_on_exp(self):
        a = 10; alpha = 3
        t = np.arange(1, 100+1)
        series = a * np.exp(-alpha*t)       
        agent = AnomalyDetector(t, series)
        statsdata = agent.check()        
        self.assertEqual(statsdata['model'], 'exp_decay')
        np.testing.assert_allclose(statsdata['popt'], [a, alpha], atol=1, rtol=1e-1)
        np.testing.assert_allclose(statsdata['perr'], 0, atol=1, rtol=1e-1)
        np.testing.assert_allclose(statsdata['anomalous_data'], [])
    
    def test_check_on_gaussian(self):
        mean = 100; std = 10
        t = np.arange(1, 100+1)
        series = np.random.normal(mean, std, size=100).astype(int)
        agent = AnomalyDetector(t, series)
        statsdata = agent.check()
        self.assertEqual(statsdata['model'], 'gaussian')
        np.testing.assert_allclose(statsdata['popt'][1:], [mean, std], atol=5, rtol=1e-1)
        #np.testing.assert_allclose(np.dot(statsdata['perr'], statsdata['perr']), 2, atol=5, rtol=1e-1)
        
    @staticmethod
    def read_from_file():
        dir_path = os.path.dirname(os.path.realpath(__file__))  
        npzfile = np.load(dir_path+'/test_series.npz')
        series_data = [npzfile['arr_%i'%i] for i in range(len(npzfile.files))]
        npzfile.close()
        return np.array(series_data)
    
    def test_check_from_file(self, check_all=False):
        min_sample_size = 10
        series = self.__class__.read_from_file()
        if check_all: epoch = range(len(series))
        else: epoch = np.random.randint(0, len(series), size=10)
        for i in epoch:
            if len(series[i]) < min_sample_size: continue
            t = np.arange(1, len(series[i])+1)
            agent = AnomalyDetector(t, series[i])
            agent.apply_policies["min_sample_size"] = min_sample_size
            statsdata = agent.check()
            statsdata["series"] = series[i]
            #print(statsdata)
      
if __name__ == '__main__':
    unittest.main()
