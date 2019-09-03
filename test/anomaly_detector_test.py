#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import unittest
from anko.anomaly_detector import AnomalyDetector

class TestAnomalyDetector(unittest.TestCase):
    
    def test_build_statsdata(self):
        return
    
    def test_check(self):
        t = np.arange(1, 100+1)
        series = 20*(np.sign(t-20)+2)
        
        agent = AnomalyDetector(t, series)
        statsdata = agent.check()
        self.assertEqual(statsdata['model'], 'increase_step_func')
        np.testing.assert_allclose(statsdata['popt'], [20, 60, 20], atol=1e-1)
        np.testing.assert_allclose(statsdata['perr'], [0.4629, 0.2242, 0])
        np.testing.assert_allclose(statsdata['anomalous_data'], [(20, 40)])
      
if __name__ == '__main__':
    unittest.main()
