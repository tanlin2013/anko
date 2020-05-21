import os
import sys
import numpy as np
sys.path.append('../anko')
from stats_util import median_absolute_deviation, z_normalization
from anomaly_detector import AnomalyDetector
from stats_util import DBSCAN


if __name__ == '__main__':

    dir_path = os.path.dirname(os.path.realpath(__file__))
    npzfile = np.load(dir_path + '/test_series.npz')
    series_data = [npzfile['arr_%i' % i] for i in range(len(npzfile.files))]
    npzfile.close()
    series = np.array(series_data)

    # for i in range(len(series)):

    i = 28
    t = np.arange(1, len(series[i]) + 1)
    print(series[i])

    out = DBSCAN(series[i], 3.0)
    print(np.std(series[i]))
    print(out)

    z = (series[i] - np.median(series[i])) / median_absolute_deviation(series[i])
    print(z)
    print(z_normalization(series[i]))
