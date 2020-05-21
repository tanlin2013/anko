import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN


class MDBSCAN:

    def __init__(self, t, x):
        self.t = t
        self.x = x
        self._params = {}
        assert(t.ndim == 1)
        assert(t.shape == x.shape)

    @property
    def params(self):
        return self._params

    @params.setter
    def set_params(self):
        pass

    @property
    def X(self):
        return np.concatenate()

    @staticmethod
    def median_absolute_deviation(x):
        # @TODO: This can be zero for super constant data
        return np.median(abs(x - np.median(x)))

    @staticmethod
    def standard_median_score(x):
        return (x - np.median(x))/MDBSCAN.median_absolute_deviation(x)

    def fit(self, eps, min_samples):
        db = DBSCAN(eps, min_samples, metric='euclidean',
                    metric_params=None, algorithm='auto',
                    leaf_size=30, p=None, n_jobs=None).fit(self.X)

        labels = db.labels_

        # Number of clusters in labels, ignoring noise if present.
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        out = {
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "labels": labels
        }
        return

    def trend_separator(self):
        pass

