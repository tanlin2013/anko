import numpy as np
import collections
from scipy.stats import linregress, normaltest
from scipy.optimize import curve_fit
from sklearn.cluster import DBSCAN as skDBSCAN
from .utils import InfoCriterion, ICScore, fitting_residual, median_absolute_deviation


class Model:

    def __init__(self, t, x):
        self.t = t
        self.x = x
        self.x_pred = None

    def score(self, info_criterion=InfoCriterion.AIC):
        if info_criterion == InfoCriterion.AIC:
            return ICScore.aic(self.x, self.x_pred, self.dof)
        elif info_criterion == InfoCriterion.BIC:
            return ICScore.bic(self.x, self.x_pred, self.dof)

    def residual(self, popt, mask_min, standardized):
        return fitting_residual(self.t, self.x, self.func, popt,
                                mask_min=mask_min,
                                standardized=standardized)


class Gaussian:
    name = 'gaussian'

    def __init__(self, x):
        self.x = x

    def is_normal_distribution(self, p_normality=1e-3):
        try:
            normality = normaltest(self.x)
        except ValueError:
            normality = [np.inf, np.inf]

        return normality[1] >= p_normality and np.isfinite(normality[1])

    @staticmethod
    def binning(x, bins='auto'):
        if bins is not None:
            hist, bin_edges = np.histogram(x, bins=bins)
            bin_edges = (0.5 * (bin_edges[1:] + bin_edges[:-1]))
        else:
            counter = collections.Counter(x)
            bin_edges = np.fromiter(counter.keys(), dtype=float)
            hist = np.fromiter(counter.values(), dtype=float)
            bins_idx = np.argsort(bin_edges)
            bin_edges = bin_edges[bins_idx]
            hist = hist[bins_idx]
        return bin_edges, hist

    @staticmethod
    def func(x, a, x0, sigma):
        return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

    def fit(self, bins='auto', maxfev=2000, bounds=[0, 1e+6]):
        bin_edges, hist = self.binning(self.x, bins)
        a_sg = max(hist) * 0.9
        m_sg = np.mean(self.x)
        std_sg = np.std(self.x)
        popt, pcov = curve_fit(self.func, bin_edges, hist, p0=[a_sg, m_sg, std_sg], maxfev=maxfev, bounds=bounds)
        perr = np.sqrt(np.diag(pcov))
        return popt, perr

    def residual(self, mask_min):
        mean_centered_series = self.x - np.mean(self.x)
        mean_centered_series[np.where(abs(mean_centered_series) < mask_min)] = 0
        return mean_centered_series / np.std(self.x)


class LinearRegression(Model):
    name = 'linear'
    dof = 2

    def __init__(self, t, x):
        super(LinearRegression, self).__init__(t, x)

    @staticmethod
    def func(t, a, b):
        return a + b*t

    def fit(self):
        slope, intercept, r_value, p_value, std_err = linregress(self.t, self.x)
        self.x_pred = np.polyval([slope, intercept], self.t)
        return np.array([intercept, slope]), std_err


class Sgn(Model):
    name = 'sgn'
    dof = 3

    def __init__(self, t, x):
        super(Sgn, self).__init__(t, x)

    @staticmethod
    def func(t: np.ndarray, a: float, b: float, t0: float) -> np.ndarray:
        return (b-a)/2 * np.sign(t-t0) + (a+b)/2

    def fit(self, maxfev: int=2000, bounds=[0, 1e+6]):
        a_sg = self.x[0]
        b_sg = self.x[-1]
        t0_sg = self.t[np.argmax(np.diff(self.x))]
        popt, pcov = curve_fit(self.func, self.t, self.x, p0=[a_sg, b_sg, t0_sg], maxfev=maxfev, bounds=bounds)
        perr = np.sqrt(np.diag(pcov))
        self.x_pred = self.func(self.t, popt[0], popt[1], popt[2])
        return popt, perr


class MAD(Model):
    name = 'mad'
    dof = 0

    def __init__(self, t, x):
        super(MAD, self).__init__(t, x)

    def func(self, t):
        return np.median(self.x)

    def fit(self):
        self.x_pred = self.func(self.t)
        perr = median_absolute_deviation(self.x)
        return [], perr


class DBSCAN:
    name = "dbscan"

    def __init__(self, x):
        # @TODO: To be implemented...
        self.x = x
        self.X = np.concatenate(
            (
                np.arange(1, self.n + 1).reshape(self.n, 1),
                x.reshape(self.n, 1)
            ),
            axis=1
        )

    @property
    def n(self):
        return self.x.size

    def fit(self, eps=0.9, min_samples=3):
        eps *= np.diff(np.histogram_bin_edges(self.x, bins='auto'))[0]
        # eps *= median_absolute_deviation(self.x)
        print(np.diff(np.histogram_bin_edges(self.x, bins='auto'))[0])
        print(median_absolute_deviation(self.x))

        db = skDBSCAN(
            eps,
            min_samples,
            metric='euclidean',
            metric_params=None,
            algorithm='auto',
            leaf_size=30,
            p=None,
            n_jobs=None
        ).fit(self.X)

        labels = db.labels_
        # @Note: Number of clusters in labels, ignoring noise if present.
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        out = {"n_clusters": n_clusters,
               "n_noise": n_noise,
               "labels": labels}
        return out
