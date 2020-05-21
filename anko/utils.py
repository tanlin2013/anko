import numpy as np
from enum import Enum


class InfoCriterion(Enum):
    AIC = 1,
    BIC = 2


class ICScore:

    @staticmethod
    def aic(y: np.ndarray, y_predict: np.ndarray, p: int) -> float:
        r"""
        Compute Akaike information criterion for model selection.

        .. math::
            \mathcal{AIC} = n \log(\mathcal{RSS}/n) + 2p,

        where :math:`\mathcal{RSS}` is the residual sum of squares, and :math:`n` is the number of data samples.

        Args:
            y (numpy.ndarray): Data samples.
            y_predict (numpy.ndarray): Prediction by fitting.
            p (int): Fitting degrees of freedom, i.e. the number of parameters to fit with.

        Returns:
            float:
                aic_score (float):

        """
        n = len(y)
        res = np.subtract(y, y_predict)
        rss = np.sum(np.power(res, 2))
        aic_score = n * np.log(rss / n) + 2 * p
        return aic_score

    @staticmethod
    def bic(y: np.ndarray, y_predict: np.ndarray, p: int) -> float:
        r"""
        Compute Bayesian information criterion for model selection.

        .. math::
            \mathcal{BIC} = n \log(\mathcal{RSS}/n) + p \log(n),

        where :math:`\mathcal{RSS}` is the residual sum of squares, and :math:`n` is the number of data samples.

        Args:
            y (numpy.ndarray): Data samples.
            y_predict (numpy.ndarray): Prediction by fitting.
            p (int): Fitting degrees of freedom, i.e. the number of parameters to fit with.

        Returns:
            float:
                bic_score (float):

        """
        n = len(y)
        res = np.subtract(y, y_predict)
        rss = np.sum(np.power(res, 2))
        bic_score = n * np.log(rss / n) + p * np.log(n)
        return bic_score


def fitting_residual(x: np.ndarray, y: np.ndarray, func, args, mask_min: float = None,
                     standardized: bool = False) -> np.ndarray:
    """
    Compute the fitting residual.

    Args:
        x (numpy.ndarray): x coordinate of input data points.
        y (numpy.ndarray): y coordinate of input data points.
        func (callable): Fitting function.
        args (numpy.ndarray): Best estimated arguments of fitting function.
        mask_min (float, optional): If not None, mask resuduals that are smaller than mask_min to zero. This is always performed before standardization.
        standardized (bool, optional): Standardize residual to z-score formalism.

    Returns:
        numpy.ndarray:
            res (numpy.ndarray): Residual of each corresponding data points (x, y).

    """
    y_predict = func(x, *args)
    res = np.subtract(y, y_predict)
    norm = np.std(res)
    if mask_min is not None:
        res[np.where(abs(res) < mask_min)] = 0
    if standardized and norm != 0:
        res /= norm
    return res


def z_score(x: np.ndarray) -> np.ndarray:
    r"""
    Perform z-score normalizaion on input array x.

    .. math::
        z = \frac{x-\mu}{\sigma}.

    Args:
        x (numpy.ndarray): Input values.

    Returns:
        numpy.ndarray:
            normalized_x (numpy.ndarray): Output array.

    """
    return (x - np.mean(x)) / np.std(x)


def median_absolute_deviation(x: np.ndarray) -> np.ndarray:
    r"""
    Calculate the median absolute deviation. The is a robust statistical measurement defined by

    .. math::
        mad = median(|x_i-median(x)|).

    Args:
        x (numpy.ndarray): Input values.

    Returns:
        numpy.ndarray:
            mad (numpy.ndarray): Output array.
    """
    return np.median(abs(x - np.median(x)))


def modified_z_score(x: np.ndarray) -> np.ndarray:
    r"""
    Perform modified z-score normalizaion on input array x,
    where 0.6745 is the 0.75th quartile of the standard normal distribution,
    to which the MAD converges to.

    .. math::
        z = \frac{0.6745(x_i - median(x))}{MAD}

    Args:
        x (numpy.ndarray): Input values.

    Returns:

    """
    return 0.6745 * (x - np.median(x)) / median_absolute_deviation(x)
