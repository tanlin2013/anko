import numpy as np
from dataclasses import dataclass
from .utils import InfoCriterion
from .models import Gaussian, LinearRegression, Sgn, MAD


@dataclass
class Params:
    # Policy
    scaleless_t: bool = True
    z_normalization: bool = True
    info_criterion: InfoCriterion = InfoCriterion.AIC
    min_sample_size: int = 10
    # Threshold
    p_normality: float = 5e-3
    std_width: float = 1.5
    linear_res: float = 1.5
    sgn_res: float = 1.5
    mad_res: float = 1.5
    min_res: float = 10
    # Tolerance
    gaussian_err: float = 10
    linear_err: float = 10
    sgn_err: float = 10
    mad_err: float = 10


@dataclass
class ErrorCode:
    ok: str = "fitting converged"
    unconverged: str = "model {} does not converged, err = {} > tol = {}"
    low_sample: str = "number of data points {} is less than Params.min_sample_size {}"


@dataclass
class FittingResult:
    best_model: str = None
    popt: float = None
    perr: float = None
    outliers: list = None
    residual: np.ndarray = None
    error_code: ErrorCode = ErrorCode.ok


class AnomalyDetector:

    def __init__(self, t, x, params=Params):
        self.params = params
        self.x = np.array(x)
        if self.x.size < self.params.min_sample_size:
            raise ValueError(ErrorCode.low_sample.format(x.size, self.params.min_sample_size))
        if params.scaleless_t:
            self.t = np.arange(self.x.size)
        else:
            self.t = np.array(t)

    def fit(self) -> FittingResult:
        result = FittingResult()
        proceed_to_ansatzes = True

        model = Gaussian(self.x)
        if model.is_normal_distribution(self.params.p_normality):
            popt, perr = model.fit()
            if np.dot(perr[1:], perr[1:]) < self.params.gaussian_err:
                result.best_model = model.name
                result.popt = popt
                result.perr = perr
                result.residual = model.residual(self.params.min_res)
                proceed_to_ansatzes = False

        models = [
            LinearRegression(self.t, self.x),
            Sgn(self.t, self.x),
            MAD(self.t, self.x)
        ]
        if proceed_to_ansatzes:
            tmp_result = {}
            for model in models:
                tmp_result[model.name] = {}
                tmp_result[model.name]['popt'], tmp_result[model.name]['perr'] = model.fit()
                tmp_result[model.name]['ic_score'] = model.score(self.params.info_criterion)
                tmp_result[model.name]['residual'] = model.residual(tmp_result[model.name]['popt'],
                                                                    self.params.min_res,
                                                                    self.params.z_normalization)
            result.best_model = min(tmp_result.items(), key=lambda k: k[1]['ic_score'])[0]
            result.popt = tmp_result[result.best_model]['popt']
            result.perr = tmp_result[result.best_model]['perr']
            result.residual = tmp_result[model.name]['residual']
        return self.get_outliers(result)

    def get_outliers(self, result: FittingResult) -> FittingResult:
        if result.best_model == Gaussian.name:
            outlier_idx = abs(result.residual) > self.params.std_width
            result.residual = result.residual[outlier_idx]
        # @TODO: This treatment isn't perfect and may result in many garbage results
        # elif result.best_model == Sgn.name and (result.popt[0] - result.popt[1]) > self.params.min_res:
        #     # @Note: Treat all points after a sudden drop as outliers
        #     outlier_idx = np.where(self.t > result.popt[2])[0]
        #     result.residual = (result.popt[1] - result.popt[0]) * np.ones(len(outlier_idx))
        else:
            outlier_idx = abs(result.residual) > getattr(self.params, "{}_res".format(result.best_model))
            result.residual = result.residual[outlier_idx]
        result.outliers = list(zip(self.t[outlier_idx], self.x[outlier_idx]))

        err_norm = np.linalg.norm(result.perr)
        err_thres = getattr(self.params, "{}_err".format(result.best_model))
        if err_norm > err_thres:
            result.error_code = ErrorCode.unconverged.format(result.best_model, err_norm, err_thres)
        return result
