import numpy as np
from . import stats_util
import copy
# TODO: add params and returns type in func entry

class AnomalyDetector:
    """!
    @param t (array_like):
    @param series (array_like):
    """
    def __init__(self, t, series):
        ## Policies for AnomalyDetector to follow with. 
        ## @param scaleless_t (bool, default True): If True, use np.arange(1, len(t)+1) for the fitting. 
        ## @param boxcox (bool, default False): If True, perform log-boxcox transformation before carrying out normal test. This will result in higher chances on selecting normal distribution method. 
        ## @param z_normalization (bool, default True): If True, apply z-score normalization to fitting residual. This parameter is stringly advised to define threshold values in AnomalyDetector.thres_params scalelessly.
        ## @param info_criterion (str, default 'AIC'): Information criterion for selecting fitting ansatzs, allowed fields are 'AIC' or 'BIC'.
        ## @param min_sample_size (int, default 10): Minimum number of data samples to execute AnomalyDetector. If provided number of samples is less than this attribute, raise ValueError.
        self.apply_policies = {
                "scaleless_t": True,
                "boxcox": False,
                "z_normalization": True,
                "info_criterion": 'AIC',
                "min_sample_size": 10
        }
        if isinstance(t, list): t = np.array(t)
        if isinstance(series, list): series = np.array(series)
        if t is None: self.apply_policies["scaleless_t"] = True
        if self.apply_policies["scaleless_t"]: 
            self.t = np.arange(1, len(series)+1)
        else:
            self.t = t
        self.series = series
        self._clone_t = copy.deepcopy(t)
        self._clone_series = copy.deepcopy(series)
        ## Boolean value if check failed.
        self.check_failed = True   
        ## Threshold values for selecting anomalous data.
        ## @param p_normality (float, default 5e-3): Threshold value for selecting normal distribution, in accordance with the p value of normal test.  
        ## @param normal_err (float, default 75): Threshold value for selecting normal distribution, in case that fitting on normal distribution failed and unconverged.
        ## @param normal_std_width (float, default 1.5): Threshold width of standard deviation, data points exceed this param will be regarded as anomalous.  
        ## @param normal_std_err (float, default 1e+1): Maximum tolerence of convergence. If fitting error is larger than this param, pass ConvergenceError to CheckResult.extra_info.
        ## @param linregress_std_err (float, default 1e+1): Maximum tolerence of convergence. If fitting error is larger than this param, pass ConvergenceError to CheckResult.extra_info.
        ## @param linregress_res (float, default 2): Threshold value of residual for linear regression, data points exceed this param will be regarded as anomalous.
        ## @param step_func_err (float, default 1e+1): Maximum tolerence of convergence. If fitting error is larger than this param, pass ConvergenceError to CheckResult.extra_info.
        ## @param step_func_res (float, default 2.5): Threshold value of residual for general step function, data points exceed this param will be regarded as anomalous.
        ## @param exp_decay_err (float, default 1e+1): Maximum tolerence of convergence. If fitting error is larger than this param, pass ConvergenceError to CheckResult.extra_info.
        ## @param exp_decay_res (float, default 2): Threshold value of residual for exponential function, data points exceed this param will be regarded as anomalous.
        ## @param skewness (float, default 20): Threshold value of skewness. If skewness of data distribution is larger than this param, pass Warning to CheckResult.extra_info.
        ## @param min_res (float, default 10): Absolute minimum value of residul, residuals that are smaller than this param will be masked into zero. This action is always performed before z-score normalizing the residual.
        self.thres_params = {
                "p_normality": 5e-3,
                "normal_err": 75,
                "normal_std_width": 1.5,
                "normal_std_err": 1e+1,            
                "linregress_std_err": 1e+1,
                "linregress_res": 2,
                "step_func_err": 1e+1,
                "step_func_res": 2.5,
                "exp_decay_err": 1e+1,
                "exp_decay_res": 2,
                "skewness": 20,
                "min_res": 10
        }
        self.error_code = {
                "0": "Check passed.",
                "-1": "ConvergenceError: Gaussian fitting may not converge, std_err > std_err_th.",
                "-2": "Warning: Normal distribution may have skewed, skewness > skewness_th.",
                "-3": "ConvergenceError: General erf fitting may not converge, perr > perr_th.",
                "-4": "ConvergenceError: Exponential fitting may not converge, perr > perr_th.",
                "-5": "ConvergenceError: Linear ansatz fitting may not converge, perr > perr_th.",
                "-6": "Warning: Rawdata might be oscillating, data flips sign repeatedly over mean.",
                "-7": "Info: AnomalyDetector is using boxcox method.",
                "-8": "Info: AnomalyDetector is using z normalization.",
                "-9": "Info: There are more than %d discontinuous points detected."
        }
        ## Models that can be considered by AnomalyDetector.
        ## @param gaussian (bool, default True): Gaussian (normal) distribution.
        ## @param half_gaussian (bool, default False): In development, unavailable for now.
        ## @param linear_regression (bool, default True): Linear ansatz.
        ## @param step_func (bool, default True): Generalize Heaviside step function.
        ## @param exp_decay (bool, default True): Exponential function.
        self.models = {
                "gaussian": True, 
                "half_gaussian": False, 
                "linear_regression": True, 
                "step_func": True, 
                "exp_decay": True
        }
        if len(series) < self.apply_policies["min_sample_size"]: 
            raise ValueError("number of samples {} are less than apply_policies['min_sample_size'] = {}".format(len(series), self.apply_policies["min_sample_size"]))
        if len(t) != len(series): 
            raise ValueError("shape {} does not match with shape {}.".format(len(t), len(series)))
        if self.apply_policies["info_criterion"] not in ["AIC", "BIC"]: 
            raise ValueError("Information criterion can only be 'AIC' or 'BIC'.")
    
    def _build_stats_data(self):
        statsdata, ref, IC_score = {}, {}, {}; proceed = False 

        try:    
            normality = stats_util.normaltest(self.series)
        except ValueError:
            normality = [np.inf, np.inf]     

        if normality[1] >= self.thres_params["p_normality"] and np.isfinite(normality[1]) and self.models["gaussian"]:
            if self.apply_policies["boxcox"]: 
                self.series = stats_util.boxcox(self.series, lmbda=0)
            try:
                statsdata["model"] = 'gaussian'
                statsdata["popt"], statsdata["perr"] = stats_util.gaussian_fit(self.series)
            except:
                pass
        
        if "popt" in statsdata:
            err_score = np.sum(np.square(statsdata["perr"]))
            if err_score > self.thres_params["normal_err"]: 
                proceed = True
        else:
            proceed = True

        if proceed:
            if self.apply_policies["boxcox"]:
                self.series = copy.deepcopy(self._clone_series)
            for model_id, run_token in self.models.items():
                if model_id == 'gaussian' or model_id == 'half_gaussian': continue
                if run_token is True:
                    ref[model_id] = {}
                    IC_score[model_id], ref[model_id]["popt"], ref[model_id]["perr"] = self._fitting_model(model_id)
            
            best_model = min(IC_score.items(), key=lambda x: x[1])
            if "linear_regression" in IC_score.keys():
                if np.isclose(best_model[1], IC_score["linear_regression"], atol=10, rtol=1e-2):
                    best_model = "linear_regression"
                else:
                    best_model = best_model[0]
            else:
                best_model = best_model[0]
            statsdata["popt"], statsdata["perr"] = ref[best_model]["popt"], ref[best_model]["perr"]
            if best_model == 'step_func':
                if ref[best_model]["popt"][1]-ref[best_model]["popt"][0] > 0:
                    statsdata["model"] = "increase_step_func"
                else:
                    statsdata["model"] = "decrease_step_func"
            else:
                statsdata["model"] = best_model
        return statsdata     
        
    def _fitting_model(self, model_id: str):
        if model_id == 'linear_regression':
            r_sq, intercept, slope, p_value, std_err = stats_util.linear_regression(self.t, self.series)
            linregress_y_pred = np.polyval([slope,intercept], self.t)
            if self.apply_policies["info_criterion"] == 'AIC':
                IC_score = stats_util.AIC_score(self.series, linregress_y_pred, 2)
            elif self.apply_policies["info_criterion"] == 'BIC':
                IC_score = stats_util.BIC_score(self.series, linregress_y_pred, 2)
            popt, perr = np.array([intercept, slope]), std_err
        
        elif model_id == 'step_func':
            try:
                popt, perr = stats_util.general_erf_fit(self.t, self.series)
                y_pred = stats_util.general_erf(self.t, *popt.tolist())
            except RuntimeError:
                popt = perr = np.inf * np.ones(3)
                y_pred = np.inf * np.ones(len(self.series))
            if self.apply_policies["info_criterion"] == 'AIC':
                IC_score = stats_util.AIC_score(self.series, y_pred, len(popt))
            elif self.apply_policies["info_criterion"] == 'BIC':
                IC_score = stats_util.BIC_score(self.series, y_pred, len(popt))    
    
        elif model_id == 'exp_decay':
            try:
                popt, perr = stats_util.exp_decay_fit(self.t, self.series)
                y_pred = stats_util.exp_decay(self.t, *popt.tolist())
            except RuntimeError:
                popt = perr = np.inf * np.ones(2)
                y_pred = np.inf * np.ones(len(self.series))
            if self.apply_policies["info_criterion"] == 'AIC':
                IC_score = stats_util.AIC_score(self.series, y_pred, len(popt))
            elif self.apply_policies["info_criterion"] == 'BIC':
                IC_score = stats_util.BIC_score(self.series, y_pred, len(popt))  
                
        return IC_score, popt, perr   
    
    def check(self) -> object:
        """!
        
        @returns check_result (CheckResult): 
        """
        statsdata = self._build_stats_data()
        model_id = statsdata["model"]
        anomalous_t, anomalous_data, res, msgs = [], [], [], []
            
        if model_id == 'gaussian' or model_id == 'flat_histo': 
            if statsdata["perr"][2] > self.thres_params["normal_std_err"]:
                msgs.append(self.error_code["-1"])
            # Get anomalous data
            norm = np.std(self.series)
            mean_centered_series = self.series - np.mean(self.series)
            mean_centered_series[np.where(abs(mean_centered_series) < self.thres_params["min_res"])] = 0
            z_normalized_series = mean_centered_series / norm
            anomalous_idx = abs(z_normalized_series) > self.thres_params["normal_std_width"]
            if np.count_nonzero(anomalous_idx) > 0:
                anomalous_data = self.series[anomalous_idx]
                anomalous_t = self._clone_t[anomalous_idx]
                res = abs(z_normalized_series)[anomalous_idx]
            histo_x, histo_y = stats_util.get_histogram(self.series)
            if abs(stats_util.skew(histo_y)) > self.thres_params["skewness"]:
                msgs.append(self.error_code["-2"])
                    
        elif model_id == "increase_step_func":
            err_score = np.sum(np.square(statsdata["perr"]))
            if err_score > self.thres_params["step_func_err"]:
                msgs.append(self.error_code["-3"]) 
            res = stats_util.fitting_residual(self.t, self.series, stats_util.general_erf, statsdata["popt"],
                                              mask_min=self.thres_params["min_res"],
                                              standardized=self.apply_policies["z_normalization"])
            anomalous_t = self._clone_t[res > self.thres_params["step_func_res"]]
            anomalous_data = self.series[res > self.thres_params["step_func_res"]]                                      
            res = res[res > self.thres_params["step_func_res"]]
            
        elif model_id == "decrease_step_func":
            err_score = np.sum(np.square(statsdata["perr"]))
            if err_score > self.thres_params["step_func_err"]:    
                res = stats_util.fitting_residual(self.t, self.series, stats_util.general_erf, statsdata["popt"],
                                                  mask_min=self.thres_params["min_res"],
                                                  standardized=self.apply_policies["z_normalization"])
                anomalous_t = self._clone_t[res > self.thres_params["step_func_res"]]
                anomalous_data = self.series[res > self.thres_params["step_func_res"]]
                res = res[res > self.thres_params["step_func_res"]]
                msgs.append(self.error_code["-3"])
            else:   
                anomalous_idx = np.where(self.t > statsdata["popt"][2])[0]
                if len(anomalous_idx) != 0 and (statsdata["popt"][0]-statsdata["popt"][1]) > self.thres_params["min_res"]: 
                    anomalous_t = self._clone_t[anomalous_idx]
                    anomalous_data = self.series[anomalous_idx]
                    res = (statsdata["popt"][0]-statsdata["popt"][1]) * np.ones(len(anomalous_idx))
# =============================================================================
#         elif model_id == 'three_stair':
#             err_score = np.sum(np.square(statsdata[key]["perr"]))
#             if err_score > self.thres_params["step_func_err"]:
#                 msgs.append(self.dyError.getErrorText(16))
#             t = np.arange(1, len(statsdata[key]["series"])+1) 
#             res = stats_util.fitting_residual(t, statsdata[key]["series"], stats_util.three_stair_erf, statsdata[key]["popt"])
#             anomalous_data = statsdata[key]["series"][res > self.thres_params["step_func_res"]]                  
# =============================================================================
            
        elif model_id == 'exp_decay':
            err_score = np.sum(np.square(statsdata["perr"]))
            if err_score > self.thres_params["exp_decay_err"]:
                msgs.append(self.error_code["-4"])
            res = stats_util.fitting_residual(self.t, self.series, stats_util.exp_decay, statsdata["popt"],
                                              mask_min=self.thres_params["min_res"],
                                              standardized=self.apply_policies["z_normalization"])
            anomalous_t = self._clone_t[res > self.thres_params["exp_decay_res"]]
            anomalous_data = self.series[res > self.thres_params["exp_decay_res"]] 
            res = res[res > self.thres_params["exp_decay_res"]]                   
             
        elif model_id == 'linear_regression':
            if statsdata["perr"] > self.thres_params["linregress_std_err"]:
                msgs.append(self.error_code["-5"])
            func = lambda x, a, b: a + b*x 
            res = stats_util.fitting_residual(self.t, self.series, func, statsdata["popt"],
                                              mask_min=self.thres_params["min_res"],
                                              standardized=self.apply_policies["z_normalization"])
            anomalous_t = self._clone_t[res > self.thres_params["linregress_res"]]
            anomalous_data = self.series[res > self.thres_params["linregress_res"]]
            res = res[res > self.thres_params["linregress_res"]]                    
                
        # Extra info
        if stats_util.is_oscillating(self.series): 
            msgs.append(self.error_code["-6"])
        if self.apply_policies["boxcox"]: 
            msgs.append(self.error_code["-7"])
        if self.apply_policies["z_normalization"]:
            msgs.append(self.error_code["-8"])    
        discontinuity = len(stats_util.discontinuous_idx(self.series))
        if discontinuity > 0:
            msgs.append(self.error_code["-9"] %discontinuity)
                
        if len(anomalous_data) == 0: 
            self.check_failed = False
            msgs.append(self.error_code["0"])
        
        if isinstance(res, np.ndarray): res = res.tolist()
        check_result = CheckResult(
                model=statsdata["model"],
                popt=statsdata["popt"].tolist(),
                perr=statsdata["perr"].tolist(),
                anomalous_data=list(zip(anomalous_t, anomalous_data)),
                residual=res,
                extra_info=msgs
        )
        return check_result

class CheckResult(dict):
    
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return '\n'.join([k.rjust(m) + ': ' + repr(v)
                              for k, v in sorted(self.items())])
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        return list(self.keys())

class AnomalousData(CheckResult):
    def __init__(self):
        super(AnomalousData, self).__init__()