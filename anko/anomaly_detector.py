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
        ## Policies for AnomalyDetector to follow.
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
        self.check_failed = True       
        self.thres_params = {
                "p_normality": 1e-3,
                "skewness": 20,
                "normal_std_width": 3,
                "normal_std_err": 1,
                "normal_err": 1e+1,
                "linregress_slope": 0.1,
                "linregress_std_err": 1e+1,
                "linregress_res": 1,
                "step_func_err": 1e+1,
                "step_func_res": 3,
                "exp_decay_err": 1e+1,
                "exp_decay_res": 2,
                "linearity": 1e-2,
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
        self.models = {
                "gaussian": True, 
                "half_gaussian": True, 
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
        
        if self.apply_policies["boxcox"]: 
            self.series = stats_util.boxcox(self.series, lmbda=0)          
        histo_x, histo_y = stats_util.get_histogram(self.series)

        try:    
            normality = stats_util.normaltest(histo_y)
        except ValueError:
            normality = [np.inf, np.inf]     
       
        if normality[1] >= self.thres_params["p_normality"] and np.isfinite(normality[1]) and self.models["gaussian"]:
            try:
                statsdata["model"] = 'gaussian'
                statsdata["popt"], statsdata["perr"] = stats_util.gaussian_fit(self.series)
            except:
                pass
    
        if "popt" in ref:
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
        
    def _fitting_model(self, model_id):
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
        
        @returns statsdata (dict): 
        """
        statsdata = self._build_stats_data()
        model_id = statsdata["model"]
        anomalous_t, anomalous_data, msgs = [], [], []
            
        if model_id == 'gaussian' or model_id == 'flat_histo': 
            histo_x, histo_y = stats_util.get_histogram(self.series)
            if statsdata["perr"][2] > self.thres_params["normal_std_err"]:
                msgs.append(self.error_code["-1"])
            # Get anomalous data
            std_th = max(self.thres_params["normal_std_width"] * statsdata["popt"][2], self.thres_params["min_res"])
            allowed_domain = (statsdata["popt"][1]-std_th, statsdata["popt"][1]+std_th)
            anomalous_idx = np.where(np.logical_or(histo_x<=allowed_domain[0], histo_x>=allowed_domain[1]))[0] 
            if len(anomalous_idx) != 0: 
                if self.apply_policies["boxcox"]:
                    histo_x, histo_y = stats_util.get_histogram(self._clone_series)
                anomalous_data = histo_x[anomalous_idx]
                anomalous_t = self._clone_t[np.where(np.in1d(anomalous_data, self.series))]
                res = abs(anomalous_data-statsdata["popt"][1])
                if self.apply_policies["z_normalization"]: res /= statsdata["popt"][2]
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
                if len(anomalous_idx) != 0: 
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

        check_result = CheckResult(
                model=statsdata["model"],
                popt=statsdata["popt"].tolist(),
                perr=statsdata["perr"].tolist(),
                anomalous_data=list(zip(anomalous_t, anomalous_data)),
                residual=res.tolist(),
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