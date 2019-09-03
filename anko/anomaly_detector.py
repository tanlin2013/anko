import numpy as np
from anko import stats_util
import copy
# TODO: add params and returns type in func entry

class AnomalyDetector:
    
    def __init__(self, t, series):
        if isinstance(t, list): t = np.array(t)
        if isinstance(series, list): series = np.array(series)
        self.t = t
        self.t_scaleless = np.arange(1, len(series)+1)
        self.series = series
        self.clone_series = copy.deepcopy(series)
        self.apply_boxcox = False
        self.apply_z_normalization = False
        self.check_failed = True
        self.info_criterion = 'AIC'
        self.min_sample_size = 10
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
                "-7": "Info: build_statsdata is using boxcox method.",
                "-8": "Info: build_statsdata is using z normalization.",
                "-9": "Info: There are more than %d discontinuous points detected."
        }
        self.models = {
                "gaussian": True, 
                "half_gaussian": True, 
                "linear_regression": True, 
                "step_func": True, 
                "exp_decay": True
        }
        if len(series) < self.min_sample_size: raise ValueError("number of samples are less than AnomalyDetector.min_sample_size[%d]" %self.min_sample_size)
        if len(t) != len(series): raise ValueError("shape %d does not match with shape %d." %len(t) %len(series))
        if self.info_criterion not in ["AIC", "BIC"]: raise ValueError("Information criterion can only be 'AIC' or 'BIC'.")
    
    def _build_stats_data(self):
        statsdata, ref, IC_score = {}, {}, {}; lmbda = 1; proceed = False           
        histo_x, histo_y = stats_util.get_histogram(self.series)
        
        try:    
            normality = stats_util.normaltest(histo_y)
        except ValueError:
            normality = [np.inf, np.inf]     
        
        if normality[1] >= self.thres_params["p_normality"] and np.isfinite(normality[1]):   
# =============================================================================
#             # Check the magnitude of input data is not too large, else use log-boxcox transformation.
#             if np.mean(self.series) > self.thres_params["max_mag"]: 
#                 lmbda = 0; self.using_boxcox = True               
# =============================================================================
            try:
                # Perform a Gaussian fit analysis, where y(x) is the histogram built from series
                statsdata["model"] = 'gaussian'
                statsdata["popt"], statsdata["perr"] = stats_util.gaussian_fit(self.series, lmbda)
            except TypeError:
                # Caused by the lake of minimum samples 
                # Treat as flat histogram, appending mode as mean and np.inf as std. All perrs are set to be 0.
                if stats_util.data_is_linear(histo_x, histo_y, self.thres_params["linearity"]):
                    statsdata["model"] = 'flat_histo'
                    statsdata["popt"], statsdata["perr"] = stats_util.flat_histogram(self.series)
                else:
                    pass
            except RuntimeError:
                pass
        
        if "popt" in ref:
            err_score = np.sum(np.square(statsdata["perr"]))
            if err_score > self.thres_params["normal_err"]: 
                proceed = True
        else:
            proceed = True
        
        if proceed:
            for model_id, run_token in self.models.items():
                if model_id == 'gaussian' or model_id == 'half_gaussian': continue
                if run_token is True:
                    ref[model_id] = {}
                    IC_score[model_id], ref[model_id]["popt"], ref[model_id]["perr"] = self._fitting_model(model_id)
            
            best_model = min(IC_score.items(), key=lambda x: x[1])
            if "linear_regression" in IC_score.keys():
                if np.isclose(best_model[1],IC_score["linear_regression"],rtol=1e-2):
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
            r_sq, intercept, slope, p_value, std_err = stats_util.linear_regression(self.t_scaleless, self.series)
            linregress_y_pred = np.polyval([slope,intercept], self.t_scaleless)
            if self.info_criterion == 'AIC':
                IC_score = stats_util.AIC_score(self.series, linregress_y_pred, 2)
            elif self.info_criterion == 'BIC':
                IC_score = stats_util.BIC_score(self.series, linregress_y_pred, 2)
            popt, perr = [intercept, slope], std_err
        
        elif model_id == 'step_func':
            try:
                popt, perr = stats_util.general_erf_fit(self.t_scaleless, self.series)
                y_pred = stats_util.general_erf(self.t_scaleless, *popt.tolist())
            except RuntimeError:
                popt = perr = np.inf * np.ones(3)
                y_pred = np.inf * np.ones(len(self.series))
            if self.info_criterion == 'AIC':
                IC_score = stats_util.AIC_score(self.series, y_pred, len(popt))
            elif self.info_criterion == 'BIC':
                IC_score = stats_util.BIC_score(self.series, y_pred, len(popt))    
    
        elif model_id == 'exp_decay':
            try:
                popt, perr = stats_util.exp_decay_fit(self.t_scaleless, self.series)
                y_pred = stats_util.exp_decay(self.t_scaleless, *popt.tolist())
            except RuntimeError:
                popt = perr = np.inf * np.ones(2)
                y_pred = np.inf * np.ones(len(self.series))
            if self.info_criterion == 'AIC':
                IC_score = stats_util.AIC_score(self.series, y_pred, len(popt))
            elif self.info_criterion == 'BIC':
                IC_score = stats_util.BIC_score(self.series, y_pred, len(popt))  
                
        return IC_score, popt, perr   
    
    def check(self):
        statsdata = self._build_stats_data()
        model_id = statsdata["model"]
        anomalous_t, anomalous_data, msgs = [], [], []
            
        if model_id == 'gaussian' or model_id == 'flat_histo': 
            histo_x, histo_y = stats_util.get_histogram(self.series)
            if statsdata["perr"][2] > self.thres_params["normal_std_err"]:
                msgs.append(self.error_code["-1"])
            if self.using_boxcox: 
                histo_x = stats_util.boxcox(histo_x, lmbda=0)
            # Large error exception: if caused by extremely flat histogram  
            if stats_util.data_is_linear(histo_x, histo_y, self.thres_params["linearity"]):
                self.check_failed = False
                return
            # Get anomalous data
            std_th = max(self.thres_params["normal_std_width"] * statsdata["popt"][2], self.thres_params["min_res"])
            allowed_domain = (statsdata["popt"][1]-std_th, statsdata["popt"][1]+std_th)
            anomalous_idx = np.where(np.logical_or(histo_x<=allowed_domain[0], histo_x>=allowed_domain[1]))[0] 
            if len(anomalous_idx) != 0: 
                anomalous_data = histo_x[anomalous_idx]
                if self.using_boxcox: anomalous_data = np.exp(anomalous_data)
                anomalous_t = self.t[np.where(np.in1d(anomalous_data, self.series))]
            if abs(stats_util.skew(histo_y)) > self.thres_params["skewness"]:
                msgs.append(self.error_code["-2"])
                    
        elif model_id == "increase_step_func":
            err_score = np.sum(np.square(statsdata["perr"]))
            if err_score > self.thres_params["step_func_err"]:
                msgs.append(self.error_code["-3"]) 
            res = stats_util.fitting_residual(self.t_scaleless, self.series, stats_util.general_erf, statsdata["popt"],
                                              standardized=self.apply_z_normalization)
            anomalous_t = self.t[res > self.thres_params["step_func_res"]]
            anomalous_data = self.clone_series[res > self.thres_params["step_func_res"]]                                      
          
        elif model_id == "decrease_step_func":
            err_score = np.sum(np.square(statsdata["perr"]))
            if err_score > self.thres_params["step_func_err"]:    
                res = stats_util.fitting_residual(self.t_scaleless, self.series, stats_util.general_erf, statsdata["popt"],
                                                  standardized=self.apply_z_normalization)
                anomalous_t = self.t[res > self.thres_params["step_func_res"]]
                anomalous_data = self.clone_series[res > self.thres_params["step_func_res"]]
                msgs.append(self.error_code["-3"])
            else:   
                anomalous_idx = np.where(self.t_scaleless > statsdata["popt"][2])[0]
                if len(anomalous_idx) != 0: 
                    anomalous_t = self.t[anomalous_idx]
                    anomalous_data = self.clone_series[anomalous_idx]
            
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
            #err_score = np.sum(np.square(statsdata[key]["perr"]))
            err_score = statsdata["perr"][1]
            if err_score > self.thres_params["exp_decay_err"]:
                msgs.append(self.error_code["-4"])
            res = stats_util.fitting_residual(self.t_scaleless, self.series, stats_util.exp_decay, statsdata["popt"],
                                              standardized=self.apply_z_normalization)
            anomalous_t = self.t[res > self.thres_params["exp_decay_res"]]
            anomalous_data = self.clone_series[res > self.thres_params["exp_decay_res"]]                   
             
        elif model_id == 'linear_regression':
            if statsdata["perr"] > self.thres_params["linregress_std_err"]:
                msgs.append(self.error_code["-5"])
            func = lambda x, a, b: a + b*x 
            res = stats_util.fitting_residual(self.t_scaleless, self.series, func, statsdata["popt"],
                                              standardized=self.apply_z_normalization)
            anomalous_t = self.t[res > self.thres_params["linregress_res"]]
            anomalous_data = self.clone_series[res > self.thres_params["linregress_res"]]                  
                
        # Extra info
        if stats_util.is_oscillating(self.series): 
            msgs.append(self.error_code["-6"])
        if self.using_boxcox: 
            msgs.append(self.error_code["-7"])
        if self.using_z_normalization:
            msgs.append(self.error_code["-8"])    
        discontinuity = len(stats_util.discontinuous_idx(self.series))
        if discontinuity > 0:
            msgs.append(self.error_code["-9"] %discontinuity)
                
        if len(anomalous_data) == 0: 
            self.check_failed = False
            msgs.append(self.error_code["0"])
            statsdata["anomalous_data"] = []
        else:
            statsdata["anomalous_data"] = list(zip(anomalous_t, anomalous_data))
        statsdata["extra_info"] = msgs
        return statsdata
