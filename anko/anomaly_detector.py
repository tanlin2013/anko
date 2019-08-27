import numpy as np
import stats_util
import copy
# TODO: remove tmp comment block

class AnomalyDetector:
    
    def __init__(self, t, series):
        if isinstance(t, list): t = np.array(t)
        if isinstance(series, list): series = np.array(series)
        self.t = t
        self.t_scaleless = np.arange(1, len(series)+1)
        self.series = series
        self.clone_series = copy.deepcopy(series)
        self.using_boxcox = False
        self.using_z_normalization = False
        self.check_failed = True
        self.info_criterion = 'AIC'
        self.thres_params = {
            "p_normality": 1e-3,
            "skewness": 20,
            "normal_std_width": 3,
            "normal_std_err": 1,
            "normal_err": 1e+1,
            "max_mag": 1e+3,
            "linregress_slope": 0.1,
            "linregress_std_err": 1e+1,
            "linregress_res": 3,
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
                "-8": "Info: build_statsdata is using z normalization." 
        }
    
    def _build_stats_model(self):
        ref = {}; IC_score = {}; lmbda = 1; proceed = False           
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
                ref["model"] = 'gaussian'
                ref["popt"], ref["perr"] = stats_util.gaussian_fit(self.series, lmbda)
            except TypeError:
                # Caused by the lake of minimum samples 
                # Treat as flat histogram, appending mode as mean and np.inf as std. All perrs are set to be 0.
                if stats_util.data_is_linear(histo_x, histo_y, self.thres_params["linearity"]):
                    ref["model"] = 'flat_histo'
                    ref["popt"], ref["perr"] = stats_util.flat_histogram(self.series)
                else:
                    pass
            except RuntimeError:
                pass
        
        if "popt" in ref:
            err_score = np.sum(np.square(ref["perr"]))
            if err_score > self.thres_params["normal_err"]: 
                proceed = True
        else:
            proceed = True
        
        if proceed:
# =============================================================================
#             # Check the magnitude of input data is not too large, else perform z normalization
#             if np.mean(self.series) > self.thres_params["max_mag"]: 
#                 self.series = stats_util.z_normalization(self.series)
#                 self.using_z_normalization = True
#                 self.using_boxcox = False
# =============================================================================
            try:
                erf_popt, erf_perr = stats_util.general_erf_fit(self.t_idx, self.series)
                erf_y_pred = stats_util.general_erf(self.t_idx, *erf_popt.tolist())
            except RuntimeError:
                erf_popt = erf_perr = np.inf * np.ones(3)
                erf_y_pred = np.inf * np.ones(len(self.series))
                
# =============================================================================
#             ThrS_popt, ThrS_perr = stats_util.general_erf_fit(t, series, three_stair=True)
#             ThrS_y_pred = stats_util.three_stair_erf(t, *ThrS_popt.tolist())
# =============================================================================
            
            try:
                exp_popt, exp_perr = stats_util.exp_decay_fit(self.t_idx, self.series)
                exp_y_pred = stats_util.exp_decay(self.t_idx, *exp_popt.tolist())
            except RuntimeError:
                exp_popt = exp_perr = np.inf * np.ones(2)
                exp_y_pred = np.inf * np.ones(len(self.series))
            
            r_sq, intercept, slope, p_value, std_err = stats_util.linear_regression(self.t_idx, self.series)
            linregress_y_pred = np.polyval([slope,intercept], self.t_idx)
            
            if self.info_criterion == 'AIC':
                IC_score["step_func"] = stats_util.AIC_score(self.series, erf_y_pred, len(erf_popt))
# =============================================================================
#                 IC_score["three_stair"] = stats_util.AIC_score(series, ThrS_y_pred, len(ThrS_popt))
# =============================================================================
                IC_score["exp_decay"] = stats_util.AIC_score(self.series, exp_y_pred, len(exp_popt))
                IC_score["flat_series"] = stats_util.AIC_score(self.series, linregress_y_pred, 2)
            elif self.info_criterion == 'BIC':
                IC_score["step_func"] = stats_util.BIC_score(self.series, erf_y_pred, len(erf_popt))
# =============================================================================
#                 IC_score["three_stair"] = stats_util.BIC_score(series, ThrS_y_pred, len(ThrS_popt))
# =============================================================================
                IC_score["exp_decay"] = stats_util.BIC_score(self.series, exp_y_pred, len(exp_popt))
                IC_score["flat_series"] = stats_util.BIC_score(self.series, linregress_y_pred, 2)
            best_model = min(IC_score.items(), key=lambda x: x[1])
            if np.isclose(best_model[1],IC_score["flat_series"],rtol=1e-2):
                best_model = "flat_series"
            else:
                best_model = best_model[0]
            if best_model == 'step_func':
                if erf_popt[1]-erf_popt[0] > 0:
                    ref["model"] = "increase_step_func"
                else:
                    ref["model"] = "decrease_step_func"
            else:
                ref["model"] = best_model
            if best_model == 'step_func':
                ref["popt"], ref["perr"] = erf_popt, erf_perr
# =============================================================================
#             elif best_model == 'three_stair':
#                 ref["popt"], ref["perr"] = ThrS_popt, ThrS_perr
# =============================================================================
            elif best_model == 'exp_decay':
                ref["popt"], ref["perr"] = exp_popt, exp_perr
            elif best_model == 'flat_series':
                ref["popt"], ref["perr"] = [intercept, slope], std_err    
        return ref
    
    def check_data(self):
        statsdata = self._build_stats_model()
        model_id = statsdata["model"]
        anomalous_data = "None"; msgs = []
            
        if model_id == 'gaussian' or model_id == 'flat_histo': 
            histo_x, histo_y = stats_util.get_histogram(self.clone_series)
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
            res = stats_util.fitting_residual(self.t_idx, self.series, stats_util.general_erf, statsdata["popt"],
                                              mask_as_zero=True, min_res=self.thres_params["min_res"])
            anomalous_t = self.t[res > self.thres_params["step_func_res"]]
            anomalous_data = self.clone_series[res > self.thres_params["step_func_res"]]                                      
          
        elif model_id == "decrease_step_func":
            err_score = np.sum(np.square(statsdata["perr"]))
            if err_score > self.thres_params["step_func_err"]:    
                res = stats_util.fitting_residual(self.t_idx, self.series, stats_util.general_erf, statsdata["popt"],
                                                  mask_as_zero=True, min_res=self.thres_params["min_res"])
                anomalous_t = self.t[res > self.thres_params["step_func_res"]]
                anomalous_data = self.series[res > self.thres_params["step_func_res"]]
                msgs.append(self.error_code["-3"])
            else:   
                anomalous_idx = np.where(self.t_idx > statsdata["popt"][2])[0]
                if len(anomalous_idx) != 0 and (statsdata["popt"][0] - statsdata["popt"][1]) > self.thres_params["min_res"]: 
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
            res = stats_util.fitting_residual(self.t_idx, self.series, stats_util.exp_decay, statsdata["popt"],
                                              mask_as_zero=True, min_res=self.thres_params["min_res"])
            anomalous_t = self.t[res > self.thres_params["exp_decay_res"]]
            anomalous_data = self.clone_series[res > self.thres_params["exp_decay_res"]]                   
             
        elif model_id == 'flat_series':
            if statsdata["perr"] > self.thres_params["linregress_std_err"]:
                msgs.append(self.error_code["-5"])
            func = lambda x, a, b: a + b*x 
            res = stats_util.fitting_residual(self.t_idx, self.series, func, statsdata["popt"],
                                              mask_as_zero=True, min_res=self.thres_params["min_res"])
            anomalous_t = self.t[res > self.thres_params["linregress_res"]]
            anomalous_data = self.clone_series[res > self.thres_params["linregress_res"]]                  
                
        # Extra info
        if stats_util.is_oscillating(self.series): 
            msgs.append(self.error_code["-6"])
        if self.using_boxcox: 
            msgs.append("build_statsdata is using boxcox method.")
        if self.using_z_normalization:
            msgs.append("build_statsdata is using z normalization.")    
        discontinuity = len(stats_util.discontinuous_idx(self.series))
        if discontinuity > 0:
            msgs.append("There are more than %d discontinuous points detected." %discontinuity)
                
        if anomalous_data == "None" or anomalous_data.size == 0: 
            self.check_failed = False
            msgs.append(self.error_code["0"])
        
        statsdata["anomalous_data"] = list(zip(anomalous_t, anomalous_data))
        statsdata["extra_info"] = msgs
        return statsdata