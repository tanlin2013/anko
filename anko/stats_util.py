import numpy as np
import collections, itertools
from scipy.stats import boxcox, linregress, skew, normaltest
from scipy.optimize import curve_fit, differential_evolution
# TODO: handle typing for returning tuple

def get_histogram(x: np.ndarray, sort_histo: bool=False):
    """!
    Return the corresponding histogram of the data x.
    
    @param x (numpy.ndarray): One-dimensional array of data.
    @param sort_histo (bool, optional): If True return the sorted histogram.
        
    @returns keys: Set of data x (no duplicate).
    @returns vals: Number of appearance for each key in keys. 
    """
    counter = collections.Counter(x)
    keys = np.fromiter(counter.keys(), dtype=float)
    vals = np.fromiter(counter.values(), dtype=float)
    if sort_histo:
        keys_idx = np.argsort(keys)
        keys = keys[keys_idx]
        vals = vals[keys_idx]
    return keys, vals
    
def normal_distr(x: np.ndarray, a: float, x0: float, sigma: float) -> np.ndarray:
    """!
    Calculate normal distribution of input array x.
    
    \f{equation*}{
            f(x) = a \exp\left(-\frac{\left(x-x_0\right)^2}{2\sigma^2}\right).
    \f}
    
    @param x (numpy.ndarray): Input values.
    @param a (float): Overall normalization constant.
    @param x0 (float): Mean.
    @param sigma (float): Standard deviation.
            
    @returns out (numpy.ndarray): Output array. 
    """
    return a * np.exp(-(x-x0)**2/(2*sigma**2))

def gaussian_fit(x: np.ndarray, sort_histo: bool=False, half: str=None, maxfev: int=2000, bounds=[0,1e+6]):
    """!
    Fitting the Gaussian (normal) distribution for input data x.
    
    @param x (numpy.ndarray): Input values.   
    @param sort_histo (bool, optional): If True use the sorted histogram.
    @param half (str, optional):  
    @param maxfev (int, optional): Maximum step of fitting iteration.
    @param bounds (list[float, float], optional): 
    
    @returns popt (numpy.ndarray): Estimate value of a, x0 and sigma of Gaussian distribution.
    @returns perr (numpy.ndarray): Error of popt. Defined by the square of diagonal element of covariance matrix.
    """
    keys, vals = get_histogram(x, sort_histo)
    a_sg = max(vals) * 0.9
    m_sg = np.mean(x)
    std_sg = np.std(x)
    if half == 'left':
        popt, pcov = curve_fit(left_half_normal_distr,keys,vals,p0=[a_sg,m_sg,std_sg],maxfev=maxfev,bounds=bounds)
    elif half == 'right':
        popt, pcov = curve_fit(right_half_normal_distr,keys,vals,p0=[a_sg,m_sg,std_sg],maxfev=maxfev,bounds=bounds)
    else:
        popt, pcov = curve_fit(normal_distr,keys,vals,p0=[a_sg,m_sg,std_sg],maxfev=maxfev,bounds=bounds)
    perr = np.sqrt(np.diag(pcov))
    return popt, perr

def left_half_normal_distr(x: np.ndarray, a: float, x0: float, sigma: float) -> np.ndarray:
    """!
    Calculate left-side half normal distribution of input array x.
    
    @param x (numpy.ndarray): Input values.
    @param a (float): Overall normalization constant.
    @param x0 (float): Mean.
    @param sigma (float): Standard deviation.
    
    @returns out (numpy.ndarray): Output array. 
    """
    return a * np.multiply(np.exp(-(x-x0)**2/(2*sigma**2)), -1*np.heaviside(x-x0, 0.5))

def right_half_normal_distr(x: np.ndarray, a: float, x0: float, sigma: float) -> np.ndarray:
    """!
    Calculate right-side half normal distribution of input array x.
    
    @param x (numpy.ndarray): Input values.
    @param a (float): Overall normalization constant.
    @param x0 (float): Mean.
    @param sigma (float): Standard deviation.
    
    @returns out (numpy.ndarray): Output array. 
    """
    return a * np.multiply(np.exp(-(x-x0)**2/(2*sigma**2)), np.heaviside(x-x0, 0.5))

def flat_histogram(x: np.ndarray):
    """!
    Deprecating... \n
    Manually assign parameters of Gaussian distrinution if the given histogram is too flat. \n
    In this senario the histogram of data is regarded as a local segment of a larger normal-distribution-like histogram, \n
    with standard deviation which exceeds the current consideration of domain. 
    
    Parameters of Gaussian distribution are assigned as following:
        1. Number of appearance of mode as normalization constant, a. 
        2. Mode of data x as mean, x0. 
        3. Standard deviation is set to infinity (numpy.inf).
    
    @param x (numpy.ndarray): Input values.
    
    @returns popt (numpy.ndarray): Assigned values for Gaussian distribution. 
    @returns perr (numpy.ndarray): Errors are set to zero.
    """
    counter = collections.Counter(x)
    mode = counter.most_common()
    popt = np.array([mode[0][1], mode[0][0], np.inf])
    perr = np.array([0, 0, 0])
    return popt, perr

def linear_regression(x: np.ndarray, y: np.ndarray):
    """!
    Fitting linear ansatz for input data (x, y). 
    
    \f{equation*}{
            f(x) = intercept + slope \times x.
    \f}
    
    @param x (numpy.ndarray): x coordinate of input data points. 
    @param y (numpy.ndarray): y coordinate of input data points.
      
    @returns r_sq (float): Coefficient of determination.
    @returns intercept (float): Intercept of the regression line.
    @returns slope (float): Slope of the regression line.
    @returns p_value (float): Two-sided p-value for a hypothesis test whose null hypothesis is that the slope is zero, \n
            using Wald Test with t-distribution of the test statistic.
    @returns std_err (float): Standard error of the estimated gradient.
    """
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    r_sq = r_value**2
    return r_sq, intercept, slope, p_value, std_err

def data_is_linear(x: np.ndarray, y: np.ndarray, std_err_th: float=1e-2) -> bool:
    """!
    Check whether the data (x, y) is linear under the given tolerance. \n
    This will perform a linear regression fitting.
    
    @param x (numpy.ndarray): x coordinate of input data points. 
    @param y (numpy.ndarray): y coordinate of input data points.
    @param std_err_th (float, optional): Threshold value of std_err.
    
    @returns out (bool): Return Ture if data is flat, else return False.
    """
    r_sq, intercept, slope, p_value, std_err = linear_regression(x, y)
    if p_value == 1 or std_err < std_err_th:
        return True
    else:
        return False

# TODO: rename to sgn
def general_sgn(x: np.ndarray, a: float, b: float, x0: float) -> np.ndarray:
    """!
    Calculate the generalize sign function of input array x.
    
    \f{equation*}{
           f(x) = 
               \begin{cases}
                   a, & x < x_0, \\
                   \frac{a+b}{2}, & x = x_0, \\
                   b, & x > x_0.
               \end{cases}
    \f}
    
    @param x (numpy.ndarray): Input values.
    @param a (float): Value of first stair. 
    @param b (float): Value of second stair.
    @param x0 (float): Location of the cliff.
    
    @returns out (numpy.ndarray): Output array.
    """
    return (b-a)/2 * np.sign(x-x0) + (a+b)/2

def three_stair_sgn(x: np.ndarray, c0: float, c1: float, c2: float, x1: float, x2: float) -> np.ndarray:
    """!
    Calculate the generalize sign function with three stairs for input array x.
    
    \f{equation*}{
           f(x) = 
               \begin{cases}
                   c_0, & x < x_1, \\
                   \frac{c_0+c_1}{2}, & x = x_1, \\
                   c_1, & x_1< x < x_2, \\
                   \frac{c_1+c_2}{2}, & x = x_2, \\
                   c_2, & x > x_2.
               \end{cases}
    \f}
    
    @param x (numpy.ndarray): Input values.
    @param c0 (float): Value of first stair. 
    @param c1 (float): Value of second stair. 
    @param c2 (float): Value of third stair. 
    @param x1 (float): Location of the first cliff.
    @param x2 (float): Location of the second cliff.
    
    @returns out (numpy.ndarray): Output array.
    """
    return c0*np.sign(x-x1) + c1*np.sign(x-x2) + c2
    
def general_sgn_fit(x: np.ndarray, y: np.ndarray, three_stair: bool=False, maxfev: int=2000, bounds=[0,1e+6]):
    """!
    Fitting generalize sign function for input data (x, y).
    
    @param x (numpy.ndarray): x coordinate of input data points. 
    @param y (numpy.ndarray): y coordinate of input data points. 
    @param three_stair (bool): If True, employing three stair error function for fitting.
    @param maxfev (int): Maximum step of fitting iteration.
    @param bounds (list[float]): 
    
    @returns popt (numpy.ndarray):
    @returns perr (numpy.ndarray):
    """
    if three_stair:
        a_sg = y[0]; b_sg = y[int(len(y)/2)]; c_sg = y[-1]
        c0_sg = (b_sg-a_sg)/2; c1_sg = (c_sg-b_sg)/2; c2_sg = (a_sg + c_sg)/2
        x1_sg, x2_sg = np.sort(np.diff(y))[::-1][:2]     
        popt, pcov = curve_fit(three_stair_sgn,x,y,p0=[c0_sg,c1_sg,c2_sg,x1_sg,x2_sg],maxfev=maxfev,bounds=bounds)
        perr = np.sqrt(np.diag(pcov))
    else:
        a_sg = y[0]; b_sg = y[-1]; x0_sg = x[np.argmax(np.diff(y))]
        popt, pcov = curve_fit(general_sgn,x,y,p0=[a_sg,b_sg,x0_sg],maxfev=maxfev,bounds=bounds)
        perr = np.sqrt(np.diag(pcov))
# TODO: consider to use scipy.optimize.differential_evolution
# =============================================================================
#         a_sg = y[0]; b_sg = y[-1]; x0_sg = x[np.argmax(np.diff(y))]
#         width = (max(y)-min(y))/10
#         optimize_result = differential_evolution(lambda p: np.sum((general_sgn(x, *p) - y)**2), [[a_sg-width,a_sg+width], [b_sg-width,b_sg+width], [x0_sg-5,x0_sg+5]], tol=1e-6)
#         popt, perr = optimize_result.x, optimize_result.fun
#         print(optimize_result.x, optimize_result.fun)
# =============================================================================
    return popt, perr    

def exp_decay(x: np.ndarray, a: float, alpha: float) -> np.ndarray:
    """!
    Calculate the exponential function of input array x. Note that domain of x >= 0.
    
    \f{equation*}{
           f(x) = a\exp\left(-\alpha x\right). 
    \f}
    
    @param x (numpy.ndarray): Input values.
    @param a (float): Overall normalized constant.
    @param alpha (float): Decay rate of exponential function. Please note that \f$ \alpha \f$ can be negative,
        and should be carefully utilized. 
        
    @returns out: Output array.
    """
    if len(np.where(np.array(x) < 0)[0]) != 0:
        raise ValueError("Domain of exp(-x) is restricted in x > 0.")
    return a * np.exp(-alpha*x)
    
def exp_decay_fit(x: np.ndarray, y: np.ndarray, mode: str='log-linregress', maxfev: int=2000, bounds=[-1e-6,1e+6]):
    """!
    
    @param x (numpy.ndarray): x coordinate of input data points.
    @param y (numpy.ndarray): y coordinate of input data points.
    @param mode (str, optional): If mode is 'log-linregress', underlying algorithm will perform linear regression in \f$ \log(x)-\log(y) \f$ scale,
        else brutal force 
    @param maxfev (int):    
    @param bounds (list[float,float]):
        
    @returns popt (numpy.ndarray):
    @returns perr (numpy.ndarray):
    """
    if mode == 'log-linregress':
        r_sq, intercept, slope, p_value, std_err = linear_regression(x, np.log(y))
        popt, perr = np.array([np.exp(intercept), -1*slope]), std_err
    else:
        popt, pcov = curve_fit(exp_decay,x,y,maxfev=maxfev,bounds=bounds)
        perr = np.sqrt(np.diag(pcov))
    return popt, perr    

def smoothness(x: np.ndarray, normalize: bool=False):
    """!
    
    @param x (numpy.ndarray):
    @param normalize (bool):
        
    @returns sm (numpy.ndarray):
    """
    # TODO: this definition is not good
    dx = np.diff(x)
    sm = np.std(dx)
    if normalize: sm /= abs(np.mean(dx))
    return sm
    
def discontinuous_idx(x: np.ndarray, std_width: int=1):
    """!
    Compute derivative of input array x, and organize the result into z-score standardized formulation. 
    Once this analysis is done, normalized results are masked for those magnitudes that are smaller than std_width, in order to ignore noises.
    
    @param x (numpy.ndarray): Input Values.
    @param std_width (int): Threshold values for masking noises.
        
    @returns idx (numpy.ndarray): Indices of discontinuous points in input array x. 
    """
    dx = np.diff(x)
    idx = np.where(abs(z_normalization(dx)) > std_width)
    return idx[0]

def is_oscillating(x: np.ndarray, osci_freq_th: float=0.3) -> bool:
    """!
    Determine whether the input array x is oscillating over its mean with frequency larger than osci_freq_th. This is equivalent to find the number of solutions of the following equation
    
    \f{equation*}{
           x - \mu = 0. 
    \f}
    
    @param x (numpy.ndarray):
    @param osci_freq_th (float):
        
    @returns out (bool):
    """
    mu = np.mean(x)
    y = x - mu
    ocsi_times = len(list(itertools.groupby(y, lambda y: y>0)))
    ocsi_freq = ocsi_times/len(x)
    if ocsi_freq > osci_freq_th:
        return True
    else:
        return False
   
def fitting_residual(x: np.ndarray, y: np.ndarray, func, args, mask_min: float=None, standardized: bool=False) -> np.ndarray:
    """!
    Compute the fitting residual.
    
    @param x (numpy.ndarray): x coordinate of input data points. 
    @param y (numpy.ndarray): y coordinate of input data points. 
    @param func (callable): Fitting function.
    @param args (numpy.ndarray): Best estimated arguments of fitting function. 
    @param mask_min (float, optional): If not None, mask resuduals that are smaller than mask_min to zero. This is always performed before standardization.
    @param standardized (bool, optional): Standardize residual to z-score formalism.
        
    @returns res (numpy.ndarray): Residual of each corresponding data points (x, y).
    """
    y_predict = func(x, *args)
    res = np.subtract(y, y_predict)
    norm = np.std(res)
    if mask_min is not None:
        res[np.where(abs(res) < mask_min)] = 0
    if standardized and norm != 0:
        res /= norm
    return res
    
def AIC_score(y: np.ndarray, y_predict: np.ndarray, p: int) -> float:
    """!
    Compute Akaike information criterion for model selection.
    
    \f{equation*}{
           \mathcal{AIC} = n \log(\mathcal{RSS}/n) + 2p, 
    \f}
    where \f$ \mathcal{RSS} \f$ is the residual sum of squares, and \f$ n \f$ is the number of data samples.
    
    @param y (numpy.ndarray): Data samples.
    @param y_predict (numpy.ndarray): Prediction by fitting.
    @param p (int): Fitting degrees of freedom, i.e. the number of parameters to fit with.
  
    @returns aic_score (float):
    """
    n = len(y)
    res = np.subtract(y, y_predict)
    rss = np.sum(np.power(res, 2))
    aic_score = n*np.log(rss/n) + 2*p
    return aic_score    

def BIC_score(y: np.ndarray, y_predict: np.ndarray, p: int) -> float:
    """!
    Compute Bayesian information criterion for model selection.
    
    \f{equation*}{
           \mathcal{BIC} = n \log(\mathcal{RSS}/n) + p \log(n), 
    \f}
    where \f$ \mathcal{RSS} \f$ is the residual sum of squares, and \f$ n \f$ is the number of data samples.
    
    @param y (numpy.ndarray): Data samples.
    @param y_predict (numpy.ndarray): Prediction by fitting.
    @param p (int): Fitting degrees of freedom, i.e. the number of parameters to fit with.
  
    @returns bic_score (float):
    """
    n = len(y)
    res = np.subtract(y, y_predict)
    rss = np.sum(np.power(res, 2))
    bic_score = n*np.log(rss/n) + p*np.log(n)
    return bic_score   

def z_normalization(x: np.ndarray) -> np.ndarray:
    """!
    Perform z-score normalizaion on input array x. 
    
    \f{equation*}{
           z = \frac{x-\mu}{\sigma}. 
    \f}
    
    @param x (numpy.ndarray): Input values.
        
    @returns normalized_x (numpy.ndarray): Output array.
    """
    return (x-np.mean(x))/np.std(x)

