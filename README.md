# anko
Toolkit for performing anomaly detection algorithm on 1D time series based on numpy, scipy.

Conventional approaches that based on statistical analysis have been implemented, with mainly two approaches included:
1. Normal Distribution  
Data samples are presumably been generated by normal distribution, and therefore anomalous data points can be targeted by analysing the standard deviation.  

2. Fitting Ansatz  
Data samples are fitted by several ansatz, and in accordance with the residual, anomalous data points can be selected. 

Regarding model selections, models are adopted dynamically by performing normal test and by computing the (Akaike/Bayesian) information criterion.

Future development will also include methods that based on density analysis such as k-nearest neighbor and other powerful techniques.

## Requirements
* python >= 3.6.0
* numpy >= 1.16.4
* scipy >= 1.2.1

## Installation
```
pip install anko
```

## Documentation
[anko documentation](https://tanlin2013.github.io/anko/html/index.html)

## Jupyter Notebook Tutorial (in dev)
[Host on mybinder](https://mybinder.org/v2/gh/tanlin2013/anko/master?filepath=anko_tutorial.ipynb)

## Basic Usage
1. Call AnomalyDetector
```
from anko.anomaly_detector import AnomalyDetector  
agent = AnomalyDetector(t, series)
```

2. Define policies and threshold values (optional)
```
agent.thres_params["linregress_res"] = 1.5  
agent.apply_policies["z_normalization"] = True  
agent.apply_policies["info_criterion"] = 'AIC'
```

3. Run check
```
check_result = agent.check()
```

The type of output **check_result** is **anko.anomaly_detector.CheckResult**, which is basically a dictionary.
> model: 'increase_step_func'  
> popt: [220.3243250055105, 249.03846355234577, 74.00000107457113]  
> perr: [0.4247789247961187, 0.7166253174634686, 0.0]  
> anomalous_data: [(59, 209)]  
> residual: [10.050378152592119]  
> extra_info: ['Info: AnomalyDetector is using z normalization.', 'Info: There are more than 1 discontinuous points detected.']        

## Run Test (in dev)
```
python -m unittest discover -s test -p '*_test.py'
```
