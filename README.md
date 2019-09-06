# anko
Toolkit for performing anomaly detection algorithm on 1D time series based on numpy, scipy.

## Installation:
```
pip install anko
```

## Documentation:
[anko](https://tanlin2013.github.io/anko/html/index.html)

## How to use:
* First step: 
    call AnomalyDetector
```
from anko.anomaly_detector import AnomalyDetector
agent = AnomalyDetector(t, series)
```
* Second step: 
    define policies and threshold values (optional)
```
agent.thres_params["linregress_res"] = 1.5
agent.apply_policies["z_normalization"] = True
agent.apply_policies["info_criterion"] = 'AIC'
```
* Third step: 
    run check
```
check_result = agent.check()
```
the type of output **check_result** is **anko.anomaly_detector.CheckResult**, which is basically a dictionary.
```
model: 'increase_step_func'
popt: [220.3243250055105, 249.03846355234577, 74.00000107457113]
perr: [0.4247789247961187, 0.7166253174634686, 0.0]
anomalous_data: [(59, 209)]
residual: [10.050378152592119]
extra_info: ['Info: AnomalyDetector is using z normalization.', 'Info: There are more than 1 discontinuous points detected.']        
``` 

## Run test (in dev)
```
python -m unittest discover -s test -p '*_test.py'
```