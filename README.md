# anko
Toolkit for performing anomaly detection algorithm on 1D time series based on numpy, scipy.

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
