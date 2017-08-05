# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 08:21:24 2017

@author: Nat
"""

def myAutoArima(fullTs, config):
    '''
    entry point for arima model. 
    the function detects seasonality, finds both trend and seasonal
        differencing order to ensure stability of the timeseries, 
        and systematically searches for the best parameters [p, q, P, Q]
    for reference see R forecast package from Hyndman
    https://github.com/robjhyndman/forecast/blob/master/R/newarima2.R
    input: univariate timeseries in either pandas dataframe 
        or numpy array format.
    output: 
        params for best SARIMAX model
    '''
    import sys
    import utilities as ut
    import numpy as np
    import itertools
    
    tsArray = ut.pdDFtoArray(fullTs)

    # find differencing order and generate diffed timeseries
    d = min(config.maxD, ut.findTrendDiff(tsArray, config.maxD, 
                                          config.kpssRegType, 
                                          config.significance))
    tsArrayDiff = np.diff(tsArray, n = d)

    # detect seasonality
    frequency = ut.seasonalDetection(tsArrayDiff, config.recommendedFreq, 
                                     config.significance)
    if not isinstance(frequency, int):
        frequency = int(frequency)
    if frequency not in config.recommendedFreq + [0]:
        print("warning: auto-detected frequency %d may be incorrect." %frequency)  
    
    if frequency <= 1:
        D = 0
        tsArraySeasonalDiff = tsArrayDiff
    else: 
        # find seasonal differencing order and generate diffed timeseries
        D = min(config.maxDD, ut.findSeasonalDiff(tsArrayDiff, frequency,config.maxDD))
        tsArraySeasonalDiff = ut.seasonalDiff(tsArrayDiff, 
                                              frequency = frequency, 
                                              order = D, padding = False)
    
    curBestAIC = sys.maxsize
    pastParams = list()
    curBestModel = np.zeros(7)
    finished = False
    
    # search step of arima seasonal P, Q orders
    PQDeltas = [-1, 0, 1] if frequency > 1 else [0, 0, 0]
    # search step of arima non-seasonal p, q orders
    pqDeltas = [-1, 0, 1]
    # search space [p, q, P, Q] initialized as suggested by Hyndman
    nextParams = config.seasonalInitialOrder if frequency > 1 \
        else config.nonSeasonalInitialOrder
    
    while not finished:
        modelList = list(map(lambda x: mySarimaxGridSearch(config, tsArraySeasonalDiff, frequency, x), nextParams))
        try:
            newBestAIC = min(item[4] for item in modelList)
        except IndexError: 
            newBestAIC = sys.maxsize
        if newBestAIC == sys.maxsize:
            finished = True
        
        if newBestAIC < curBestAIC:
            curBestAIC = newBestAIC
            curBestModel = [item for item in modelList if item[4] == curBestAIC][0][0:4]
        
        # log used parameters to save time in consequent searches
        pastParams = pastParams + nextParams
        
        nextParams = list()
        for r in itertools.product(pqDeltas, pqDeltas, PQDeltas, PQDeltas): 
            p = max(0, curBestModel[0] + r[0])
            q = max(0, curBestModel[1] + r[1])
            P = max(0, curBestModel[2] + r[2])
            Q = max(0, curBestModel[3] + r[3])
            nextParams = nextParams + [[p,q,P,Q]]
        nextParams.sort()
        nextParams = [item for item,_ in itertools.groupby(nextParams) \
                      if (sum(item)>0) & (item not in pastParams)]
        if nextParams == []:
            finished = True
        
        if curBestModel[2] + curBestModel[3] + D == 0: frequency = 0
    return [curBestModel[0], d, curBestModel[1], curBestModel[2], D, curBestModel[3], frequency]
        
def mySarimaxGridSearch(config, fullTs, frequency = 12, params = [1,0,1,0]):
    '''
    statsmodels reference:
    http://www.statsmodels.org/dev/statespace.html#seasonal-autoregressive-integrated-moving-average-with-exogenous-regressors-sarimax
        details through links to class SARIMAX and SARIMAXResults
    http://www.statsmodels.org/dev/examples/notebooks/generated/statespace_sarimax_stata.html
    '''
    import sys
    import statsmodels.api as sm
    if (params[0] > config.maxP) | (params[1] > config.maxQ) \
        | (params[2] > config.maxPP) | (params[3] > config.maxQQ):
        return(params + [sys.maxsize]), False
            
    if params[2] + params[3] == 0: frequency = 0
    try:
        model = sm.tsa.statespace.SARIMAX(fullTs,                  
                         order = (params[0], 0, params[1]), 
                         seasonal_order = (params[2], 0, params[3], frequency), 
                         trend = config.kpssRegType)
        modelfit = model.fit()
        aic = modelfit.aic
    except:
        aic = sys.maxsize
    return params + [aic]
