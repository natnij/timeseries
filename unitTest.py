# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 11:54:33 2017

@author: Nat
"""

import numpy as np
from initialization import DfConfig as cf
config = cf()
ts = 0
ts2 = 0
exec(open('sampleData.py').read())

# unit test for trendStationaryTest(config, ts)
def testKPSS(config):
    from utilities import trendStationaryTest as tst
    tmp = np.arange(0, 200, 1) + np.random.normal(0, 0.001, 200)
    result = [tst(np.diff(np.power(tmp, config.maxD), diffOrder), 
                  config.kpssRegType, config.significance) for 
              diffOrder in np.arange(0, config.maxD + 1)]
    return result == ['non-stationary'] * config.maxD + ['trend-stationary']

# unit test for findTrendDiff(config, ts)
def testFindTrendDiffOrder(config):
    from utilities import findTrendDiff as ftd
    tmp = np.arange(0, 200, 1) + np.random.normal(0, 0.0001, 200)
    result = [ftd(np.power(tmp, min(diffOrder, config.maxD + 1)), 
                  config.maxD, config.kpssRegType, config.significance) for 
              diffOrder in np.arange(1, config.maxD + 3)]
    return result == [min(x, config.maxD + 1) for x in np.arange(1, config.maxD + 3)]

# unit test for seasonal dummy
def testSeasonalDummy(ts, frequency = 12):
    from utilities import seasonalDummy as sd
    test = sd(ts, frequency)
    return (test[0, 0] > 0.86602) & (test[0, 0] < 0.86603)

# unit test for SDtest
def testSDtest(ts, frequency = 12):
    from utilities import SDtest
    test = SDtest(ts, frequency)
    return (test > 1.524365) & (test < 1.524366)

# unit test for myTrendDiff
def testTrendDiff():
    from utilities import myTrendDiff as mtd
    a = np.arange(0, 20) * np.arange(0, 20)
    try:
        lag1 = (mtd(a, 1, padding = False) == np.diff(a, 1)).all()
    except AttributeError: lag1 = False
    try:
        lag2 = (mtd(a, 2, padding = False) == np.diff(a, 2)).all()
    except AttributeError: lag2 = False
    try:        
        lag3 = (mtd(a, 3, padding = True) == np.hstack(([0,1,2], np.diff(a, 3)))).all()
    except AttributeError: lag3 = False
    return lag1 & lag2 & lag3

# unit test for seasonalDiff
def testSeasonalDiff():
    from utilities import seasonalDiff
    a = np.arange(0, 20) * np.arange(0, 20)
    try:
        lag1 = (seasonalDiff(a, 4, 1) == np.arange(16, 144, 8)).all()
    except AttributeError: lag1 = False
    try:
        lag2 = (seasonalDiff(a, 4, 2) == np.array([32] * 12)).all()
    except AttributeError: lag2 = False
    try:
        lag3 = (seasonalDiff(a, 4, 3) == np.array([0] * 8)).all()
    except AttributeError: lag3 = False
    try:
        lag4 = (seasonalDiff(a, 4, 3, padding = True) \
            == np.hstack(([0,1,4,9,16,24,32,40,32,32,32,32], np.array([0] * 8)))).all()
    except AttributeError: lag4 = False
    return lag1 & lag2 & lag3 & lag4

# unit test for myAutoArima and mySarimaxGridSearch
def testMyAutoArima(ts, config):
    from myArima import myAutoArima as maa
    return maa(ts, config) == [0, 1, 1, 0, 0, 0, 0]

# unit test for seasonality detection
def testSeasonalDetection(config):
    from utilities import seasonalDetection as sDet
    test = [np.cos(np.pi * x / 6) for x in np.arange(0, 100)]
    return sDet(test, config.recommendedFreq, config.significance) == 12
    
# unit test for inverse differencing of seasonal differenced timeseries
def testInverseSeasonalDiff():
    from utilities import inverseSeasonalDiff as isd
    a = np.arange(0, 20) * np.arange(0, 20)
    b1 = np.arange(16, 144, 8)
    c1 = isd(a[0: 4], b1, 4, 1)
    b2 = np.array([32] * 12)
    c2 = isd(a[0: 8], b2, 4, 2)
    b3 = np.array([0] * 8)
    c3 = isd(a[0: 12], b3, 4, 3)
    return (c1 == a).all() & (c2 == a).all() & (c3 == a).all()

# unit test for inverse differencing of trend differenced timeseries
def testInverseTrendDiff():
    from utilities import inverseTrendDiff as itd
    a = np.arange(0, 20) * np.arange(0, 20)
    b1 = np.diff(a, 1)
    c1 = itd(a[0:1], b1, 1)
    b2 = np.diff(a, 2)
    c2 = itd(a[0:2], b2, 2)
    b3 = np.diff(a, 3)
    c3 = itd(a[0:3], b3, 3)
    return (c1 == a).all() & (c2 == a).all() & (c3 == a).all()    

# test lstm adding autoregressive columns in window
def testLstmAddARcols(ts):
    ts = np.array(ts).reshape([-1,1])
    tmp = np.column_stack((ts * ts, ts))
    from lstm import addARcols as ar
    result = ar(tmp, lag = 4)
    return (result.shape[1] == 6) & (np.sum(result[0:4,0]) == 0)

# test lstm adding seasonal autoregressive columns in window
def testLstmAddSeasonalARcols(ts):
    ts = np.array(ts).reshape([-1,1])
    tmp = np.column_stack((ts * ts, ts))
    from lstm import addSeasonalARcols as sar
    result = sar(tmp, frequency = 4, seasLag = 2)
    return (result.shape[1] == 4) & (np.sum(result[0:8,0]) == 0)

# test myLstm()
def testMyLstm(ts2):
    from lstm import myLstm, scores
    trainPredInv, trainY_actual, \
    testPredInv, testY_actual = myLstm(ts2, [2,1,0,1,1,0,4])   
    trainR2, testR2 = scores(trainPredInv, trainY_actual, 
                             testPredInv, testY_actual )
    return trainR2, testR2

def testAll():
    return testKPSS(config) & testFindTrendDiffOrder(config) & \
        testSeasonalDummy(ts) & testSDtest(ts) & testTrendDiff() & \
        testSeasonalDiff() & testMyAutoArima(ts, config) & \
        testSeasonalDetection(config) & \
        testInverseTrendDiff() & testInverseSeasonalDiff() & \
        testLstmAddARcols(ts) & testLstmAddSeasonalARcols(ts)


    