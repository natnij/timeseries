# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 09:42:18 2017

@author: Nat
"""

class DfConfig(object):
    def __init__(self):
        self.dataDir = '../data/'
        self.modelDir = '../model/'
        self.figureDir = '../figure/'
        self.significance = 5 # p-value threshold in percentage
        self.maxD = 3 # maximum differencing order for timeseries trend
        self.maxDD = 2 # maximum differencing order for timeseries seasonality
        self.maxP = 5
        self.maxPP = 2
        self.maxQ = 5
        self.maxQQ = 2
        self.kpssRegType = 'c' # constant
        # recommended frequency corresponds to: every quarter;
        # every half-year; every year
        self.recommendedFreq = [3,6,12]
        # As described by the "A step-wise procedure for traversing the 
        # model space" section Hyndman and Kandahar:
        self.seasonalInitialOrder = [[2,2,1,1],[1,1,0,0],[1,0,1,0],[0,1,0,1]]
        self.nonSeasonalInitialOrder = [[2,2,0,0],[1,1,0,0],[1,0,0,0],[0,1,0,0]]
        self.epochs = 10 # number of epochs in lstm
        self.testSize = 0.2 #test size as % of total if < 1,
                            #as absolute number if >= 1.
        self.dateCol = [] # columns which need to be converted to datetime
        self.oneHotCol = [] # original categorical column names to be 
                            # converted to one-hot dummies and used 
                            # as features in multivariate timeseries analysis
        self.removeCol = [] # values to be eliminated as one-hot dummy features
        self.featureCol = [] # final list of feature names (after creating
                             # one-hot keys)
        self.aggDate = None # column name for datetime aggregation 
        self.forecastProduct = None # column of aggregated product level 
                                    # as concatenated string
        self.useOptyData = True # use opportunity data as exogenous regressor
        self.MATERIAL_CODE = None # column of smallest prod. unit
        self.matchTblName = None # tbl of smallest unit to agg. level match
        self.aggregatedDataName = None # tbl of original transactional data
        self.cutoffDate = [] # cutoffDate[0] as earliest considered date, 
                             # cutoffDate[1] as latest considered date
        self.currentDate = None # date in 'YYYYmm' format from which month 
                                # starts the forecast.
        self.minNobs = 6  # minimum number of observations in the timeseries
        self.tooShort = [] # stores invalid forecast units
        self.noMatch = [] # stores smallest units which cannot be matched 
                          # to aggregation level
        self.graph = False # default is no export of graphics during 
                           # during descriptive analysis
        self.fcstPeriod = 18 # forecast period
        self.histPeriod = 3 # number of periods to consider in  
                            # smoothing and forecast of exogenous regressors
        self.useExog = False # if to use exogenous regressor in ARIMA
        self.exogMethod = 'lowess' # possible methods are 'lowess',
                                   # 'robust_linear',
                                   # 'simple_weighted_moving_average' and 
                                   # 'exponential_weighted_moving_average'
        self.exogCol = [] # which columns to consider as exogenous columns
        self.checkAdditive = True # if to automatically check model is 
                                # additive or multiplicative, and automatically
                                # log-transform.
        self.doDescriptive = False
        self.regionCol = [] # region dimension
        self.optyCol = [] # opportunities dimension
        self.impulseResp = 'auto' # 'manual' for manual input of impulses 
                                  # in the future
                                  # or 'auto' for auto creation of 
                                  # impulse response
        self.manualInputDir = []   # data received from front-end user inputs,
                                  # same number of column as exogenous regressors,
                                  # same number of row as config.fcstPeriod