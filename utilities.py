# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 08:29:01 2017
Common utilities

@author: Nat
"""

def pdDFtoArray(fullTs):
    '''
    check input and convert pandas dataframes to numpy arrays
    input: univariate timeseries in either pandas dataframe format or 
        numpy array format.
    output: always a 1d array.
    '''
    import pandas as pd
    import numpy as np
    if isinstance(fullTs, pd.DataFrame):
        tsArray = np.array(fullTs.iloc[:, 0])
    else:
        tsArray = np.array(fullTs)
    return tsArray

def regressMissingData(x, y, xnew, robust = True):
    '''
    linear or robust linear regression to fill in missing data
    '''
    import pandas as pd
    from sklearn.linear_model import LinearRegression as lr
    m = lr()
    m.fit(x, y)
    ynew_lr = pd.DataFrame(m.predict(xnew), columns = ['WON_MONTH2'])

    from sklearn.linear_model import RANSACRegressor as ransac
    m_ransac = ransac(lr())
    m_ransac.fit(x, y)   
    ynew_ransac = pd.DataFrame(m_ransac.predict(xnew), 
                               columns = ['WON_MONTH2'])
#    import numpy as np
#    from matplotlib import pyplot as plt
#    yhat_lr = pd.DataFrame(m.predict(x))
#    yhat_ransac = pd.DataFrame(m_ransac.predict(x))
#    inlier_mask = m_ransac.inlier_mask_
#    outlier_mask = np.logical_not(inlier_mask) 
#    plt.scatter(x[inlier_mask], y[inlier_mask], 
#                color='green', marker='.',
#                label='Inliers')
#    plt.scatter(x[outlier_mask], y[outlier_mask], 
#                color='red', marker='.',
#                label='Outliers')
#    plt.plot(pd.concat([x,xnew]), pd.concat([yhat_ransac, ynew_ransac]), '-', 
#             label='RANSAC regressor')
#    plt.plot(pd.concat([x,xnew]), pd.concat([yhat_lr, ynew_lr]), '-', 
#             label='linear regressor')
#    plt.show()
    if robust == True:
        return ynew_ransac
    else:
        return ynew_lr

# use kpss test for trend stationarity check
def trendStationaryTest(fullTs, kpssRegType = 'c', significance = 5):
    '''
    perform kpss test.
    input:
        fullTs: univariate timeseries to be tested. either in pandas.Dataframe 
            format with one column, or in numpy 1d array format.
        kpss type: c: constant. ct: constant trend.
        significance: e.g. 5 for 5%. 
    output:
        'trend-stationary': passing trend stationary test (confirm kpss
            null-hypothesis)
        'non-stationary': not passing test (reject kpss null)
    '''
    from statsmodels.tsa.stattools import kpss
    tsArray = pdDFtoArray(fullTs)
    try:
        result = kpss(tsArray, regression = kpssRegType)
    except (ValueError, ZeroDivisionError): 
        return 'non-stationary'
    pValue = result[1]
    return 'trend-stationary' if pValue >= (significance/100) else 'non-stationary'

def findTrendDiff(fullTs, maxD = 3, kpssRegType = 'c', significance = 5):
    '''
    auto-differencing until kpss null hypothesis is true
    '''
    import numpy as np
    tsArray = pdDFtoArray(fullTs)
    for x in np.arange(0, maxD + 2):
        tmpTs = np.diff(tsArray, x)
        if trendStationaryTest(tmpTs, kpssRegType, significance) == 'trend-stationary':
            break
        else: x = x + 1
    if x > maxD: 
        print("warning: trend differencing order %d is higher than specified maximum order %d" 
              %(x, maxD))
    return x

def seasonalDummy(fullTs, frequency):
    '''
    generate seasonal dummy matrix using Fourier series 
    for Canova-Hansen test
    '''
    import numpy as np
    tsArray = pdDFtoArray(fullTs)
    n = len(tsArray)
    m = frequency
    #if m == 1: tsArray.reshape([n, m])
    tt = np.arange(1, n + 1, 1)
    mat = np.zeros([n, 2 * m], dtype = float)
    for i in np.arange(0, m):
        mat[:, 2 * i] = np.cos(2.0 * np.pi * (i + 1) * tt / m)
        mat[:, 2 * i + 1] = np.sin(2.0 * np.pi * (i + 1) *tt / m)
    return mat[:, 0:(m-1)]
    
def SDtest(fullTs, frequency):
    '''
    based on R uroot test and forecast package from Hyndman
    for reference see 
    https://github.com/robjhyndman/forecast/blob/master/R/arima.R
    '''
    import numpy as np
    if frequency <= 1:
        return 0 # no seasonality with frequency == 0 or 1
    tsArray = pdDFtoArray(fullTs)
    n = len(tsArray)
    if (n <= frequency): 
        return 0 # insufficient data
    
    frec = np.ones(int((frequency + 1) / 2), dtype = int)
    ltrunc = int(np.round(frequency * np.power(n / 100.0, 0.25)))
    # create dummy column
    r1 = seasonalDummy(tsArray, frequency)
    #create intercept column for regression
    r1wInterceptCol = np.column_stack([np.ones(r1.shape[0], dtype = float), r1])
    from numpy.linalg import lstsq as lsq
    result = lsq(a = r1wInterceptCol, b = tsArray)
    residual = tsArray - np.matmul(r1wInterceptCol, result[0])
    fhat = np.zeros([n, frequency - 1], dtype = float)
    fhataux = np.zeros([n, frequency - 1], dtype = float)
    
    for i in np.arange(0, frequency - 1):
        fhataux[:, i] = r1[:, i] * residual
    
    for i in np.arange(0, n):
        for j in np.arange(0, frequency - 1):
            mySum = sum(fhataux[0:(i + 1), j])
            fhat[i, j] = mySum
    
    wnw = np.ones(ltrunc, dtype = float) - np.arange(1, ltrunc + 1, 1) / (ltrunc + 1)
    Ne = fhataux.shape[0]
    omnw = np.zeros([fhataux.shape[1], fhataux.shape[1]], dtype = float)
    for k in np.arange(0, ltrunc):
        omnw = omnw + np.matmul(fhataux.T[:, (k+1):Ne], fhataux[0:(Ne-(k+1)), :]) * float(wnw[k])
    
    cross = np.matmul(fhataux.T, fhataux)
    omnwplusTranspose = omnw + omnw.T
    omfhat = (cross + omnwplusTranspose) / float(Ne)
    
    sq = np.arange(0, frequency - 1, 2)
    frecob = np.zeros(frequency - 1, dtype = int)
    
    for i in np.arange(0, len(frec)):
        if (frec[i] == 1) & (i + 1 == int(frequency / 2.0)):
            frecob[sq[i]] = 1
        if (frec[i] == 1) & (i + 1 < int(frequency / 2.0)):
            frecob[sq[i]] = 1
            frecob[sq[i] + 1] = 1
    
    a = frecob.tolist().count(1)  # find nr of 1's in frecob
    A = np.zeros([frequency - 1, a], dtype = float)
    j = 0
    for i in np.arange(0, frequency - 1):
        if frecob[i] == 1:
            A[i, j] = 1
            j = j + 1
        
    aTomfhat = np.matmul(A.T, omfhat)
    tmp = np.matmul(aTomfhat, A)
    
    machineDoubleEps = 2.220446e-16
    from numpy.linalg import svd
    problems = min(svd(tmp)[1]) < machineDoubleEps # svd[1] are the singular values
    if problems:
        stL = 0.0
    else:
        solved = np.linalg.solve(tmp, np.eye(tmp.shape[1], dtype = float))
        step1 = np.matmul(solved, A.T)
        step2 = np.matmul(step1, fhat.T)
        step3 = np.matmul(step2, fhat)
        step4 = np.matmul(step3, A)
        stL = (1.0 / np.power(n, 2.0)) * sum(np.diag(step4))
    
    return stL

def CHtest(fullTs, frequency):
    '''
    Canova-Hansen test of seasonal stability 
    based on R forecast package and uroot package for CH.test()
    for coding reference see https://github.com/robjhyndman/forecast/blob/master/R/newarima2.R
    critical values with different frequencies corresponding to 0.1 significance level
    http://www.ssc.wisc.edu/~bhansen/papers/jbes_95.pdf.(10% significance level)
    frequency > 12 : critical values are copied from R package forecast
    input:
        fullTs: univariate timeseries to be tested. either in pandas 
            dataframe or in numpy 1d array format.
        frequency: frequency to be tested. integer.
    output:
        'non-stationary' : stat value > critical value, p-value < significance level, 
            reject CHtest null hypothesis of seasonal stationary.
        'seasonal-stationary': CHtest null accepted.
    '''
    import numpy as np
    tsArray = pdDFtoArray(fullTs)
    if len(tsArray) < 2 * frequency + 5:
        return 'seasonal-stationary' # insufficient data
    
    chstat = SDtest(tsArray, frequency)
    # keys correspond to frequency, values correspond to critical values 
    # at 0.1 significance level except for freq >= 24 (those at 0.05)
    critValues = {2: 0.353, 3: 0.610, 4: 0.846, 5: 1.070, 6: 1.280, 7: 1.490, 
                  8: 1.690, 9: 1.890, 10:2.100, 11:2.290, 12:2.490, 13:2.690,
                  24:5.098624, 52:10.341416, 365:65.44445}
    if frequency not in critValues.keys():
        return chstat > 0.269 * np.power(frequency, 0.928)
    
    return 'non-stationary' if chstat > critValues.get(frequency) else 'seasonal-stationary'

def findSeasonalDiff(fullTs, frequency, maxDD = 2):
    '''
    auto-seasonal-differencing until canova-hansen null hypothesis is true
    '''
    import numpy as np
    tsArray = pdDFtoArray(fullTs)
    for x in np.arange(0, maxDD + 2):
        tmpTs = np.diff(tsArray, x * frequency)
        if CHtest(tmpTs, frequency) == 'seasonal-stationary':
            break
        else: x = x + 1
    if x > maxDD: 
        print("warning: seasonal differencing order %d is higher than specified maximum order %d" 
              %(x, maxDD))
    return x
        
def seasonalDetection(fullTs, recommendedFreq, significance = 5):
    '''
    return frequency as detected by acf and power density analysis
    '''
    tsArray = pdDFtoArray(fullTs)
    freq = 0
    for frequency in recommendedFreq:
        fftResult = myFft(tsArray, frequency)
        acfResult = myAcf(tsArray, frequency, significance)
        
        if fftResult * acfResult == True: 
            freq = frequency
            break       
    return freq

def myFft(fullTs, frequency):
    '''
    simple power density check on given frequency
    input:
        univariate timeseries
        frequency to be tested
    output:
        if fftResult == True then seasonality at the tested frequency 
            is confirmed.
    '''
    import numpy as np
    tsArray = pdDFtoArray(fullTs)
    # extend ts length to multiply of frequency
    n = int(len(tsArray) / frequency) * frequency
    tsTrunc = tsArray[-n:]
    
    fft1 = np.fft.fft(tsTrunc)
    pr = np.power(np.abs(fft1), 2)
    
    maxPower = 0
    indexPower = 0
    for i in np.arange(1, int(n/2)):
        if pr[i] > maxPower:
            maxPower = pr[i]
            indexPower = i        
    try:
        fftResult = (frequency % int(n / indexPower) == 0)
    except ZeroDivisionError: 
        fftResult = False
    return fftResult

def myAcf(fullTs, frequency, significance = 5, unbiased = True):
    from scipy.stats import norm
    import numpy as np
    from statsmodels.tsa.stattools import acf
    tsArray = pdDFtoArray(fullTs)
    n = len(tsArray)
    ci = 1 - significance / 100 # confidence interval
    try:
        acfResult = acf(tsArray, unbiased = True, nlags = frequency)
    except TypeError:
        return False
    # calculate clim line via gaussian inverse CDF, or percent point function
    # at confidence level of 95% (or 2,5% on each side if two-tailed)
    clim = norm.ppf((1 + ci) / 2) / np.sqrt(n)
    try:
        result = acfResult[frequency] >= clim
    except IndexError:
        return False
    return result

def seasonalDiff(fullTs, frequency = 12, order = 1, padding = False):
    '''
    differencing with lag > 1.
    input:
        fullTs: univariate timeseries in pandas dataframe or numpy array.
        frequency: differencing lag.
        order: differencing order. 
        padding: boolean. if true, then the first elements of output 
            array will be zero-padded to keep same array size as input.
    output: 
        differenced timeseries in numpy array format.
        length of the output 1d array is original length - frequency * order
            if padding == False; same as input if padding == True
    '''
    import numpy as np
    tsArray = pdDFtoArray(fullTs)
    tsArrayPadding = np.array([])
    
    n = len(tsArray)
    if order == 0:
        return fullTs
    if n <= frequency * order:
        print('Warning: insufficient data. Original series is returned.')
        return fullTs
    
    for j in np.arange(0, order):
        tmp = tsArray.copy()    
        tsArrayDiffed = [tsArray[i] - tmp[i-frequency] 
            for i in np.arange(frequency, n)]
        # padded with original timeseries:
        if padding == True: 
            tsArrayPadding = np.hstack((tsArrayPadding, tsArray[0: frequency]))
        tsArray = tsArrayDiffed.copy()
        n = len(tsArray)
        
    if padding == True:
        tsArrayDiffed = np.hstack((tsArrayPadding, tsArrayDiffed))
    
    return tsArrayDiffed

def myTrendDiff(fullTs, order = 1, padding = True):
    '''
    wrapper for np.diff in case of padding.
    if padding == True: beginning of the original series will be copied to 
        the new diffedTs.
    '''
    import numpy as np
    tsArray = pdDFtoArray(fullTs)
    tsArrayPadding = np.array([])
    n = len(tsArray)
    
    if order == 0:
        return fullTs
    if n <= order:
        print('Warning: insufficient data. Original series is returned.')
        return fullTs
    
    for j in np.arange(0, order):
        tmp = tsArray.copy()    
        tsArrayDiffed = [tsArray[i] - tmp[i - 1] 
            for i in np.arange(1, n)]
        # padded with original timeseries:
        if padding == True: 
            tsArrayPadding = np.hstack((tsArrayPadding, tsArray[0]))
        tsArray = tsArrayDiffed.copy()
        n = len(tsArray)

    if padding == True:
        tsArrayDiffed = np.hstack((tsArrayPadding, tsArrayDiffed))
    
    return tsArrayDiffed

def inverseTrendDiff(_originalTs, _diffedTs, order = 1):
    '''
    convert a differenced timeseries back to its original
    reference at 
    https://github.com/sryza/spark-timeseries/blob/master/src/main/scala/com/cloudera/sparkts/UnivariateTimeSeries.scala
    '''
    import numpy as np
    originalTs = _originalTs.copy()
    diffedTs = _diffedTs.copy()
    if order == 0:
        return diffedTs
    
    originalTs = myTrendDiff(pdDFtoArray(originalTs)[-order:], order - 1)
    diffedTs = pdDFtoArray(diffedTs)
    for i in np.arange(order, 0, -1):
        # add back the part of original timeseries relevant to the order
        diffedTs = np.hstack((originalTs[-1], diffedTs))
        originalTs = originalTs[0:-1]
        addedTs = inverseDiffAtLag(diffedTs, 1, 1)
        diffedTs = addedTs.copy()
    return addedTs

def inverseSeasonalDiff(_originalTs, _diffedTs, frequency = 12, order = 1):
    '''
    convert a seasonal-differenced timeseries back to its original
    '''
    import numpy as np
    originalTs = _originalTs.copy()
    diffedTs = _diffedTs.copy()
    if order == 0:
        return diffedTs

    originalTs = seasonalDiff(pdDFtoArray(originalTs)[-(order * frequency):],
                              frequency = frequency, order = order - 1, 
                              padding = True)
    diffedTs = pdDFtoArray(diffedTs)
    for i in np.arange(order, 0, -1):
        # add back the part of original timeseries relevant to the order
        diffedTs = np.hstack((originalTs[-frequency:], diffedTs))
        originalTs = originalTs[0:-frequency]
        addedTs = inverseDiffAtLag(diffedTs, frequency, frequency)
        diffedTs = addedTs.copy()
    return addedTs    

def inverseDiffAtLag(_diffedTs, lag = 1, startIndex = 1):
    diffedTs = _diffedTs.copy()    
    if lag == 0: 
        return diffedTs
    if startIndex - lag < 0:
        startIndex = lag
        
    addedTs = diffedTs.copy()
    n = len(diffedTs)
    i = 0
    while i < n:
        #elements prior to starting point are copied
        if i < startIndex:
            addedTs[i] = diffedTs[i] 
        else: 
            addedTs[i] = diffedTs[i] + addedTs[i - lag]
        i = i + 1
    return addedTs

def plotNormality(ts, ttl, figureDir):
    '''
    boxcox reference: https://stats.stackexchange.com/questions/61217/transforming-variables-for-multiple-regression-in-r
    scipy reference: https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.boxcox.html
    '''
    from scipy import stats
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax1 = fig.add_subplot(211) # nrow = 2, ncol = 1, chart no.1
    x = ts.copy()
    stats.probplot(x, dist=stats.norm, plot=ax1)
    ax1.set_xlabel('')
    ax1.set_title('Probplot against normal distribution')
    ax2 = fig.add_subplot(212) # nrow = 2, ncol = 1, chart no.2
    xt, boxcox_lambda = stats.boxcox(x)
    stats.probplot(xt, dist=stats.norm, plot=ax2)
    ax2.set_title('Probplot after Box-Cox transformation')
    plt.savefig(figureDir + ttl + '_boxcox.png')
    
#    from scipy.special import inv_boxcox
#    xt_inv = inv_boxcox(xt, boxcox_lambda)
    
    return xt, boxcox_lambda

def scoreR2(_yhat, _y, nCoeffs = 1, adjusted = True):
    '''    
    when-to-use-R2 reference:
        http://blog.minitab.com/blog/adventures-in-statistics-2/why-is-there-no-r-squared-for-nonlinear-regression
        http://people.duke.edu/~rnau/rsquared.htm
    which-adjusted-R2-to-use reference:
        https://stats.stackexchange.com/questions/25214/how-to-choose-between-the-different-adjusted-r2-formulas
    or: sklearn.metrics.r2_score
    '''
    import numpy as np
    yhat = np.array(_yhat.copy())
    y = np.array(_y.copy())
    ybar = np.ones(len(y)) * np.mean(y)
    ss_ttl = np.sum(np.power(np.subtract(y.reshape(-1,), ybar), 2))
    ss_res = np.sum(np.power(np.subtract(y.reshape(-1,), 
                                         yhat.reshape(-1,)), 2))
    R2 = 1 - ss_res / ss_ttl
    
    if (adjusted == True) & (len(y) - nCoeffs - 1 > 0):
        R2 = 1 - (len(y) - 1) / (len(y) - nCoeffs - 1) * (1 - R2)
        print('returning adjusted R-squared from Ezekiel, M. (1930).')
    
    return R2

def scoreRMSE(_yhat, _y):
    '''
    when-to-use reference:
        https://people.duke.edu/~rnau/compare.htm
    or: sklearn.metrics.mean_squared_error, then square-root
    '''
    import numpy as np
    yhat = np.array(_yhat.copy())
    y = np.array(_y.copy())
    residual = np.subtract(y, yhat)
    s = np.sum(np.power(residual, 2))
    RMSE = np.sqrt(s / len(y))
    return RMSE 
    
def scoreMAE(_yhat, _y):
    '''
    when-to-use reference:
        https://people.duke.edu/~rnau/compare.htm
    or: sklearn.metrics.mean_absolute_error
    '''
    import numpy as np
    yhat = np.array(_yhat.copy())
    y = np.array(_y.copy())
    return np.abs(np.subtract(y, yhat)).mean()

def scoreMAPE(_yhat, _y):
    '''
    only apply to data that are strictly positive!
    possible division by a number close to zero! use with caution!
    heavier penalty on positive errors than negative errors.
    function returns the percent number scaled to (0,100).
    '''
    import numpy as np
    yhat = np.array(_yhat.copy())
    y = np.array(_y.copy())
    return np.abs(np.subtract(y, yhat) / (y + 1e-7)).mean() * 100

def scoreSymmetricMAPE(_yhat, _y):
    '''
    possible division by a number close to zero! use with caution!
    function returns the percent number scaled to 100, however can be negative.    
    '''    
    import numpy as np
    yhat = np.array(_yhat.copy())
    y = np.array(_y.copy())
    return (np.abs(np.subtract(y, yhat)) / (y + yhat)).mean() * 200
    
def scoreMASE(_yhat, _ytrain, _ytest, frequency = 0):
    '''
    mean absolute scaled error, Hyndman, R. J. (2006).
    compares forecast with the one-step (seasonal) naive forecast method.
    reference: https://robjhyndman.com/papers/foresight.pdf
               https://en.wikipedia.org/wiki/Mean_absolute_scaled_error
    input:
        ytrain: original timeseries in training set
        ytest: y_true in testset
        yhat: predicted y of testset
    '''
    import numpy as np
    yhat = np.array(_yhat.copy())
    ytrain = np.array(_ytrain.copy())
    ytest = np.array(_ytest.copy())
    t = len(ytrain)
    error = scoreMAE(yhat, ytest)

    if (frequency <= 1) | (t - frequency <= 0):
        # non-seasonal:
        ydiff = np.diff(ytrain)
        d = np.abs(ydiff).sum() / (t - 1)
    else:
        # seasonal:
        ydiff = seasonalDiff(ytrain, frequency)
        d = np.abs(ydiff).sum() / (t - frequency)
    return error / d

def myLr(x, y, xnew):
    from sklearn.linear_model import LinearRegression as lr
    import numpy as np
    model = lr()
    model.fit(x, y)
    ynew = model.predict(xnew)
    ynew = np.where(ynew < 0, 0, ynew)
    return ynew

def shiftLeft(_ar, step = 1):
    if step == 0:
        return _ar
    ar = _ar.copy()
    ar[0:-step] = ar[step:]
    ar[-step:] = 0
    return ar

def shiftRight(_ar, step = 1):
    if step == 0:
        return _ar
    ar = _ar.copy()
    ar[step:] = ar[0:-step]
    ar[0:step] = 0
    return ar

def mySma(x, histPeriod = 6, fcstPeriod = 18, weightDecayFactor = 2):
    '''
    simple moving average
    weightDecayFactor defines speed of decay:
        weight[n-i] = weight[n] / weightDecayFactor^i,
        sum(weight) = 1
    '''
    import numpy as np
    weight = np.ones(histPeriod)
    xfit = np.zeros(len(x))
    xfit[:(histPeriod-1)] = x[:(histPeriod-1)].copy()
    xpred = np.zeros(fcstPeriod)
    for i in np.arange(1, len(weight)):
        weight[i] = weight[i-1] * weightDecayFactor
    weight = weight / sum(weight)
    xfit[(histPeriod-1):] = np.convolve(x, weight, mode='valid')
    tmp = x[-histPeriod:].copy()
    for i in np.arange(0, fcstPeriod):
        xpred[i] = np.matmul(tmp, weight.T)
        tmp = shiftLeft(tmp)
        tmp[-1] = xpred[i]
    xfit[:(histPeriod-1)]= 0
    return xfit, xpred

def myEwma(x, histPeriod = 6, fcstPeriod = 18):    
    '''
    exponential weighted moving average
    weighted_avg[n] = (1-alpha) * weighted_avg[n-1] + alpha * x[n],
    weighted_avg[0] = x[0],
    alpha = 2 / (span + 1)
    '''
    from pandas import ewma
    import numpy as np
    xfit = ewma(x, span = histPeriod)
    xpred = np.zeros(fcstPeriod)
    tmp = np.zeros(histPeriod + 1)
    tmp[:histPeriod] = x[-histPeriod:].copy()
    tmp[histPeriod] = xfit[-1]
    for i in np.arange(0, fcstPeriod):
        xpred[i] = ewma(tmp, span = histPeriod)[-1]
        tmp = shiftLeft(tmp)
        tmp[-1] = xpred[i]
    return xfit, xpred

def findExog(timeseries, histPeriod = 3, fcstPeriod = 18, 
             maxResponseDelay = 12, method = 'lowess',
             title = None, figureDir = None, impulseResp = 'auto',
             manualImpulse = None):
    '''
    find exogenous regressor for univariate timeseries
    input: 
        ts: either 1d for univariate timeseries, or multidimensional with last
            column being the dependent variable.
        manualImpulse: same number of columns as ts, and 
            same number of rows as fcstPeriod. 
            Should be direct inputs from user to directly manipulate
            trend and impulse. All numeric values
        histPeriod: number of period in the past to consider for 
            future linear fit. default is 3.
        fcstPeriod: number of points to generate as future forecast. 
            default is 18.
        maxResponseDelay: maximum nr. of periods to test covariance between 
            exogenous regressors and the trend (to lag the xregressor)
        method: lowess, robust_linear, simple_weighted_moving_average, 
            exponential_weighted_moving_average.
        inpulseResp: 'manual' for manual input of impulses for future periods
            entered by user, or 'auto' for auto creation of 
            impulse response for future periods
    if method == 'lowess':
        use lowess to fit the curve on target exogenous variable(y), and
        use the last [histPeriod] periods of lowess result for future forecast.
        lowess is only univariate. Extra regressors(x) will be ignored.
    if method == 'robust_linear':
        use ransac on linear model to fit the curve and forcast.
    if method = 'simple_weighted_moving_average' or 
        'exponential_weighted_moving_average', use simple smoothing functions
        with custom or exponentially decaying weights. If there is extra 
        regressor(x), use corresponding moving average methods (exponential
        weight as default) to forecast future extra regressor value, and 
        use linear regression to fit and forecast target exogenous variable(y).
    '''
    import statsmodels.api as sm
    from sklearn.linear_model import LinearRegression as lr
    import numpy as np
    import matplotlib.pyplot as plt
    ts = timeseries.copy()
    if manualImpulse is not None:
        impulse = manualImpulse.copy()
    impulseRespType = 'not detected'
    if method == 'simple_weighted_moving_average':
        myFunc = mySma
    else:
        myFunc = myEwma
    try:
        ts.shape[1]
    except IndexError:
        ts = ts[:, np.newaxis]
    timeDim = np.arange(1, (ts.shape[0] + 1))
    timeNew = np.arange(np.max(timeDim) + 1, np.max(timeDim) + 1 + fcstPeriod)   
    for i in np.arange(0, ts.shape[1] - 1):
        if np.var(ts[:,i]) == 0:
            ts = np.delete(ts, i, 1)
            if manualImpulse is not None:
                impulse = np.delete(impulse, i, 1)
    if ts.shape[1] > 1:
        x = np.array(ts)[:, :-1].copy()
        xnew = np.zeros([fcstPeriod, x.shape[1]])
        for i in np.arange(0, x.shape[1]):
            # smoothing function to predict extra regressors
            _, xnew[:,i] = myFunc(x[:,i].reshape(-1,), histPeriod, fcstPeriod)
        xnew = np.row_stack([x, xnew])
    else:
        x = timeDim[:, np.newaxis]
        xnew = np.hstack([timeDim, timeNew])[:, np.newaxis]
    y = np.array(ts).copy()[:,-1].reshape(-1, 1)
   
    if method == 'lowess':
        lowess = sm.nonparametric.lowess(y.reshape(-1,), timeDim, frac=.3)
        lowess_y = np.array(list(zip(*lowess)))[1]
        
        ynew = myLr(timeDim[-histPeriod:, np.newaxis], 
                    lowess_y[-histPeriod:, np.newaxis], timeNew[:, np.newaxis])
        if title is not None:
            fig, ax = plt.subplots()
            plt.plot(timeDim, np.array(y), 'o')
            plt.plot(timeDim, lowess_y, '-')
            plt.plot(timeNew, ynew, '*')
            ax.set_title(title)
            fig.tight_layout()
            plt.savefig(figureDir + method +'_' + title + '.png')
            plt.close()    
            
        result = np.hstack([lowess_y.reshape(-1,), ynew.reshape(-1,)])
        x_exog = result.copy()
    elif method == 'robust_linear':
        #use ransac regressor for robust linear regression
        from sklearn.linear_model import RANSACRegressor as ransac
        model_ransac = ransac(lr())
        try:
            model_ransac.fit(x, y)
            ynew = model_ransac.predict(xnew)
            ynew = np.where(ynew < 0, 0, ynew)
            if title is not None:
                fig, ax = plt.subplots()
                inlier_mask = model_ransac.inlier_mask_
                outlier_mask = np.logical_not(inlier_mask)    
                plt.scatter(timeDim[inlier_mask], y[inlier_mask], 
                            color='green', marker='.',
                            label='Inliers')
                plt.scatter(timeDim[outlier_mask], y[outlier_mask], 
                            color='red', marker='.',
                            label='Outliers')
                plt.plot(np.hstack([timeDim, timeNew]), 
                         ynew, '-', label='RANSAC regressor')
                ax.set_title(title)
                fig.tight_layout()
                plt.savefig(figureDir + method +'_' + title + '.png')
                plt.close()    
        except ValueError:
            ynew = myLr(x, y, xnew)
            if title is not None:
                fig, ax = plt.subplots()
                plt.scatter(timeDim, y, color='red', marker='.')
                plt.plot(np.hstack([timeDim, timeNew]), 
                         ynew, '-', label='linear regressor')
                ax.set_title(title)
                fig.tight_layout()
                plt.savefig(figureDir + method +'_' + title + '.png')
                plt.close()    
        result = ynew    
        x_exog = xnew
    else:
        # use simple linear / exponential 
        # weighted moving avarege of periods = histPeriod
        x_exog = xnew
        if ts.shape[1] > 1:
            y_imp = myEwma(y.reshape(-1,), 6)[0]
            n = maxResponseDelay
            for i in np.arange(0, x.shape[1]):
                original = x[:,i].copy()
                movingAverage = myEwma(x[:,i], 6)[0]
                ewma_cum = movingAverage.cumsum()
                corcoef = np.zeros((3,n)) 
                impulseList = ['original', 'movingAverage', 'ewma_cum']
                tmp_orig = original.copy()
                tmp_imp = movingAverage.copy()
                tmp_impCum = ewma_cum.copy()
                for j in np.arange(0, n):
                    # time lag between impulse (exog. regressor) and
                    # response (timeseries trend)
                    corcoef[0,j] = np.corrcoef(y_imp, tmp_orig)[0,1]
                    corcoef[1,j] = np.corrcoef(y_imp, tmp_imp)[0,1]
                    corcoef[2,j] = np.corrcoef(y_imp, tmp_impCum)[0,1]
                    tmp_orig = shiftRight(tmp_orig)
                    tmp_imp = shiftRight(tmp_imp)
                    tmp_impCum = shiftRight(tmp_impCum)
                
                corcoef = np.nan_to_num(corcoef)
                lag = np.mod(np.argmax(np.abs(corcoef)), n)
                impulseRespType = impulseList[int(np.argmax(corcoef) / n)] 
                x[:,i] = shiftRight(eval(impulseRespType), step = lag)
                               
                if impulseResp == 'auto':
                    xnew[:x.shape[0],i] = x[:, i].copy()
                    xnew[x.shape[0]:,i] = myFunc(x[:, i], histPeriod, fcstPeriod)[1]
                else: 
                    # add manual impulse (x-input)
                    xnew[x.shape[0]:, i] = impulse[:fcstPeriod, i]
                    x_exog = xnew
                    if impulseRespType == 'movingAverage':
                        xnew = myEwma(xnew[:,i], 6)[0]
                    if impulseRespType == 'ewma_cum':
                        xnew = myEwma(xnew[:,i], 6)[0].cumsum()
                    xnew = shiftRight(xnew, step = lag)
                    xnew = xnew.reshape(-1,1)
            ynew = myLr(x, y_imp, xnew)
            
            if title is not None:
                fig, ax = plt.subplots(2)
                ax[0].plot(x[:,0], np.array(y_imp), 'o')
                ax[0].plot(np.array(xnew[:,0]), ynew, '-')        
                ax[0].set_title(title)
                ax[1].plot(timeDim, np.array(y), 'o')
                ax[1].plot(timeDim, np.array(y_imp), '*')
                ax[1].plot(np.hstack([timeDim, timeNew]), ynew, '-', label = method)
                fig.tight_layout()
                plt.savefig(figureDir + 'impulse_' + title + '.png')
                plt.close()               
        else:
            try:
                yfit, ynew = myFunc(y.reshape(-1,), histPeriod, fcstPeriod)
                ynew = np.hstack([yfit, ynew])
            except ValueError:
                ynew = myLr(x, y, xnew)
        
        if title is not None:
            fig, ax = plt.subplots()
            plt.plot(timeDim, np.array(y), 'o')
            plt.plot(np.hstack([timeDim, timeNew]), ynew, '-', label = method)
            ax.set_title(title)
            fig.tight_layout()
            plt.savefig(figureDir + method +'_' + title + '.png')
            plt.close()
        result = ynew.copy()
    
    # add y_trend directly
    result = np.array(result).reshape(-1,)
    if impulseResp == 'manual':
        impulse = impulse[:fcstPeriod, -1].reshape(-1,)
        result[-fcstPeriod:] = result[-fcstPeriod:] + impulse
    
    return result, x_exog, impulseRespType

def isAdditive(timeseries, zeroReplacement = 1):
    '''
    simple function to determine if timeseries is additive or multiplicative 
    based on sum of squares of acf.
    '''
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import acf
    import numpy as np
    ts = timeseries.copy()
    ts[ts == 0] = zeroReplacement
    additive = sm.tsa.seasonal_decompose(ts, model = 'additive', freq = 6)
    multi = sm.tsa.seasonal_decompose(ts, model = 'multiplicative', freq = 6)
    
    add_res = [x for x in additive.resid if ~np.isnan(x)]
    mult_res = [x for x in multi.resid if ~np.isnan(x)]
    
    addAcf = np.sum(np.power(acf(add_res),2))
    multAcf = np.sum(np.power(acf(mult_res),2))
    
    return addAcf <= multAcf
        
        
        
    