#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fit ARIMA to ice cores
Created on Wed May 19 14:48:10 2021

@author: lizz
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg, ar_select_order


## Read in Greenland time series
core_accum_fpath = '/Users/lizz/Documents/GitHub/Data_unsynced/Ice_core_accum/Andersen_etal_2006_Annual_Accumulation_22Mar2011-trimmed.csv'

core_tseries = pd.read_csv(core_accum_fpath, index_col=0, parse_dates=[0])
core_names = core_tseries.columns

series_to_test = core_tseries
core_to_test = core_names[0]

## Pre-process data
anomaly_series = series_to_test - series_to_test.mean()
test_period = anomaly_series.iloc[150:180]

# def adf_test(timeseries):
#     print('A timeseries ready for AR(n) fitting should have ADF test statistic more negative than critical value (reject the null hypothesis).')
#     print ('Results of Dickey-Fuller Test:')
#     dftest = sm.tsa.stattools.adfuller(timeseries, autolag='AIC')
#     dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
#     for key,value in dftest[4].items():
#         dfoutput['Critical Value (%s)'%key] = value
#     print (dfoutput)


# def kpss_test(timeseries):
#     print('A timeseries ready for AR(n) fitting should have KPSS statistic lower than the critical value (fail to reject the null hypothesis).')
#     print ('Results of KPSS Test:')
#     kpsstest = sm.tsa.stattools.kpss(timeseries, regression='c', nlags="auto")
#     kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])
#     for key,value in kpsstest[3].items():
#         kpss_output['Critical Value (%s)'%key] = value
#     print (kpss_output)

# stationarity_test_case = anomaly_series[core_to_test][~np.isnan(anomaly_series[core_to_test])]
# adf_test(stationarity_test_case)
# kpss_test(stationarity_test_case)

## Fit an ARIMA model
for c in core_names:
    print('Core: {}'.format(c))
    ## Fit an ARIMA model
    try:
        lag_acf = sm.tsa.stattools.acf(test_period[c], nlags=10)
        lag_pacf = sm.tsa.stattools.pacf(test_period[c], nlags=10, method='ols')
        upper_confidence = 1.96/np.sqrt(len(test_period[c]))
        p_candidates = np.argwhere(lag_pacf > upper_confidence).squeeze()
        p = p_candidates[p_candidates >0][0] # choose first nonzero value that exceeds upper CI
        q_candidates = np.argwhere(lag_acf > upper_confidence).squeeze()
        q = p_candidates[p_candidates >0][0] 
        d = 0 # no differencing in ARIMA; apply only the differencing above in fracdiff
        # print('Order of ARIMA model: ({},{},{})'.format(p,d,q))
    
        mod = sm.tsa.arima.ARIMA(test_period[c], order=(p,d,q), dates=pd.to_datetime(test_period.index))
        mod_fit = mod.fit()
    
        results = mod.fit()
        print('Bayesian Information Criterion for core {}, ARIMA({}, {}, {}): {}'.format(
            c, p,d,q, mod_fit.bic))
    except:
        continue


# ## Compile BIC for different AR[n] models

# comparison_n = range(1,6)
# bic_per_n = pd.DataFrame(index=comparison_n, columns=core_names)
# for c in core_names:
#     for n in comparison_n:
#         mod = AutoReg(test_period[c], n, seasonal=False)
#         results = mod.fit()
#         bic_per_n[c][n] = results.bic
# b = bic_per_n.astype(float)

# ## Is there a best fit AR[n], as judged by BIC?
# bic_difference = b.transform(lambda x: x-x.min())
# print('Among AR(n) fits with n in {}: \r'.format(comparison_n))
# print('Best fit n per model is as follows: \n',
#   b.idxmin())
# for c in core_names:
#     bic_difference[c] = pd.to_numeric(bic_difference[c])
#     if any(bic_difference[c] > 2):
#         print('Difference is statistically significant for core {}.'.format(c))

#     else:
#         print('No significant difference among fits to {} output'.format(c))
    