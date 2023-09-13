#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ice core AR(n) fits
Created on Wed May 12 14:00:10 2021

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

## Pre-process data
anomaly_series = series_to_test - series_to_test.mean()

def adf_test(timeseries):
    print('A timeseries ready for AR(n) fitting should have ADF test statistic more negative than critical value (reject the null hypothesis).')
    print ('Results of Dickey-Fuller Test:')
    dftest = sm.tsa.stattools.adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)


def kpss_test(timeseries):
    print('A timeseries ready for AR(n) fitting should have KPSS statistic lower than the critical value (fail to reject the null hypothesis).')
    print ('Results of KPSS Test:')
    kpsstest = sm.tsa.stattools.kpss(timeseries, regression='c', nlags="auto")
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])
    for key,value in kpsstest[3].items():
        kpss_output['Critical Value (%s)'%key] = value
    print (kpss_output)

stationarity_test_case = anomaly_series[core_names[0]][~np.isnan(anomaly_series[core_names[0]])]
adf_test(stationarity_test_case)
kpss_test(stationarity_test_case)

# ## Fit an AR[n] model
# n = 1
# for m in model_names:
#     mod = AutoReg(anomaly_series[m], n, seasonal=True)
#     results = mod.fit()
#     print('Bayesian Information Criterion for model {}, AR({}): {}'.format(
#         m, n, results.bic))


## Compile BIC for different AR[n] models
test_period = anomaly_series.iloc[150:180]

comparison_n = range(1,6)
bic_per_n = pd.DataFrame(index=comparison_n, columns=core_names)
for c in core_names:
    for n in comparison_n:
        mod = AutoReg(test_period[c], n, seasonal=False)
        results = mod.fit()
        bic_per_n[c][n] = results.bic
b = bic_per_n.astype(float)

## Is there a best fit AR[n], as judged by BIC?
bic_difference = b.transform(lambda x: x-x.min())
print('Among AR(n) fits with n in {}: \r'.format(comparison_n))
print('Best fit n per model is as follows: \n',
  b.idxmin())
for c in core_names:
    bic_difference[c] = pd.to_numeric(bic_difference[c])
    if any(bic_difference[c] > 2):
        print('Difference is statistically significant for core {}.'.format(c))

    else:
        print('No significant difference among fits to {} output'.format(c))
    