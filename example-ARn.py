#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stats model fits

Created on Tue Mar 23 12:59:13 2021

@author: lizz
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg, ar_select_order


## Read in Greenland time series
gld_accum_fpath = '/Users/lizz/Documents/GitHub/Data_unsynced/SMBMIP-processed/20210331-accumulation-tseries.csv'
gld_se_fpath = '/Users/lizz/Documents/GitHub/Data_unsynced/SMBMIP-processed/20210331-abl_se-tseries.csv'
gld_nw_fpath = '/Users/lizz/Documents/GitHub/Data_unsynced/SMBMIP-processed/20210331-abl_nw-tseries.csv'
catchment_fpath = '/Users/lizz/Documents/GitHub/Data_unsynced/SMBMIP-processed/20210405-helheim_catchment-tseries.csv'

accum_tseries = pd.read_csv(gld_accum_fpath, index_col=0, parse_dates=[0])
abl_se_tseries = pd.read_csv(gld_se_fpath, index_col=0, parse_dates=[0])
abl_nw_tseries = pd.read_csv(gld_nw_fpath, index_col=0, parse_dates=[0])
catchment_tseries = pd.read_csv(catchment_fpath, index_col=0, parse_dates=[0])

series_to_test = catchment_tseries

model_names = ['ANICE-ITM_Berends', 'CESM_kampenhout', 'dEBM_krebs','HIRHAM_mottram', 
                'NHM-SMAP_niwano', 'RACMO_noel', 'SNOWMODEL_liston']

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

stationarity_test_case = 'ANICE-ITM_Berends'
adf_test(anomaly_series[stationarity_test_case])
kpss_test(anomaly_series[stationarity_test_case])

# ## Fit an AR[n] model
# n = 5
# for m in model_names:
#     mod = AutoReg(accum_tseries[m], n, seasonal=True)
#     results = mod.fit()
#     print('Bayesian Information Criterion for model {}, AR({}): {}'.format(
#         m, n, results.bic))


## Compile BIC for different AR[n] models
comparison_n = range(1,6)
bic_per_n = pd.DataFrame(index=comparison_n, columns=model_names)
for n in comparison_n:
    for m in model_names:
        mod = AutoReg(catchment_tseries[m], n, seasonal=True)
        results = mod.fit()
        bic_per_n[m][n] = results.bic

## Is there a best fit AR[n], as judged by BIC?
bic_difference = bic_per_n.transform(lambda x: x-x.min())
print('Among AR(n) fits with n in {}: \r'.format(comparison_n))
for m in model_names:
    bic_per_n[m] = pd.to_numeric(bic_per_n[m]) # needed for idxmin
    if any(bic_difference[m] > 2):
        print('Best fit n per model is as follows: \n'.format(comparison_n),
              bic_per_n.idxmin())
    else:
        print('No significant difference among fits to {} output'.format(
            m, comparison_n))
    