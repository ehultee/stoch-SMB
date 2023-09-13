#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ice core ACF
Created on Wed May 19 14:05:10 2021

@author: lizz
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics import tsaplots


## Read in Greenland time series
core_accum_fpath = '/Users/lizz/Documents/GitHub/Data_unsynced/Ice_core_accum/Andersen_etal_2006_Annual_Accumulation_22Mar2011-trimmed.csv'

core_tseries = pd.read_csv(core_accum_fpath, index_col=0, parse_dates=[0])
core_names = core_tseries.columns

series_to_test = core_tseries
core_to_test = core_names[3]

## Pre-process data
anomaly_series = series_to_test - series_to_test.mean()

test_series = anomaly_series[core_to_test][~np.isnan(anomaly_series[core_to_test])]

fig, ax = plt.subplots()
tsaplots.plot_acf(test_series, ax=ax, lags=100, zero=False)
ax.set(xlabel='Lag [yr]', ylabel='Autocorr.', title='{} Core'.format(core_to_test.split()[0]))
plt.show()

## Plot all 5 cores together
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, figsize=(10, 12), 
                                              sharex=True, sharey=True, tight_layout=True)
for c, ax in zip(core_names, (ax1,ax2,ax3,ax4,ax5)):
    try:
        ts = anomaly_series[c][~np.isnan(anomaly_series[c])]
        tsaplots.plot_pacf(ts, ax=ax, lags=100, zero=False)
        ax.set(ylabel='Partial acorr.', title='{} Core'.format(c.split()[0]))
    except:
        continue
ax5.set(xlabel='Lag [yr]')

## Plot 4 cores together - 500 years. Milcent not long enough
fig, (ax6, ax7, ax8, ax9) = plt.subplots(4, figsize=(10, 12), 
                                              sharex=True, sharey=True, tight_layout=True)
for c, ax in zip((name for name in core_names if 'Milcent' not in name), 
                 (ax6,ax7,ax8,ax9)):
    try:
        ts = anomaly_series[c][~np.isnan(anomaly_series[c])]
        tsaplots.plot_pacf(ts, ax=ax, lags=500, zero=False)
        ax.set(ylabel='Partial acorr.', title='{} Core'.format(c.split()[0]))
    except:
        continue
ax5.set(xlabel='Lag [yr]')

## Plot with a different coloring when significant
fig, ax = plt.subplots(figsize=(10,5))
tsaplots.plot_pacf(test_series, ax=ax, lags=100, zero=False)
ax.set(xlabel='Lag [yr]', ylabel='Autocorr.', title='{} Core'.format(core_to_test.split()[0]))
plt.show()
