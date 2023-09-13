#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fit annual var in single catchment example

Created on Tue Apr 27 16:40:46 2021

@author: lizz
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from matplotlib import cm
import glob

model_names = ['ANICE-ITM_Berends', 'CESM_kampenhout', 'dEBM_krebs','HIRHAM_mottram', 
                'NHM-SMAP_niwano', 'RACMO_noel', 'SNOWMODEL_liston']

## Read in time series
def read_catchment_series(fpath, anomaly=True):
    catchment_fpath = fpath
    catchment_tseries = pd.read_csv(catchment_fpath, index_col=0, parse_dates=[0])
    catchment_tseries.mask(catchment_tseries>1e30)
    anomaly_series = catchment_tseries - catchment_tseries.mean()
    if anomaly:
        return anomaly_series
    else:
        return catchment_tseries

def fit_catchment_series(tseries, comparison_n=range(1,6), multi_model_mode=True, 
                         strength_of_fit=False, seasonal=True):
    bic_per_n = pd.DataFrame(index=comparison_n, columns=model_names)
    for n in comparison_n:
        for m in model_names:
            mod = AutoReg(tseries[m], n, trend='ct', seasonal=seasonal)
            results = mod.fit()
            bic_per_n[m][n] = results.bic
    
    if multi_model_mode:
        for m in model_names:
            bic_per_n[m] = pd.to_numeric(bic_per_n[m]) # needed for idxmin
        best_n = bic_per_n.idxmin().mode()[0]

    return best_n
    

mod_fits = {m: [] for m in model_names}
mod_resids = {m: [] for m in model_names}
basin_i=101
ctmt_fpath = glob.glob('/Users/lizz/Documents/GitHub/Data_unsynced/SMBMIP-processed/*-catchment_{}-tseries.csv'.format(basin_i))[0]
s = read_catchment_series(ctmt_fpath)
a = s.resample('A').sum()
best_n = fit_catchment_series(a, seasonal=False)
for m in model_names:
    mod = AutoReg(a[m], best_n, trend='ct', seasonal=False).fit()
    fv = mod.fittedvalues
    r = mod.resid
    mod_fits[m] = fv
    mod_resids[m] = r
    

## Plot a single timeseries with AR(n) fit
colors_w = cm.get_cmap('Blues')(np.linspace(0.2, 1, num=len(model_names)))
fig2, ax2 = plt.subplots(figsize=(10,4))
for i,m in enumerate(model_names):
    if 'NHM' in m:
        ax2.plot(a[m], label=m, color=colors_w[i])
        ax2.plot(mod_fits[m], color='k', alpha=0.8, marker='d', 
                  label='AR({}) fit to {}'.format(best_n, m))
    else:
        pass
ax2.set(xlabel='Year', ylabel='Catchment SMB [mm w.e.]',
        # title='Basin {}, all models'.format(basin_i)
        title='Kangerlussuaq catchment, SMB model and AR(n) fit',
        xticks=(np.datetime64('1980-01-01'), np.datetime64('1990-01-01'),
                np.datetime64('2000-01-01'), np.datetime64('2010-01-01')),
        xticklabels=(1980,1990,2000,2010)
        )
ax2.legend(bbox_to_anchor=(1.05, 1.0, 0.3, 0.2), loc='upper left')
plt.tight_layout()
plt.show()

fig3, (ax3, ax4) = plt.subplots(2, figsize=(10,6), sharex=True)
for i,m in enumerate(model_names):
    ax3.plot(a[m], label=m, color=colors_w[i])
    if 'NHM' in m:
        ax4.plot(a[m], label=m, color=colors_w[i])
        ax4.plot(mod_fits[m], color='k', alpha=0.8, marker='d', 
                  label='AR({}) fit to {}'.format(best_n, m))
    else:
        pass
ax3.set(ylabel='Catchment SMB [mm w.e.]', 
        title='Kangerlussuaq catchment, SMB models and AR(n) fit')
ax4.set(xlabel='Year', ylabel='Catchment SMB [mm w.e.]',
        # title='Basin {}, all models'.format(basin_i)
        # title='Kangerlussuaq catchment, SMB model and AR(n) fit',
        xticks=(np.datetime64('1980-01-01'), np.datetime64('1990-01-01'),
                np.datetime64('2000-01-01'), np.datetime64('2010-01-01')),
        xticklabels=(1980,1990,2000,2010)
        )
ax3.legend(bbox_to_anchor=(1.05, 1.0, 0.3, 0.2), loc='upper left')
ax4.legend(bbox_to_anchor=(1.05, 1.0, 0.3, 0.2), loc='upper left')
plt.tight_layout()
plt.show()
    