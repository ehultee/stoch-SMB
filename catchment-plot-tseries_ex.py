#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot example timeseries and AR(n) fit for a single basin

Created on Mon Apr 26 16:50:02 2021

Edited Mon Nov 14 2022: plot an AR(1) fit for illustration
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
def read_catchment_series(fpath, anomaly=False):
    catchment_fpath = fpath
    catchment_tseries = pd.read_csv(catchment_fpath, index_col=0, parse_dates=[0])
    catchment_tseries.mask(catchment_tseries>1e30)
    anomaly_series = catchment_tseries - catchment_tseries.mean()
    if anomaly:
        return anomaly_series
    else:
        return catchment_tseries

def fit_catchment_series(tseries, which_model, comparison_n=range(1,6), 
                         seasonal=True):
    bic_per_n = pd.DataFrame(index=comparison_n, columns=model_names)
    
    if 'multi' in which_model:  ## allow multi-model mode reporting
        for m in model_names:
            for n in comparison_n:
                mod = AutoReg(tseries[m], n, trend='ct', seasonal=seasonal)
                results = mod.fit()
                bic_per_n[m][n] = results.bic
            bic_per_n[m] = pd.to_numeric(bic_per_n[m])
        best_n = bic_per_n.idxmin().mode()[0]
    else:
        for n in comparison_n:
            mod = AutoReg(tseries[which_model], n, trend='ct', seasonal=seasonal)
            results = mod.fit()
            bic_per_n[which_model][n] = results.bic
        bic_per_n[which_model] = pd.to_numeric(bic_per_n[which_model])
        best_n = bic_per_n[which_model].idxmin()
    
    bic_difference = bic_per_n.transform(lambda x: x-x.min())
    
    return best_n, bic_difference
    

mod_fits = {m: [] for m in model_names}
mod_resids = {m: [] for m in model_names}
basin_i=101
ctmt_fpath = glob.glob('/Users/lizz/Documents/GitHub/Data_unsynced/SMBMIP-processed/*-catchment_{}-tseries.csv'.format(basin_i))[0]
s = read_catchment_series(ctmt_fpath, anomaly=False)
a = s.resample('A').sum()
best_n, bic_diff = fit_catchment_series(a, which_model='multi', seasonal=False)
#best_n = 1 ## force an AR(1) fit
for m in model_names:
    mod = AutoReg(a[m], best_n, trend='ct', seasonal=False).fit()
    fv = mod.fittedvalues
    r = mod.resid
    mod_fits[m] = fv
    mod_resids[m] = r
    

## Plot a single timeseries with AR(n) fit
colors_w = cm.get_cmap('Blues')(np.linspace(0.2, 1, num=len(model_names)))
fig, ax = plt.subplots(figsize=(10,4))
for i,m in enumerate(model_names):
    if 'NHM' in m:
        ax.plot(a[m], label=m, color=colors_w[i])
        ax.plot(mod_fits[m], color='k', alpha=0.8, marker='d', 
                  label='AR({}) fit to {}'.format(best_n, m))
    else:
        pass
ax.set(xlabel='Year', ylabel='Catchment SMB [mm w.e.]',
        # title='Basin {}, all models'.format(basin_i)
        title='Kangerlussuaq catchment, SMB model and AR(n) fit',
        xticks=(np.datetime64('1980-01-01'), np.datetime64('1990-01-01'),
                np.datetime64('2000-01-01'), np.datetime64('2010-01-01')),
        xticklabels=(1980,1990,2000,2010)
        )
ax.legend(bbox_to_anchor=(1.05, 1.0, 0.3, 0.2), loc='upper left')
plt.tight_layout()
plt.show()

fig1, (ax1, ax2, ax3) = plt.subplots(3, figsize=(11,6), sharex=True)
for i,m in enumerate(model_names):
    ax1.plot(s[m], label=m, color=colors_w[i])
    ax2.plot(a[m], label=m, color=colors_w[i])
    if 'NHM' in m:
        ax3.plot(a[m], label=m, color=colors_w[i])
        ax3.plot(mod_fits[m], color='k', alpha=0.8, marker='d', 
                  label='AR({}) fit to {}'.format(best_n, m))
    else:
        pass
ax1.set(ylabel='Monthly SMB [mm w.e.]', 
        title='Kangerlussuaq catchment, SMB models and AR(n) fit')
ax2.set(ylabel='Annual SMB [mm w.e.]')
ax3.set(xlabel='Year', ylabel='Catchment SMB [mm w.e.]',
        # title='Basin {}, all models'.format(basin_i)
        # title='Kangerlussuaq catchment, SMB model and AR(n) fit',
        xticks=(np.datetime64('1980-01-01'), np.datetime64('1990-01-01'),
                np.datetime64('2000-01-01'), np.datetime64('2010-01-01')),
        xticklabels=(1980,1990,2000,2010)
        )
ax1.legend(bbox_to_anchor=(1.05, 1.0, 0.3, 0.2), loc='upper left')
ax3.legend(bbox_to_anchor=(1.05, 1.0, 0.3, 0.2), loc='upper left')
fig1.align_ylabels((ax1, ax2, ax3))
plt.tight_layout()
# plt.show()
# plt.savefig('/Users/lizz/Desktop/20210622-kanger_tseries-large_ex.png', dpi=300)


fig2, (ax4, ax5) = plt.subplots(2, figsize=(11,6), sharex=True)
for i,m in enumerate(model_names):
    ax4.plot(s[m], label=m, color=colors_w[i])
    ax5.plot(a[m], label=m, color=colors_w[i]) 
## highlight AR fit to just one model
example_model = model_names[4]
ax5.plot(mod_fits[example_model], color='k', alpha=0.8, marker='d', 
         label='AR({}) fit to {}'.format(best_n, example_model))

ax4.set(ylabel='Monthly SMB [mm w.e.]', 
        title='Kangerlussuaq catchment, SMB models and AR(n) fit')
#ax5.set(ylabel='Annual SMB [mm w.e.]')
ax5.set(xlabel='Year', ylabel='Catchment SMB [mm w.e.]',
        # title='Basin {}, all models'.format(basin_i)
        # title='Kangerlussuaq catchment, SMB model and AR(n) fit',
        xticks=(np.datetime64('1980-01-01'), np.datetime64('1990-01-01'),
                np.datetime64('2000-01-01'), np.datetime64('2010-01-01')),
        xticklabels=(1980,1990,2000,2010)
        )
#ax4.legend(bbox_to_anchor=(1.05, 1.0, 0.3, 0.2), loc='upper left')
ax5.legend(bbox_to_anchor=(1.05, 1.0, 0.3, 0.2), loc='upper left') ## shared legend
fig2.align_ylabels((ax4, ax5))
plt.tight_layout()
# plt.show()
# plt.savefig('/Users/lizz/Desktop/20210622-kanger_tseries-large_ex.png', dpi=300)
    
    