#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate monthly series misfit and compare with annual misfit for example catchment

Created on Mon Jun 21 16:15:49 2021

@author: lizz
"""

from sklearn.covariance import GraphicalLassoCV
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from statsmodels.tsa.ar_model import AutoReg
import scipy.linalg
import glob


model_names = ['ANICE-ITM_Berends', 'CESM_kampenhout', 'dEBM_krebs','HIRHAM_mottram', 
                'NHM-SMAP_niwano', 'RACMO_noel', 'SNOWMODEL_liston']
highlight_model = 'ANICE-ITM_Berends'
highlight_catchment_name, highlight_catchment_id = 'KANGERLUSSUAQ', 101

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

def find_AR_residuals(tseries, which_model, chosen_n=1, 
                         seasonal=False):
    mod = AutoReg(tseries[which_model], chosen_n, trend='ct', seasonal=seasonal)
    results = mod.fit()
    resids = results.resid
    
    return resids

## Time series from AR(n) fits
monthly_resids = {m: [] for m in model_names}
annual_resids = {m: [] for m in model_names}

ctmt_fpath = glob.glob('/Users/lizz/Documents/GitHub/Data_unsynced/SMBMIP-processed/*-catchment_{}-tseries.csv'.format(highlight_catchment_id))[0]
s = read_catchment_series(ctmt_fpath, anomaly=True)
a = s.resample('A').sum()
best_n, _ = fit_catchment_series(a, which_model='multi', seasonal=False)
for m in model_names:
    mod_m = AutoReg(s[m], best_n, trend='ct', seasonal=True).fit()
    mod_a = AutoReg(a[m], best_n, trend='ct', seasonal=False).fit()
    r_monthly = mod_m.resid
    r_annual = mod_a.resid
    monthly_resids[m] = r_monthly
    annual_resids[m] = r_annual
    
aggregated_resids = pd.DataFrame.from_dict(monthly_resids).resample('A').sum()
annual_resids = pd.DataFrame.from_dict(annual_resids)

fig, ax = plt.subplots(figsize=(8,6))
ax.plot(aggregated_resids[model_names[5]], color='r', label='Monthly')
ax.fill_between(x=aggregated_resids.index, y1=aggregated_resids[model_names[5]], y2=0, 
                color='r', hatch='/', alpha=0.5)
ax.plot(annual_resids[model_names[5]], color='k', label='Annual')
ax.fill_between(x=annual_resids.index, y1=annual_resids[model_names[5]], y2=0, 
                color='k', hatch='|', alpha=0.5)
ax.axhline(0)
ax.legend(loc='best')
ax.set(xlabel='Year', ylabel=r'AR(n) misfit [mm w.e. a$^{-1}$]',
        xticks=(np.datetime64('1980-01-01'), np.datetime64('1990-01-01'),
                np.datetime64('2000-01-01'), np.datetime64('2010-01-01')),
        xticklabels=(1980,1990,2000,2010))
plt.show()