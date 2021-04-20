#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare best fit AR(n) for several catchments

Created on Tue Apr 20 11:47:48 2021

@author: lizz
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
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
                         strength_of_fit=False):
    bic_per_n = pd.DataFrame(index=comparison_n, columns=model_names)
    for n in comparison_n:
        for m in model_names:
            mod = AutoReg(tseries[m], n, trend='ct', seasonal=True)
            results = mod.fit()
            bic_per_n[m][n] = results.bic
    
    if multi_model_mode:
        for m in model_names:
            bic_per_n[m] = pd.to_numeric(bic_per_n[m]) # needed for idxmin
        best_n = bic_per_n.idxmin().mode()[0]
    
    # if strength_of_fit:
    #     ## test whether the fit is actually stronger than 
    #     bic_difference = bic_per_n.transform(lambda x: x-x.min())
    #     for m in model_names:
    #         if any(bic_difference[m] > 2): # there is an actual best fit
    #             print('Best fit n per model is as follows: \n',
    #                   bic_per_n.idxmin())
    #         else:
    #             print('No significant difference among fits to {} output'.format(
    #                 m, comparison_n))
    
    return best_n
    

Ns = []
for i in range(150, 200):
    print(i)
    ctmt_fpath = glob.glob('/Users/lizz/Documents/GitHub/Data_unsynced/SMBMIP-processed/*-catchment_{}-tseries.csv'.format(i))[0]
    s = read_catchment_series(ctmt_fpath)
    n = fit_catchment_series(s)
    Ns.append(n)

fig, ax = plt.subplots()
ax.hist(Ns)
ax.set(xlabel='n', ylabel='Basins for which n is best AR(n)',
       xticks=(1, 2, 3, 4, 5)
       )
plt.show()