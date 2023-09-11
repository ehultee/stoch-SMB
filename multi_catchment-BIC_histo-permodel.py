#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summarize best-fit AR(n) separated by SMB process model

Created on Sat Aug 19 16:07:46 2023

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

def fit_catchment_series(tseries, which_model, comparison_n=range(0,6), 
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
            mod = AutoReg(tseries[which_model], n, trend='ct', 
                          seasonal=seasonal)
            results = mod.fit()
            bic_per_n[which_model][n] = results.bic
        bic_per_n[which_model] = pd.to_numeric(bic_per_n[which_model])
        best_n = bic_per_n[which_model].idxmin()
    
    bic_difference = bic_per_n.transform(lambda x: x-x.min())
    
    return best_n, bic_difference
    



Ns_annual_bymodel = {m: [] for m in model_names}
bic_differences = []
# bic_vs_ar1 = []
non1_count = 0
ts_toplot = []
for i in range(0, 260):
    ctmt_fpath = glob.glob('/Users/lizz/Documents/GitHub/Data_unsynced/SMBMIP-processed/*-catchment_{}_mean-tseries.csv'.format(i))[0]
    s = read_catchment_series(ctmt_fpath)
    a = s.resample('A').sum()
    if a.isna().sum().sum()>0:
        print('NaNs found in catchment {}'.format(i))
        continue
    else:
        for m in model_names:
            n2, b = fit_catchment_series(a, which_model=m, 
                                         comparison_n=range(0,6),seasonal=False)
            # if n2!=1: # note BIC difference for non-AR(1) choices
            #     non1_count +=1
            #     bva = b.loc[1] - b.loc[n2] # difference between chosen fit and AR(1)
            #     # for diffmag in b.nsmallest(2, columns=b.columns).max():
            #     #     bic_differences.append(diffmag)
            #     for m in model_names:
            #         bic_vs_ar1.append(bva[m])
            
            Ns_annual_bymodel[m].append(n2)
# bd = np.asarray(bic_differences)[np.isfinite(bic_differences)]
# bvar1 = np.asarray(bic_vs_ar1)[np.isfinite(bic_vs_ar1)]

fig, axs = plt.subplots(7,1)
for i, m in enumerate(model_names):
    ax = axs.ravel()[i]
    ax.hist(Ns_annual_bymodel[m])
    ax.set(xlabel='n', # ylabel=m,
            xticks=(0, 1, 2, 3, 4, 5),
            title=m
            )
fig.supylabel('Basins for which n is best AR(n)')
plt.show()

# fig, ax = plt.subplots()
# ax.hist(bvar1)
# ax.set(xlabel='BIC difference between best and AR(1) fit', ylabel='Instances',
#         # xticks=(1, 2, 3, 4, 5),
#         # title='SMB model: {}'.format(model_names[6])
#         title='Single SMB model fits to annual SMB'
#         )
# plt.show()

# m = model_names[0]
# fig1, ax1 = plt.subplots()
# for i in range(len(ts_toplot)):
#     ax1.plot(ts_toplot[i][m])
# ax1.set(xlabel='Year', ylabel='Catchment SMB [mm w.e.]',
#         title='{} basins, model {}'.format(len(ts_toplot), m))
# plt.show()


# ## nice single-basin visualization
# colors_w = cm.get_cmap('Blues')(np.linspace(0.2, 1, num=len(model_names)))
# basin_i = 101
# fig2, ax2 = plt.subplots()
# for i,m in enumerate(model_names):
#     ax2.plot(ts_toplot[basin_i][m], label=m, color=colors_w[i])
# ax2.set(xlabel='Year', ylabel='Catchment SMB [mm w.e.]',
#         # title='Basin {}, all models'.format(basin_i)
#         title='Kangerlussuaq catchment, all SMB models',
#         xticks=(np.datetime64('1980-01-01'), np.datetime64('1990-01-01'),
#                 np.datetime64('2000-01-01'), np.datetime64('2010-01-01')),
#         xticklabels=(1980,1990,2000,2010)
#         )
# ax2.legend(bbox_to_anchor=(1.05, 1.0, 0.3, 0.2), loc='upper left')
# plt.tight_layout()
# plt.show()
    