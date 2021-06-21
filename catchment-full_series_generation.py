#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate example time series of SMB for a given catchment
Pre-select AR(1) for annual variability portion
Include noise from sparse covar method

Created on Mon Jun 21 11:54:52 2021

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
mod_fits = {m: [] for m in model_names}
mods = {m: [] for m in model_names}

ctmt_fpath = glob.glob('/Users/lizz/Documents/GitHub/Data_unsynced/SMBMIP-processed/*-catchment_{}-tseries.csv'.format(highlight_catchment_id))[0]
s = read_catchment_series(ctmt_fpath, anomaly=True)
a = s.resample('A').sum()
best_n, _ = fit_catchment_series(a, which_model='multi', seasonal=False)
for m in model_names:
    mod = AutoReg(a[m], best_n, trend='ct', seasonal=False).fit()
    fv = mod.fittedvalues
    r = mod.resid
    mod_fits[m] = fv
    mods[m] = mod


## Residuals with spatial information
ar_resids = []
ts_toplot = []
for i in range(1, 200):
    # print(i)
    ctmt_fpath = glob.glob('/Users/lizz/Documents/GitHub/Data_unsynced/SMBMIP-processed/*-catchment_{}-tseries.csv'.format(i))[0]
    s = read_catchment_series(ctmt_fpath, anomaly=True)
    a = s.resample('A').sum()
    ts_toplot.append(a)
    r = find_AR_residuals(a, which_model=highlight_model, chosen_n=best_n, seasonal=False)
    ar_resids.append(r)

ar_resids -= np.mean(ar_resids, axis=0) # normalize
emp_C = np.corrcoef(ar_resids)

np.random.seed(0)
X = np.random.multivariate_normal(mean=np.zeros(len(ar_resids)), cov=emp_C, size=len(ar_resids[0]))

gl_model = GraphicalLassoCV()
gl_model.fit(X)
cov_ = gl_model.covariance_

## Now take the Cholesky decomposition of the sparse cov matrix and generate noise with it
L = scipy.linalg.cholesky(cov_, lower=True)
N = np.random.normal(size=np.shape(ar_resids)) # draws from normal dist

D = np.diag(np.std(ar_resids,1)) ## diagonal matrix of standard devs
scaled_noise = D @ L @ N


## Plot example time series of resids
## set matplotlib font size defaults
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

noise_realizations = []
## Several realizations
for j in range(100):
    Nj = np.random.normal(size=np.shape(ar_resids))
    noise_j = D @ L @ Nj
    noise_realizations.append(noise_j[highlight_catchment_id-1])
noise_colors = cm.get_cmap('Blues')

fig, ax = plt.subplots(1, figsize=(10,6))
for k in range(len(noise_realizations)):
    ax.plot(range(1980,2012), noise_realizations[k], 
            color=noise_colors(k/len(noise_realizations)), alpha=0.5)
ax.plot(range(1980, 2012), ar_resids[highlight_catchment_id-1], color='k')
ax.set(xlabel='Year', ylabel='Catchment SMB residual [mm w.e.]')
plt.show()

## Plot a single timeseries with AR(n) fit
colors_w = cm.get_cmap('Blues')(np.linspace(0.2, 1, num=len(model_names)))
fig1, ax1 = plt.subplots(figsize=(10,4))
for i,m in enumerate(model_names):
    if 'ANICE' in m:
        ax1.plot(a[m], label=m, color=colors_w[i])
        ax1.plot(mod_fits[m], color='k', alpha=0.8, marker='d', 
                  label='AR({}) fit to {}'.format(1, m))
    else:
        pass
ax1.set(xlabel='Year', ylabel='Catchment SMB [mm w.e.]',
        # title='Basin {}, all models'.format(basin_i)
        title='Kangerlussuaq catchment, SMB model and AR(n) fit',
        xticks=(np.datetime64('1980-01-01'), np.datetime64('1990-01-01'),
                np.datetime64('2000-01-01'), np.datetime64('2010-01-01')),
        xticklabels=(1980,1990,2000,2010)
        )
ax1.legend(bbox_to_anchor=(1.05, 1.0, 0.3, 0.2), loc='upper left')
plt.tight_layout()
plt.show()

## Plot a sum of the two
fig2, ax2 = plt.subplots(figsize=(10,6))
for k in range(len(noise_realizations)):
    ax2.plot(mods[highlight_model].predict(best_n, len(a)-1)
             +noise_realizations[k], 
            color=noise_colors(k/len(noise_realizations)), alpha=0.5)
ax2.plot(ts_toplot[highlight_catchment_id-1][highlight_model], color='k',
         label='{} output'.format(highlight_model))
ax2.plot(np.NaN, np.NaN, color=noise_colors(0.5), label='Stochastic realizations')
ax2.set(xlabel='Year', ylabel=r'Catchment SMB anomaly [mm w.e. a$^{-1}$]',
        xticks=(np.datetime64('1980-01-01'), np.datetime64('1990-01-01'),
                np.datetime64('2000-01-01'), np.datetime64('2010-01-01')),
        xticklabels=(1980,1990,2000,2010))
ax2.legend(loc='best')
plt.show()