#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Construct spatially informative noise from Graphical Lasso covar matrix

Created on Wed Jun 16 15:47:29 2021

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

def find_AR_residuals(tseries, which_model, chosen_n=1, 
                         seasonal=False):
    mod = AutoReg(tseries[which_model], chosen_n, trend='ct', seasonal=seasonal)
    results = mod.fit()
    resids = results.resid
    
    return resids
    


ar_resids = []
ts_toplot = []
for i in range(1, 200):
    # print(i)
    ctmt_fpath = glob.glob('/Users/lizz/Documents/GitHub/Data_unsynced/SMBMIP-processed/*-catchment_{}-tseries.csv'.format(i))[0]
    s = read_catchment_series(ctmt_fpath)
    a = s.resample('A').sum()
    ts_toplot.append(a)
    r = find_AR_residuals(a, which_model=model_names[0], chosen_n=4, seasonal=False)
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
spatial_noise = L @ N ## matrix multiply these two

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

highlight_catchment_name, highlight_catchment_id = 'KANGERLUSSUAQ', 101
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