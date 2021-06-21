#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Construct spatially informative noise from Graphical Lasso covar matrix

Created on Wed Jun 16 15:47:29 2021

@author: lizz
"""

from sklearn.covariance import GraphicalLassoCV, ledoit_wolf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
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
    r = find_AR_residuals(a, which_model=model_names[0], seasonal=False)

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
N = np.random.normal(size=np.shape(ar_resids))
spatial_noise = L @ N ## matrix multiply these two

D = np.diag(np.std(ar_resids,1)) ## diagonal matrix of standard devs
scaled_noise = D @ L @ N