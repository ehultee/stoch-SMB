#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export residualsof AR(1) fit and apply graphical lasso
Created on Mon May 24 12:39:22 2021

@author: lizz
"""
from sklearn.covariance import GraphicalLassoCV, ledoit_wolf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

def find_AR_residuals(tseries, which_model, chosen_n=1, 
                         seasonal=False):
    mod = AutoReg(tseries[which_model], chosen_n, trend='ct', seasonal=seasonal)
    results = mod.fit()
    resids = results.resid
    
    return resids
    


ar_resids = []
ts_toplot = []
for i in range(1, 10):
    # print(i)
    ctmt_fpath = glob.glob('/Users/lizz/Documents/GitHub/Data_unsynced/SMBMIP-processed/*-catchment_{}-tseries.csv'.format(i))[0]
    s = read_catchment_series(ctmt_fpath)
    a = s.resample('A').sum()
    ts_toplot.append(a)
    r = find_AR_residuals(a, which_model=model_names[0], seasonal=False)

    ar_resids.append(r)

emp_C = np.cov(ar_resids)
emp_C -= emp_C.mean() # normalize
emp_C /= emp_C.std()

np.random.seed(0)
X = np.random.multivariate_normal(mean=np.zeros(len(ar_resids)), cov=emp_C, size=len(emp_C))

gl_model = GraphicalLassoCV()
gl_model.fit(X)
cov_ = gl_model.covariance_

lw_cov_, _ = ledoit_wolf(X)
lw_prec_ = np.linalg.inv(lw_cov_)

# #############################################################################
# Plot the results
plt.figure(figsize=(10, 6))
plt.subplots_adjust(left=0.02, right=0.98)

# plot the covariances
covs = [('Empirical', emp_C), ('Ledoit-Wolf', lw_cov_),
        ('GraphicalLassoCV', cov_)]
vmax = cov_.max()
for i, (name, this_cov) in enumerate(covs):
    plt.subplot(1, 3, i + 1)
    plt.imshow(this_cov, interpolation='nearest', vmin=-vmax, vmax=vmax,
               cmap=plt.cm.RdBu_r)
    plt.xticks(())
    plt.yticks(())
    plt.title('%s covariance' % name)