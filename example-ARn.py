#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stats model fits

Created on Tue Mar 23 12:59:13 2021

@author: lizz
"""

from netCDF4 import Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyproj as pyproj
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from scipy import signal

# helheim_smb_fpath = '/Users/lizz/Documents/GitHub/Data_unsynced/Helheim-processed/smb_rec._.BN_RACMO2.3p2_ERA5_3h_FGRN055.1km.MM.csv'
# helheim_smb = pd.read_csv(helheim_smb_fpath, index_col=0, parse_dates=[0])

## Read in Greenland time series
gld_accum_fpath = '/Users/lizz/Documents/GitHub/Data_unsynced/SMBMIP-processed/20210331-accumulation-tseries.csv'
gld_se_fpath = '/Users/lizz/Documents/GitHub/Data_unsynced/SMBMIP-processed/20210331-abl_se-tseries.csv'
gld_nw_fpath = '/Users/lizz/Documents/GitHub/Data_unsynced/SMBMIP-processed/20210331-abl_nw-tseries.csv'

accum_tseries = pd.read_csv(gld_accum_fpath, index_col=0, parse_dates=[0])
abl_se_tseries = pd.read_csv(gld_se_fpath, index_col=0, parse_dates=[0])
abl_nw_tseries = pd.read_csv(gld_nw_fpath, index_col=0, parse_dates=[0])

model_names = ['ANICE-ITM_Berends', 'CESM_kampenhout', 'dEBM_krebs','HIRHAM_mottram', 
                'NHM-SMAP_niwano', 'RACMO_noel', 'SNOWMODEL_liston']

# ## Fit an AR[n] model
# n = 5
# for m in model_names:
#     mod = AutoReg(accum_tseries[m], n, seasonal=True)
#     results = mod.fit()
#     print('Bayesian Information Criterion for model {}, AR({}): {}'.format(
#         m, n, results.bic))


## Compile BIC for different AR[n] models
comparison_n = range(1,6)
bic_per_n = pd.DataFrame(index=comparison_n, columns=model_names)
for n in comparison_n:
    for m in model_names:
        mod = AutoReg(abl_nw_tseries[m], n, seasonal=True)
        results = mod.fit()
        bic_per_n[m][n] = results.bic

## Which n is best fit for each SMB model?
for m in model_names:
    bic_per_n[m] = pd.to_numeric(bic_per_n[m]) # needed for idxmin
print('Among AR(n) models with n in {}, best fit n per model is as follows: \n'.format(comparison_n),
      bic_per_n.idxmin())
    