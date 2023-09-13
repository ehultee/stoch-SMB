#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create catchment integrated SMB
Created on Mon Mar 29 15:00:45 2021

@author: lizz
"""

from netCDF4 import Dataset
import numpy as np
import pandas as pd
# import pyproj as pyproj
# from scipy import interpolate
import csv
import datetime
import time


model_names = ['ANICE-ITM_Berends', 'CESM_kampenhout', 'dEBM_krebs','HIRHAM_mottram', 
                'NHM-SMAP_niwano', 'RACMO_noel', 'SNOWMODEL_liston']
years = range(1980,2013)

## Select single points to extract example series (by index)
accum_pt_xi, accum_pt_yi = 926, 1660
abl_nw_xi, abl_nw_yi = 411, 2020
abl_se_xi, abl_se_yi = 909, 682

## Create data frames to store per-model time series
start_date = datetime.datetime(years[0],1,1)
end_date = datetime.datetime(years[-1],12,31)
dates = pd.date_range(start=start_date, end=end_date, freq='M')
df_accum = pd.DataFrame(columns=model_names, index=dates)
df_se = pd.DataFrame(columns=model_names, index=dates)
df_nw = pd.DataFrame(columns=model_names, index=dates)

for m in model_names:
    t0 = time.time()
    if m=='CESM_kampenhout':
        vname = 'SMBCORR'
    else:
        vname = 'SMBcorr'
    for y in years:
        fpath = '/Users/lizz/Documents/GitHub/Data_unsynced/SMBMIP/{}-monthly-ERA-Interim-{}.nc'.format(m, y)
        fh = Dataset(fpath, mode='r')
        smb_m = fh.variables[vname][:].copy()
        fh.close()
        d_subset = [d for d in dates if d.year==y]
        for i in range(len(smb_m)):
            df_accum[m][d_subset[i]] = smb_m[i][accum_pt_yi, accum_pt_xi]
            df_se[m][d_subset[i]] = smb_m[i][abl_se_yi, abl_se_xi]
            df_nw[m][d_subset[i]] = smb_m[i][abl_nw_yi, abl_nw_xi]
    t1 = time.time()
    print('Finished processing model {} in time {}'.format(m, t1-t0))

csv_accum = '/Users/lizz/Documents/GitHub/Data_unsynced/SMBMIP-processed/{}-accumulation-tseries.csv'.format(datetime.date.today().strftime('%Y%m%d'))
csv_se ='/Users/lizz/Documents/GitHub/Data_unsynced/SMBMIP-processed/{}-abl_se-tseries.csv'.format(datetime.date.today().strftime('%Y%m%d'))
csv_nw ='/Users/lizz/Documents/GitHub/Data_unsynced/SMBMIP-processed/{}-abl_nw-tseries.csv'.format(datetime.date.today().strftime('%Y%m%d'))

df_accum.to_csv(csv_accum)
df_se.to_csv(csv_se)
df_nw.to_csv(csv_nw)
