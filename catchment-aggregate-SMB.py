#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 17:30:39 2021

@author: lizz
"""

from shapely.geometry import MultiPoint
from shapely.ops import triangulate
import shapefile
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

## Select region of catchment (by index)


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


###
## CATCHMENT AGGREGATION
## Example: Helheim Glacier
###

# ## Example field read in for visualization
# nhm_smb_path = '/Users/lizz/Documents/GitHub/Data_unsynced/SMBMIP/NHM-SMAP_niwano-monthly-ERA-Interim-1980.nc'
# fh2 = Dataset(nhm_smb_path, mode='r')
# xlon_nhm = fh2.variables['LON'][:].copy() #x-coord (latlon)
# ylat_nhm = fh2.variables['LAT'][:].copy() #y-coord (latlon)
# ts_nhm = fh2.variables['time'][:].copy()
# smb_nhm = fh2.variables['SMBcorr'][:].copy()
# fh2.close()

# debm_smb_path = '/Users/lizz/Documents/GitHub/Data_unsynced/SMBMIP/dEBM_krebs-monthly-ERA-Interim-1980.nc'
# fh3 = Dataset(debm_smb_path, mode='r')
# xlon_debm = fh3.variables['LON'][:].copy() #x-coord (latlon)
# ylat_debm = fh3.variables['LAT'][:].copy() #y-coord (latlon)
# ts_debm = fh3.variables['time'][:].copy()
# smb_debm = fh3.variables['SMBcorr'][:].copy()
# fh3.close()

# # ## Read in Mankoff catchment from shapefile
# # catchment_fn = '/Users/lizz/Documents/GitHub/Data_unsynced/Helheim-processed/catchment/helheim_ice_catchment_mankoff'
# # sf = shapefile.Reader(catchment_fn) 
# # outline = sf.shapes()[0]

# # ## Aid reprojection with BedMachine background
# # gl_bed_path ='/Users/lizz/Documents/GitHub/Data_unsynced/BedMachine-Greenland/BedMachineGreenland-2017-09-20.nc'
# # fh = Dataset(gl_bed_path, mode='r')
# # xx = fh.variables['x'][:].copy() #x-coord (polar stereo (70, 45))
# # yy = fh.variables['y'][:].copy() #y-coord
# # s_raw = fh.variables['surface'][:].copy() #surface elevation
# # fh.close()

# # # ## Select topo in area of Helheim
# # # xl1, xr1 = 5000, 7000
# # # yt1, yb1 = 10000, 14000
# # # x_hel = xx[xl1:xr1:20] # sample at ~3 km resolution
# # # y_hel = yy[yt1:yb1:20]

# # # ## Select large area of SMB around Helheim
# # # xl2, xr2 = 150, 300
# # # yt2, yb2 = 305, 410
# # # x_lon_h = x_lon[yt2:yb2, xl2:xr2] 
# # # y_lat_h = y_lat[yt2:yb2, xl2:xr2] # resolution ~0.015 deg or abut 2.5 km

# # # wgs84 = pyproj.Proj("+init=EPSG:4326") # LatLon with WGS84 datum used by HIRHAM
# # # psn_gl = pyproj.Proj("+init=epsg:3413") # Polar Stereographic North used by BedMachine (as stated in NetCDF header)
# # # xs, ys = pyproj.transform(wgs84, psn_gl, x_lon_h, y_lat_h)
# # # Xmat, Ymat = np.meshgrid(x_hel, y_hel) # BedMachine coords from helheim-profiles

# # # SMB_dict = {} #set up a dictionary of surface mass balance fields indexed by year
# # # time_indices = range(311, 444) # go from Jan 2006 to Dec 2016 in monthly series
# # # smb_dates = pd.date_range(start='2006-01-01', end='2016-12-31', periods=len(time_indices))
# # # # smb_d = [d.utctimetuple() for d in smb_dates]
# # # # dates_interp = [ice.timeutils.datestr2tdec(d[0], d[1], d[2]) for d in smb_d]
# # # for t,d in zip(time_indices, smb_dates):
# # #     smb_t = smb_raw[t][0][::-1, ::][yt2:yb2, xl2:xr2]
# # #     regridded_smb_t = interpolate.griddata((xs.ravel(), ys.ravel()), smb_t.ravel(), (Xmat, Ymat), method='nearest')
# # #     SMB_dict[d] = regridded_smb_t

# # # ## Perform Delaunay triangulation over catchment region
# # # catchment = MultiPoint(outline.points)
# # # triangles = triangulate(catchment)

# # # ## Sample SMB at each triangle and sum, for each time step
# # # catchment_sum = np.zeros(len(time_indices))
# # # for tri in triangles:
# # #     rep_x, rep_y = tri.representative_point().x, tri.representative_point().y
# # #     area_m2 = tri.area
# # #     smb_x = (np.abs(x_hel - rep_x)).argmin()
# # #     smb_y = (np.abs(y_hel - rep_y)).argmin()
# # #     local_series = [SMB_dict[d][smb_y, smb_x]*area_m2 for d in smb_dates]
# # #     catchment_sum = catchment_sum + np.asarray(local_series)
    
# # # ## Write out results
# # # fn_out = '/Users/lizz/Documents/GitHub/Data_unsynced/Helheim-processed/HIRHAM_integrated_SMB.csv'
# # # with open(fn_out, 'a', newline='') as csvfile:
# # #     fieldnames = ['Date', 'SMB_int']
# # #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
# # #     for d, s in zip(smb_dates, catchment_sum):
# # #         writer.writerow({'Date': d, 'SMB_int': s})
    
