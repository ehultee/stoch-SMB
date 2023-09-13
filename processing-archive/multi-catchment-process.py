#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Read in Mouginot-Rignot catchments

Created on Tue Apr 13 13:41:23 2021
Edit 2 Sep: divide catchments sums by area to get catchment mean

@author: lizz
"""

from shapely.geometry import MultiPoint
from shapely.ops import triangulate
import shapefile
from netCDF4 import Dataset
import numpy as np
import pandas as pd
import pyproj as pyproj
from scipy import interpolate
import datetime
import time

###------------------------
### CHOOSE CATCHMENTS
###------------------------
catchments_to_pull = np.arange(95, 105)

###------------------------
### DATA READ-IN  AND PROJECTION
###------------------------

## Read in BedMachine grid to reproject SMB
gl_bed_path ='/Users/lizz/Documents/GitHub/Data_unsynced/BedMachine-Greenland/BedMachineGreenland-2017-09-20.nc'
fh = Dataset(gl_bed_path, mode='r')
xx = fh.variables['x'][:].copy() #x-coord (polar stereo (70, 45))
yy = fh.variables['y'][:].copy() #y-coord
fh.close()

## Read in Mouginot catchments from shapefile
print('Reading in Mouginot catchments')
catchment_fn = '/Users/lizz/Documents/GitHub/Data_unsynced/Greenland-catchments-Mouginot/Greenland_Basins_PS_v1.4.2.'
sf = shapefile.Reader(catchment_fn) 

## Example SMB field read in for grid
print('Reading in example SMB field')
nhm_smb_path = '/Users/lizz/Documents/GitHub/Data_unsynced/SMBMIP/NHM-SMAP_niwano-monthly-ERA-Interim-1980.nc'
fh2 = Dataset(nhm_smb_path, mode='r')
xlon_nhm = fh2.variables['LON'][:].copy() #x-coord (latlon)
ylat_nhm = fh2.variables['LAT'][:].copy() #y-coord (latlon)
fh2.close()


###------------------------
### SET UP SMB REPROJECTION
###------------------------

## Down-sample bed topo
x_3km = xx[::20] # sample at ~3 km resolution
y_3km = yy[::20]

## Down-sample SMB
x_lon_h = xlon_nhm[::2, ::2] 
y_lat_h = ylat_nhm[::2, ::2] # resolution about 2 km

print('Creating reprojected meshgrid')
wgs84 = pyproj.Proj("+init=EPSG:4326") # LatLon with WGS84 datum used by SMB data
psn_gl = pyproj.Proj("+init=epsg:3413") # Polar Stereographic North used by BedMachine and Mankoff
xs, ys = pyproj.transform(wgs84, psn_gl, x_lon_h, y_lat_h)
Xmat, Ymat = np.meshgrid(x_3km, y_3km) # Downsampled BedMachine coords

###------------------------
### CREATE FRAMEWORK
###------------------------

## Perform Delaunay triangulation over each catchment region
tri_ctmts = {i: [] for i in catchments_to_pull}
for i in catchments_to_pull:
    print('Triangulating catchment {}'.format(sf.record(i)['NAME']))
    c = MultiPoint(sf.shape(i).points)
    tris = triangulate(c)
    tri_ctmts[i] = tris

## Create data frames to store per-model time series
model_names = ['ANICE-ITM_Berends', 'CESM_kampenhout', 'dEBM_krebs','HIRHAM_mottram', 
                'NHM-SMAP_niwano', 'RACMO_noel', 'SNOWMODEL_liston']
# model_names = ['ANICE-ITM_Berends',]
years = range(1980,2013)
start_date = datetime.datetime(years[0],1,1)
end_date = datetime.datetime(years[-1],12,31)
dates = pd.date_range(start=start_date, end=end_date, freq='M')
df_per_ctchmnt = {i: pd.DataFrame(columns=model_names, index=dates) for i in catchments_to_pull}

###------------------------
### CATCHMENT-SUM FOR ALL MODELS
###------------------------

for m in model_names:
    t0 = time.time()
    if m=='CESM_kampenhout':
        vname = 'SMBCORR'
    else:
        vname = 'SMBcorr'
    for y in years:
        ti = time.time()
        fpath = '/Users/lizz/Documents/GitHub/Data_unsynced/SMBMIP/{}-monthly-ERA-Interim-{}.nc'.format(m, y)
        fh = Dataset(fpath, mode='r')
        smb_m = fh.variables[vname][:].copy()
        fh.close()
        d_subset = [d for d in dates if d.year==y]
        for i in range(len(smb_m)): # for each month
            ## downsample and interpolate SMB
            smb_ds = smb_m[i][::2, ::2]
            regridded_smb = interpolate.griddata((xs.ravel(), ys.ravel()), smb_ds.ravel(), (Xmat, Ymat), method='nearest')
            ## Sample SMB at each Delaunay triangle and sum
            for j in catchments_to_pull:
                catchment_sum = 0
                area_sum = 0
                triangles = tri_ctmts[j]
                for tri in triangles:
                    rep_x, rep_y = tri.representative_point().x, tri.representative_point().y
                    area_m2 = tri.area
                    smb_x = (np.abs(x_3km - rep_x)).argmin()
                    smb_y = (np.abs(y_3km - rep_y)).argmin()
                    local_val = regridded_smb[smb_y, smb_x]*area_m2
                    catchment_sum += local_val
                    area_sum += area_m2
                df_per_ctchmnt[j][m][d_subset[i]] = catchment_sum/area_sum
        tf = time.time()
        print('Finished processing year {} in time {}s'.format(y, tf-ti))
    t1 = time.time()
    print('Finished processing model {} in time {}'.format(m, t1-t0))

## Write to CSV
for i in catchments_to_pull:  
    csv_ctchmnt = '/Users/lizz/Documents/GitHub/Data_unsynced/SMBMIP-processed/{}-catchment_{}_mean-tseries.csv'.format(datetime.date.today().strftime('%Y%m%d'), i)
    df_per_ctchmnt[i].to_csv(csv_ctchmnt)
    
