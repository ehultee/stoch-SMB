#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 10:40:49 2021

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
### DATA READ-IN  AND PROJECTION
###------------------------

## Read in BedMachine grid to reproject SMB
gl_bed_path ='/Users/lizz/Documents/GitHub/Data_unsynced/BedMachine-Greenland/BedMachineGreenland-2017-09-20.nc'
fh = Dataset(gl_bed_path, mode='r')
xx = fh.variables['x'][:].copy() #x-coord (polar stereo (70, 45))
yy = fh.variables['y'][:].copy() #y-coord
s_raw = fh.variables['surface'][:].copy() #surface elevation
fh.close()

## Select BedMachine topo in area of Helheim
xl1, xr1 = 5000, 7000
yt1, yb1 = 10000, 14000
x_hel = xx[xl1:xr1:20] # sample at ~3 km resolution
y_hel = yy[yt1:yb1:20]

## Read in Mankoff catchment from shapefile
print('Reading in Mankoff catchment')
catchment_fn = '/Users/lizz/Documents/GitHub/Data_unsynced/Helheim-processed/catchment/helheim_ice_catchment_mankoff'
sf = shapefile.Reader(catchment_fn) 
outline = sf.shapes()[0]

## Example SMB field read in for grid
print('Reading in example SMB field')
nhm_smb_path = '/Users/lizz/Documents/GitHub/Data_unsynced/dEBM-SMB-PI/5km_ISMIP6grid.nc'
fh2 = Dataset(nhm_smb_path, mode='r')
xlon_nhm = fh2.variables['lon'][:].copy() #x-coord (latlon)
ylat_nhm = fh2.variables['lat'][:].copy() #y-coord (latlon)
# ts_nhm = fh2.variables['time'][:].copy()
# smb_nhm = fh2.variables['SMBcorr'][:].copy()
fh2.close()

## Select large area of SMB around Helheim
xl2, xr2 = 950, 1300
yt2, yb2 = 1100, 1350
x_lon_h = xlon_nhm[yt2:yb2:2, xl2:xr2:2] 
y_lat_h = ylat_nhm[yt2:yb2:2, xl2:xr2:2] # resolution about 2 km

print('Creating reprojected meshgrid')
wgs84 = pyproj.Proj("+init=EPSG:4326") # LatLon with WGS84 datum used by SMB data
psn_gl = pyproj.Proj("+init=epsg:3413") # Polar Stereographic North used by BedMachine and Mankoff
xs, ys = pyproj.transform(wgs84, psn_gl, x_lon_h, y_lat_h)
Xmat, Ymat = np.meshgrid(x_hel, y_hel) # BedMachine coords around Helheim

###------------------------
### CREATE FRAMEWORK
###------------------------

## Perform Delaunay triangulation over catchment region
catchment = MultiPoint(outline.points)
triangles = triangulate(catchment)

## Create data frames to store per-model time series
model_names = ['dEBM_preindust',]
years = range(1,101)
start_date = datetime.datetime(years[0],1,1)
end_date = datetime.datetime(years[-1],12,31)
dates = pd.date_range(start=start_date, end=end_date, freq='M')
df_ctchmnt = pd.DataFrame(columns=model_names, index=dates)


###------------------------
### CATCHMENT-SUM FOR ALL MODELS
###------------------------


t0 = time.time()
vname = 'SMB'
for y in years:
    ti = time.time()
    fpath = '/Users/lizz/Documents/GitHub/Data_unsynced/dEBM-SMB-PI/dEBM_AWI_CM2_sidorenko2019_{}.nc'.format(y)
    fh = Dataset(fpath, mode='r')
    smb_m = fh.variables[vname][:].copy()
    fh.close()
    d_subset = [d for d in dates if d.year==y]
    for i in range(len(smb_m)): # for each month
        ## interpolate regional SMB
        smb_local = smb_m[i][yt2:yb2:2, xl2:xr2:2]
        regridded_smb = interpolate.griddata((xs.ravel(), ys.ravel()), smb_local.ravel(), (Xmat, Ymat), method='nearest')
        ## Sample SMB at each Delaunay triangle and sum
        catchment_sum = 0
        for tri in triangles:
            rep_x, rep_y = tri.representative_point().x, tri.representative_point().y
            area_m2 = tri.area
            smb_x = (np.abs(x_hel - rep_x)).argmin()
            smb_y = (np.abs(y_hel - rep_y)).argmin()
            local_val = regridded_smb[smb_y, smb_x]*area_m2
            catchment_sum += local_val
        df_ctchmnt[m][d_subset[i]] = catchment_sum
    tf = time.time()
    print('Finished processing year {} in time {}s'.format(y, tf-ti))
t1 = time.time()
print('Finished processing all in time {}'.format(t1-t0))

csv_ctchmnt = '/Users/lizz/Documents/GitHub/Data_unsynced/SMBMIP-processed/{}-helheim_catchment-preindust_tseries.csv'.format(datetime.date.today().strftime('%Y%m%d'))
df_ctchmnt.to_csv(csv_ctchmnt)
    
