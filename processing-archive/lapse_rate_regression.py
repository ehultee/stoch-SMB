#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Load process-model SMB in a given catchment and regress a mass balance gradient

Created on Thu Aug  5 14:14:05 2021
Edit Sept 2: compute 30-year mean of anomaly

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
import matplotlib.pyplot as plt
from matplotlib import cm
from stochSMB import segments_fit

###------------------------
### CHOOSE CATCHMENTS
###------------------------
catchments_to_pull = (101,)

###------------------------
### DATA READ-IN  AND PROJECTION
###------------------------

## Read in BedMachine grid to reproject SMB
gl_bed_path ='/Users/lizz/Documents/GitHub/Data_unsynced/BedMachine-Greenland/BedMachineGreenland-2017-09-20.nc'
fh = Dataset(gl_bed_path, mode='r')
xx = fh.variables['x'][:].copy() # x-coord (polar stereo (70, 45))
yy = fh.variables['y'][:].copy() # y-coord
ss = fh.variables['surface'][:].copy() # surface elevation
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
s_3km = ss[::20,::20]

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

## Create data frames to store per-model data
# model_names = ['ANICE-ITM_Berends', 'CESM_kampenhout', 'dEBM_krebs','HIRHAM_mottram', 
#                 'NHM-SMAP_niwano', 'RACMO_noel', 'SNOWMODEL_liston']
model_names = ['ANICE-ITM_Berends',]
years = range(1980,2010)
start_date = datetime.datetime(years[0],1,1)
end_date = datetime.datetime(years[-1],12,31)
dates = pd.date_range(start=start_date, end=end_date, freq='M')
df_per_ctchmnt = {i: {m: {n: {y: pd.DataFrame(columns=('elevation', 
                                               'point_smb'))  for y in years} for n in range(12)} for m in model_names} for i in catchments_to_pull}

###------------------------
### CATCHMENT-SUM FOR ALL MODELS
###------------------------

## Store regridded SMBs for use in each catchment to take all available data

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
                triangles = tri_ctmts[j]
                elevations = []
                smb_point_vals = []
                for tri in triangles:
                    rep_x, rep_y = tri.representative_point().x, tri.representative_point().y
                    smb_x = (np.abs(x_3km - rep_x)).argmin()
                    smb_y = (np.abs(y_3km - rep_y)).argmin()
                    elevations.append(s_3km[smb_y, smb_x])
                    smb_point_vals.append(regridded_smb[smb_y, smb_x])
                df_per_ctchmnt[j][m][i][y] = df_per_ctchmnt[j][m][i][y].assign(elevation=elevations,
                                            point_smb=smb_point_vals)
                
        tf = time.time()
        print('Finished processing year {} in time {}s'.format(y, tf-ti))
    t1 = time.time()
    print('Finished processing model {} in time {}'.format(m, t1-t0))
    
###------------------------
### PLOT ALL DATES TOGETHER
###------------------------

df_c = df_per_ctchmnt[101]['ANICE-ITM_Berends']
dict_per_month = {i: [] for i in range(12)}
for i in range(12):
    d_to_join = [df_c[i][y] for y in years]
    dict_per_month[i] = pd.concat(d_to_join)

c = cm.get_cmap('tab20b')

plt.figure()
for i in range(12):
    df = dict_per_month[i]
    # f = interpolate.splrep(df['elevation'], df['point_smb'])
    # plt.scatter(df['elevation'], df['point_smb'], color=c(i), label='Month '+d_subset[i].strftime('%m'))
    plt.scatter(df['elevation'], df['point_smb'], color=c(i%12), label=dates[i].strftime('%m/%Y'))
# plt.legend(loc='best')
plt.show()


# ##
# ## Make a piecewise linear fit and plot it
# # def piecewise_linear(x, x0, y0, k1, k2):
# #     return np.piecewise(x, [x < x0], [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0])

monthly_mbg = {i: [] for i in range(12)}

fig, axs = plt.subplots(3, 4, sharex=True, sharey=True)
# plt.title('Auto-fitting segments based on AIC & BIC')
for i in range(12):
    ax = axs.ravel()[i]
    # df = df_per_ctchmnt[101]['ANICE-ITM_Berends'][dates[i]]
    df = dict_per_month[i]
    # axs.ravel()[i].scatter(df['elevation'], df['point_smb'], color=c(i%12), label=dates[i].strftime('%m/%Y'))
    pt_smbs = np.asarray(df.sort_values(by='elevation')['point_smb'])
    anomalies = pt_smbs - np.mean(pt_smbs)
    ax.scatter(df.sort_values(by='elevation')['elevation'], anomalies, color=c(i%12), label=dates[i].strftime('%m/%Y'))
    ## Piecewise implementation - works but misjudges line segments
    # p, e = optimize.curve_fit(piecewise_linear, 
    #                            np.asarray(df.sort_values(by='elevation')['elevation']), 
    #                            np.asarray(df.sort_values(by='elevation')['point_smb']))
    # xnew = np.linspace(0, max(df['elevation']))
    # plt.plot(xnew, piecewise_linear(xnew, *p))
    px, py = segments_fit(np.asarray(df.sort_values(by='elevation')['elevation']),
                          # np.asarray(df.sort_values(by='elevation')['point_smb']),
                          anomalies,
                          maxcount=3)
    monthly_mbg[i] = (px,py)
    ax.plot(px, py)
    ax.text(1000,-1100,'Month {}'.format(i+1))
# plt.legend(loc='best')
plt.show()

# ## scaled to fraction of mean
# fig, axs = plt.subplots(3, 4, sharex=True, sharey=True)
# # plt.title('Auto-fitting segments based on AIC & BIC')
# for i in range(12):
#     df = df_per_ctchmnt[101]['ANICE-ITM_Berends'][dates[i]]
#     # axs.ravel()[i].scatter(df['elevation'], df['point_smb'], color=c(i%12), label=dates[i].strftime('%m/%Y'))
#     pt_smbs = np.asarray(df.sort_values(by='elevation')['point_smb'])
#     frac_anomalies = (pt_smbs - np.mean(pt_smbs))/np.mean(pt_smbs)
#     axs.ravel()[i].scatter(df.sort_values(by='elevation')['elevation'], frac_anomalies, color=c(i%12), label=dates[i].strftime('%m/%Y'))
#     ## Piecewise implementation - works but misjudges line segments
#     # p, e = optimize.curve_fit(piecewise_linear, 
#     #                            np.asarray(df.sort_values(by='elevation')['elevation']), 
#     #                            np.asarray(df.sort_values(by='elevation')['point_smb']))
#     # xnew = np.linspace(0, max(df['elevation']))
#     # plt.plot(xnew, piecewise_linear(xnew, *p))
#     px, py = segments_fit(np.asarray(df.sort_values(by='elevation')['elevation']),
#                           # np.asarray(df.sort_values(by='elevation')['point_smb']),
#                           frac_anomalies,
#                           maxcount=3)
#     axs.ravel()[i].plot(px, py)
# # plt.legend(loc='best')
# plt.show()

