#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monthly SMB anomaly map

Created on Thu Sep  2 12:26:37 2021

Edited Mon 14 Nov 2022: read in data from external hard drive; make two-panel plot
@author: lizz
"""

from shapely.geometry import MultiPoint, Polygon, Point
from shapely.ops import triangulate
from scipy.spatial import distance
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
#nhm_smb_path = '/Users/lizz/Documents/GitHub/Data_unsynced/SMBMIP/NHM-SMAP_niwano-monthly-ERA-Interim-1980.nc'
nhm_smb_path = '/Volumes/Backup Plus/Large data moved 20220331/SMBMIP/NHM-SMAP_niwano-monthly-ERA-Interim-1980.nc'
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
x_lon_h = xlon_nhm[::10, ::10] 
y_lat_h = ylat_nhm[::10, ::10] # resolution about 10 km

print('Creating reprojected meshgrid')
wgs84 = pyproj.Proj("+init=EPSG:4326") # LatLon with WGS84 datum used by SMB data
psn_gl = pyproj.Proj("+init=epsg:3413") # Polar Stereographic North used by BedMachine and Mankoff
xs, ys = pyproj.transform(wgs84, psn_gl, x_lon_h, y_lat_h)
Xmat, Ymat = np.meshgrid(x_3km, y_3km) # Downsampled BedMachine coords

##------------------------
## CREATE FRAMEWORK
##------------------------

## Identify grid points within catchment
pts_all = [(xs.ravel()[k], ys.ravel()[k]) for k in range(len(xs.ravel()))]
pt_ctmts = {i: [] for i in catchments_to_pull}
for i in catchments_to_pull:
    print('Point-checking catchment {}'.format(sf.record(i)['NAME']))
    c = Polygon(sf.shape(i).points)
    pts_in = [Point(p).within(c) for p in pts_all]
    pts = np.asarray(pts_all)[pts_in]
    pt_ctmts[i] = pts

## Create data frames to store per-model data
# model_names = ['ANICE-ITM_Berends', 'CESM_kampenhout', 'dEBM_krebs','HIRHAM_mottram', 
#                 'NHM-SMAP_niwano', 'RACMO_noel', 'SNOWMODEL_liston']
model_names = ['ANICE-ITM_Berends',]
years = range(1980,1985)
start_date = datetime.datetime(years[0],1,1)
end_date = datetime.datetime(years[-1],12,31)
dates = pd.date_range(start=start_date, end=end_date, freq='M')
example_df_per_ctchmnt = {i: 
                  {m: 
                   {n: 
                    {y: pd.DataFrame(columns=('elevation', 
                                                'point_smb'))  for y in years}
                    for n in range(12)}
                     for m in model_names}
                      for i in catchments_to_pull}

###------------------------
### EXTRACT EXAMPLE SMB FIELD FOR CATCHMENT
###------------------------

## Store regridded SMBs for use in each catchment to take all available data
smb_ex_monthly = {i: [] for i in range(12)}

for m in ('ANICE-ITM_Berends',):
    t0 = time.time()
    if m=='CESM_kampenhout':
        vname = 'SMBCORR'
    else:
        vname = 'SMBcorr'
    for y in years:
        ti = time.time()
        #fpath = '/Users/lizz/Documents/GitHub/Data_unsynced/SMBMIP/{}-monthly-ERA-Interim-{}.nc'.format(m, y)
        fpath = '/Volumes/Backup Plus/Large data moved 20220331/SMBMIP/{}-monthly-ERA-Interim-{}.nc'.format(m, y)
        fh = Dataset(fpath, mode='r')
        smb_m = fh.variables[vname][:].copy()
        fh.close()
        d_subset = [d for d in dates if d.year==y]
        for i in range(len(smb_m)): # for each month
            ## downsample SMB
            smb_ds = smb_m[i][::10, ::10]
            ## Sample SMB at each Delaunay triangle and sum
            for j in catchments_to_pull:
                points = pt_ctmts[j]
                elevations = []
                smb_point_vals = smb_ds.ravel()[pts_in] ## go back and revise this to use new pts_in for every catchment
                for p in points:
                    surf_x = (np.abs(x_3km - p[0])).argmin()
                    surf_y = (np.abs(y_3km - p[1])).argmin()
                    elevations.append(s_3km[surf_y, surf_x])
                example_df_per_ctchmnt[j][m][i][y] = example_df_per_ctchmnt[j][m][i][y].assign(elevation=elevations,
                                            point_smb=smb_point_vals)
                
        tf = time.time()
        print('Finished processing year {} in time {}s'.format(y, tf-ti))
    t1 = time.time()
    print('Finished processing model {} in time {}'.format(m, t1-t0))


## Select points for tseries comparison
series_points = [
    (497.73, -2298.93), # near terminus of Kangerlussuaq, in km Easting, Northing
    (397.96, -2342.95), # small positive anomaly, southwest
    (483.06, -2161.00) # farther up in accumulation area
    ]
ser = {'Point 1':[], 'Point 2':[], 'Point 3':[]}
for i,p in enumerate(series_points):
    idx = distance.cdist([1000*np.array(p)], pt_ctmts[101]).argmin()
    ts = []
    for y in years:
        for m in range(12):
            d = example_df_per_ctchmnt[101]['ANICE-ITM_Berends'][m][y]['point_smb'][idx]
            ts.append(d)
    ser['Point {}'.format(i+1)] = ts
df = pd.DataFrame(ser)


# ## Make map plot
# rep_pts = np.array([(tri.representative_point().x, tri.representative_point().y) 
#                     for tri in triangles])
# fig, ax = plt.subplots(1)
# ax.scatter(rep_pts[:,0], rep_pts[:,1], c=smb_point_vals)
# ax.set(aspect=1)
# plt.show()

# ## Make map plot
# outlines = np.array([(tri.exterior.xy) 
#                     for tri in triangles])
# colors = cm.get_cmap('viridis')(smb_point_vals)
# fig, ax = plt.subplots()
# ax.plot(outlines[:,0], outlines[:,1], color='k')
# for k in range(len(triangles)):
#     ax.fill(outlines[k,0], outlines[k,1], c=colors[k])
# ax.set(aspect=1)
# plt.show()

## Map plot of gridded SMB
ex_month = 1 #calendar month to plot
fig, ax = plt.subplots()
example_smb = example_df_per_ctchmnt[101]['ANICE-ITM_Berends'][ex_month-1][1980]['point_smb']
sc = ax.scatter(0.001*np.array(pt_ctmts[101][:,0]), 0.001*np.array(pt_ctmts[101][:,1]), 
           c=example_smb - np.mean(example_smb),
           vmin=-500, vmax=500,
           cmap='RdBu')
# ax.set(aspect=1)
cb = fig.colorbar(sc, ax=ax)
cb.ax.set_title('SMB anomaly')
ax.set(ylabel='Northing [km]', xlabel='Easting', title='Month {}, ANICE-ITM'.format(ex_month))


fig, axs = plt.subplots(3,4, sharex=True, sharey=True)
for i in range(12):
    ax=axs.ravel()[i]
    example_smb = example_df_per_ctchmnt[101]['ANICE-ITM_Berends'][i][1980]['point_smb']
    sc = ax.scatter(0.001*pt_ctmts[101][:,0], 0.001*pt_ctmts[101][:,1], 
               c=example_smb - np.mean(example_smb),
               vmin=-500, vmax=500,
               cmap='RdBu')
    ax.set(aspect=1)
    if (i+1)%4==0:
        cb = fig.colorbar(sc, ax=ax)
        # cb.ax.set_title('SMB anomaly')
    else:
        continue  
plt.show()

