#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot catchments over map of Greenland
Highlight a given catchment
[Plot its time series from different models - see catchment-plot-tseries_ex]

Created on Mon Apr 26 12:11:39 2021

@author: lizz
"""

import shapefile
from netCDF4 import Dataset
import numpy as np
# import pandas as pd
# import pyproj as pyproj
import cartopy.crs as ccrs
from cartopy.io.img_tiles import Stamen
import matplotlib.pyplot as plt


## Read in BedMachine surface to evaluate
gl_bed_path ='/Users/lizz/Documents/GitHub/Data_unsynced/BedMachine-Greenland/BedMachineGreenland-2017-09-20.nc'
fh = Dataset(gl_bed_path, mode='r')
xx = fh.variables['x'][:].copy() #x-coord (polar stereo (70, 45))
yy = fh.variables['y'][:].copy() #y-coord
s_raw = fh.variables['surface'][:].copy() #surface elevation
thick_mask = fh.variables['mask'][:].copy()
ss = np.ma.masked_where(thick_mask !=2, s_raw)#mask values: 0=ocean, 1=ice-free land, 2=grounded ice, 3=floating ice, 4=non-Greenland land
fh.close()

X = xx[::4]
Y = yy[::4]
S = ss[::4, ::4]

## Read in Mouginot catchments from shapefile
print('Reading in Mouginot catchments')
catchment_fn = '/Users/lizz/Documents/GitHub/Data_unsynced/Greenland-catchments-Mouginot/Greenland_Basins_PS_v1.4.2.'
sf = shapefile.Reader(catchment_fn) 

highlight_catchment_name = 'KANGERLUSSUAQ'

## plot all together, including disjoint multi-catchments
fig, ax = plt.subplots(1)
ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=-45.0))
ax.set(xlim=(min(X)-100, max(X)+100), ylim=(min(Y)-100, max(Y)+100))
ax.stock_img()
for shape in sf.shapeRecords():
    if len(shape.shape.parts)>1:
        catchment_color='grey'
    else:
        catchment_color='k'
    for i in range(len(shape.shape.parts)): ## plot disjointed parts separately
        i_start = shape.shape.parts[i]
        if i==len(shape.shape.parts)-1:
            i_end = len(shape.shape.points)
        else:
            i_end = shape.shape.parts[i+1]
        x = [i[0] for i in shape.shape.points[i_start:i_end]]
        y = [i[1] for i in shape.shape.points[i_start:i_end]]
        ax.plot(x,y, color=catchment_color)
    if 'KANGERLUSSUAQ' in shape.record['NAME']:
        ax.fill(x,y, color='b')
# ax.contour(X, Y, S)
plt.show()

# ## plot only simply connected catchments
# fig, ax = plt.subplots(1)
# ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=-45.0))
# ax.set(xlim=(min(X)-100, max(X)+100), ylim=(min(Y)-100, max(Y)+100))
# ax.stock_img()
# for shape in sf.shapeRecords():
#     if len(shape.shape.parts)>1:
#         continue
#     else:
#         catchment_color='k'
#         x = [i[0] for i in shape.shape.points[:]]
#         y = [i[1] for i in shape.shape.points[:]]
#         plt.plot(x,y, color=catchment_color)
# # ax.contour(X, Y, S)
# plt.show()