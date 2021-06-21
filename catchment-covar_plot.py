#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 16:40:17 2021

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
from matplotlib import cm, colors, colorbar
from mpl_toolkits.axes_grid1 import make_axes_locatable


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

## Read in inter-catchment covariance from csv
covar_fpath = '/Users/lizz/Desktop/20210618-GL_sparse_corrcoef.csv'
# covar_fpath = '/Users/lizz/Desktop/ANICE_empC-20210611.csv'
catchment_corrs = np.loadtxt(covar_fpath)


## Read in Mouginot catchments from shapefile
print('Reading in Mouginot catchments')
catchment_fn = '/Users/lizz/Documents/GitHub/Data_unsynced/Greenland-catchments-Mouginot/Greenland_Basins_PS_v1.4.2.'
sf = shapefile.Reader(catchment_fn) 

# highlight_catchment_name, highlight_catchment_id = 'UMIAMMAKKU', 0
# highlight_catchment_name, highlight_catchment_id = 'KANGIATA_NUNAATA_SERMIA', 15
highlight_catchment_name, highlight_catchment_id = 'KANGERLUSSUAQ', 101

highlight_corrs = catchment_corrs[:,highlight_catchment_id]

## plot all together, including disjoint multi-catchments
corr_cmap = cm.get_cmap('Reds')
corr_colors = corr_cmap(highlight_corrs)

fig, ax = plt.subplots(1)
ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=-45.0))
ax.set(xlim=(min(X)-100, max(X)+100), ylim=(min(Y)-100, max(Y)+100))
ax.stock_img()
for i,shape in enumerate(sf.shapeRecords()[0:199]):
    corr_with_highlight_ctmt = highlight_corrs[i]
    corr_color = corr_cmap(corr_with_highlight_ctmt)
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
        ax.fill(x,y, color=corr_color)
ax.set(title='Corr with catchment {}'.format(highlight_catchment_name))

divider = make_axes_locatable(ax)
cax = divider.append_axes("bottom", size="5%", pad=0.05, axes_class=plt.Axes)
norm = colors.Normalize(vmin=min(highlight_corrs), vmax=1)

plt.colorbar(cm.ScalarMappable(norm=norm, cmap=corr_cmap),
             cax=cax, orientation='horizontal', label='Correlation')
plt.show()
