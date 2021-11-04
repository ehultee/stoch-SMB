#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Selecting Mouginot catchments that are part of the contiguous GrIS.
+ mapping catchment limits
Based on code from Lizz Ultee: catchment-plot.py

@author: vincent
"""

import os
import sys
import shapefile
from netCDF4 import Dataset
import numpy as np
# import pandas as pd
# import pyproj as pyproj
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import pyproj

MapSurfaceElev = True #choose between surface elevation and bed topography map
MapBedTopo     = False #choose between surface elevation and bed topography map


### Read BedMachine ###
pathBM = '/media/vincent/TOSH4TB/GeorgiaTech/DataSearch/BedMachine/BedMachineGreenland-2021-04-20.nc'
ds = Dataset(pathBM,mode='r') #open dataset in reading mode
xx = ds.variables['x'][:].copy() #x-coord (polar stereo (70, 45))
yy = ds.variables['y'][:].copy() #y-coord
if MapSurfaceElev:
    surfBM = ds.variables['surface'][:].copy() #surface elevation
    maskBM = ds.variables['mask'][:].copy() #BM mask values: 0=ocean, 1=ice-free land, 2=grounded ice, 3=floating ice, 4=non-Greenland land
    surfproc = np.ma.masked_where(maskBM!=2,surfBM) #Surface elev for grounded ice
if MapBedTopo:
    bedBM  = ds.variables['bed'][:].copy() #bed topo
ds.close()
## Down-sampling BedMachine (original resolution: 150m) ##
x1 = xx[::20]
y1 = yy[::20]
if MapSurfaceElev:
    surf1 = surfproc[::20,::20]
    del(surfBM,surfproc,maskBM)
if MapBedTopo:
    bed1  = bedBM[::20,::20]
    del(bedBM)
del(xx,yy)

### Read Mouginot catchments from shapefile ###
pathMoug = '/media/vincent/TOSH4TB/GeorgiaTech/DataSearch/CatchmentsMouginot/Greenland_Basins_PS_v1.4.2.shp'
sf       = shapefile.Reader(pathMoug) 

### Mapping ###
fig1 = plt.figure(figsize=[8,9])
ax = fig1.subplots(1)
ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=-45.0))
ax.set(xlim=(min(x1)-10, max(x1)+10), ylim=(min(y1)-10, max(y1)+10))
if MapSurfaceElev: #Mapping the surface elevation
    surflevels = np.linspace(np.min(surf1)-1,np.max(surf1)+1,101)
    ax.contourf(x1,y1,surf1,cmap='jet',levels=surflevels)
elif MapBedTopo: #Mapping the bed topography
    bedlevels  = np.linspace(-3500,3500,101)
    ax.contourf(x1,y1,bed1,cmap='jet',levels=bedlevels)

catchID     = 0 #keeping track of catchment ID number
allAreas0     = [] #areas of the catchments
allCentroids0 = [] #centroids of the catchments
allxvertices0 = [] #all x-coordinates of the vertices
allyvertices0 = [] #all y-coordinates of the vertices
allIDs0       = [] #all catchment IDs
for shape in sf.shapeRecords():
    if 'ICE_CAPS' not in shape.record['NAME']: #excluding Ice Caps catchments
        if len(shape.shape.parts)>1:
            # Disjointed catchments have shape.shape.parts>1
            catchment_color='grey'
        else:
            # Unified catchments have shape.shape.parts==1
            catchment_color='k'
        for i in range(len(shape.shape.parts)): ## plot disjointed parts separately
                i_start = shape.shape.parts[i] #index starting this part of the catchment
                if i==len(shape.shape.parts)-1: #at last part of the catchment
                    i_end = len(shape.shape.points) #last point to plot is the overall last of the catchment
                else:
                    i_end = shape.shape.parts[i+1] #otherwise,plot until index starting next part of the catchment
                x = [i[0] for i in shape.shape.points[i_start:i_end]] #x is first element of the sublist
                y = [i[1] for i in shape.shape.points[i_start:i_end]] #y is second element of the sublist
                allxvertices0.append(x)
                allyvertices0.append(y)
                allAreas0.append(Polygon(shape.shape.points[i_start:i_end]).area)
                allCentroids0.append(list(Polygon(shape.shape.points[i_start:i_end]).centroid.coords)[0])
                allIDs0.append(catchID)
                #ax.plot(x,y, color=catchment_color)
    catchID += 1
    
# Conversion to numpy arrays #
allAreas0     = np.array(allAreas0)
allCentroids0 = np.array(allCentroids0) #centroids of the catchments
allxvertices0 = np.array(allxvertices0) #all x-coordinates of the vertices
allyvertices0 = np.array(allyvertices0) #all y-coordinates of the vertices
allIDs0       = np.array(allIDs0) #all catchment IDs

# Find largest part of the catchments with multiple parts #
multiplecatch = np.unique(np.array(allIDs0)[np.where(np.diff(allIDs0)==0)[0]])
todel = np.array([]) #indices of smaller parts that we want to remove
for idnb in multiplecatch:
    inds = np.where(np.array(allIDs0)==idnb)[0] #indices having the same catchment ID
    maxarea = 0 #largest area among the parts of the catchment
    for ii in inds:
        if allAreas0[ii]>maxarea:
            maxarea = allAreas0[ii]
    #Add the smaller parts to the indices to delete
    todel = np.append(todel,[index for index in inds if allAreas0[index]<maxarea]).astype(int)
 
allAreas      = np.delete(allAreas0,todel) #areas of the catchments
allCentroids  = np.delete(allCentroids0,todel,axis=0) #centroids of the catchments
allxvertices  = np.delete(allxvertices0,todel,axis=0) #all x-coordinates of the vertices
allyvertices  = np.delete(allyvertices0,todel,axis=0) #all y-coordinates of the vertices
allIDs        = np.delete(allIDs0,todel) #all catchment IDs

for ii,ctr in enumerate(allCentroids0):
    if ii not in todel:
        ax.plot(allxvertices0[ii],allyvertices0[ii],'k',label='Contiguous GrIS')
        #ax.text(ctr[0],ctr[1],str(allIDs[ii]))
    else:
        ax.plot(allxvertices0[ii],allyvertices0[ii],'r',label='Excluded')
        #ax.text(ctr[0],ctr[1],str(allIDs[ii]),color='r')

#plt.legend(fontsize=11,loc='lower right')
plt.show()
fig1.tight_layout()

### Projecting to lat-lon ###
#Projection info is from Lizz Ultee (could not find info about projection for Mouginot catchments)
wgs84  = pyproj.Proj("+init=EPSG:4326") # LatLon with WGS84 data (equiv. to EPSG4326)
psn_M  = pyproj.Proj("+init=epsg:3413") # Polar Stereographic North
allCentroids0_lonlat = []
for ii in range(len(allCentroids0)):
    allCentroids0_lonlat.append(pyproj.transform(psn_M,wgs84,allCentroids0[ii][0],allCentroids0[ii][1]))
allCentroids_lonlat = []
for ii in range(len(allCentroids)):
    allCentroids_lonlat.append(pyproj.transform(psn_M,wgs84,allCentroids[ii][0],allCentroids[ii][1]))








