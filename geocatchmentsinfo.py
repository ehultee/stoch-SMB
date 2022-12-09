#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate geographical information files for the catchment of Mouginot et al. (2019)
Source of catchments is doi:10.7280/D1WT11
Elevation data is from BedMachine Version 4

@author: vincent
"""

import sys
import os
import csv
import time
import numpy as np
import pandas as pd
import shapefile
import matplotlib.pyplot as plt
from shapely.geometry import Polygon,Point
import netCDF4 as nc
import pyproj
import cartopy.crs as ccrs


# Select: save general data about the catchments and/or outlines of the specific catchments #
saveGeneralData       = True
saveCatchmentOutlines = True

### Paths to datasets ###
# BedMachine #
pathBM = '/media/vincent/TOSH4TB/GeorgiaTech/DataSearch/BedMachine/BedMachineGreenland-2021-04-20.nc'
# Catchment shapefile #
pathMoug = '/media/vincent/TOSH4TB/GeorgiaTech/DataSearch/CatchmentsMouginot/Greenland_Basins_PS_v1.4.2.shp'


### First: find catchments of interest (exclude isolated ice) and derive their properties ###
sf       = shapefile.Reader(pathMoug) 
catchID     = 0 #keeping track of catchment ID number
allAreas0     = [] #areas of the catchments
allCentroids0 = [] #centroids of the catchments
allxvertices0 = [] #all x-coordinates of the vertices
allyvertices0 = [] #all y-coordinates of the vertices
allIDs0       = [] #all catchment IDs
for shape in sf.shapeRecords():
    if 'ICE_CAPS' not in shape.record['NAME']: #excluding Ice Caps catchments
        for i in range(len(shape.shape.parts)): #process disjointed parts separately
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
    catchID += 1
# Conversion to numpy arrays #
allAreas0     = np.array(allAreas0)
allCentroids0 = np.array(allCentroids0) #centroids of the catchments
allxvertices0 = np.array(allxvertices0,dtype=object) #all x-coordinates of the vertices
allyvertices0 = np.array(allyvertices0,dtype=object) #all y-coordinates of the vertices
allIDs0       = np.array(allIDs0) #all catchment IDs
# Find largest part of the catchments with multiple parts (smaller parts are isolated) #
multiplecatch = np.unique(np.array(allIDs0)[np.where(np.diff(allIDs0)==0)[0]])
allTotAreas0  = np.copy(allAreas0) #prepare array of total catchment areas (including all separate parts)
todel = np.array([]) #indices of smaller parts that we want to remove
for idnb in multiplecatch:
    inds = np.where(np.array(allIDs0)==idnb)[0] #indices having the same catchment ID
    allTotAreas0[inds] = np.sum(allAreas0[inds]) #compute total catchment area
    maxarea = 0 #largest area among the parts of the catchment
    for ii in inds:
        if allAreas0[ii]>maxarea:
            maxarea = allAreas0[ii]
    #Add the smaller parts to the indices to delete
    todel = np.append(todel,[index for index in inds if allAreas0[index]<maxarea]).astype(int)
allTotAreas   = np.delete(allTotAreas0,todel) #areas of the catchments
allCentroids  = np.delete(allCentroids0,todel,axis=0) #centroids of the catchments
allxvertices  = np.delete(allxvertices0,todel,axis=0) #all x-coordinates of the vertices
allyvertices  = np.delete(allyvertices0,todel,axis=0) #all y-coordinates of the vertices
allIDs        = np.delete(allIDs0,todel) #all catchment IDs
ctot          = len(allIDs) #total number of catchments
# Put geographical variables in Lat-Lon #
transformer = pyproj.Transformer.from_crs('epsg:3413','epsg:4326') #transformer of x,y (Polar Stereo 70,-45) to lat-lon
allCentroids_latlon = []
allVertices_latlon  = []
for ii in range(len(allCentroids)):
    allCentroids_latlon.append(transformer.transform(allCentroids0[ii][0],allCentroids0[ii][1]))
    allVertices_latlon.append([transformer.transform(allxvertices[ii][jj],allyvertices[ii][jj]) for jj in range(len(allxvertices[ii]))])
allCentroids_latlon = np.round(allCentroids_latlon,5) #conversion to array and rounding
allVertices_latlon  = np.array(allVertices_latlon,dtype=object) #conversion to array
for ii in range(len(allVertices_latlon)):
    allVertices_latlon[ii] = np.round(allVertices_latlon[ii],5) #and rounding

# Read BedMachine #
ds = nc.Dataset(pathBM,mode='r') #open dataset in reading mode
xx = ds.variables['x'][:].copy() #x-coord (polar stereo (70, 45))
yy = ds.variables['y'][:].copy() #y-coord (polar stereo (70, 45))
surfBM0 = ds.variables['surface'][:].copy() #surface elevation
maskBM  = ds.variables['mask'][:].copy() #BedMachine mask
surfBM1 = np.ma.masked_where(maskBM!=2,surfBM0)#mask values: 0=ocean, 1=ice-free land, 2=grounded ice, 3=floating ice, 4=non-Greenland land
ds.close()
# Down-sampling BedMachine at 3km resolution
x1 = xx[::20]
y1 = yy[::20]
surf1 = surfBM1[::20,::20]
# List of BM indices not classified in any catchment yet #
lSearch = []
for kk in range(len(x1)):
    for ll in range(len(y1)):
        if type(surf1[ll,kk])==np.float32:
            lSearch.append([kk,ll])
# pNotFound = np.reshape([[[kk,ll] for kk in range(len(x1))] for ll in range(len(y1))],(-1,2)).tolist()

del(surfBM0,surfBM1,maskBM,xx,yy)
allMeanElevs = np.zeros(ctot) #mean elevation for all catchments
allindsBM    = [] #BedMachine indices included in each catchment
for ii in range(ctot):
    print(f'Processing catchment {ii}')
    # Create polygon object from catchment vertices
    verts = [(allxvertices[ii][jj],allyvertices[ii][jj]) for jj in range(len(allxvertices[ii]))]
    polygon = Polygon(verts)
    indsBM  = [] #indices of BedMachine grid included in the polygon
    elevcalc = 0 #calculation of mean elevation
    for [ix,iy] in lSearch:
        if polygon.contains(Point([x1[ix],y1[iy]])):
            indsBM.append([ix,iy]) #add to list of BedMachine indices of the catchment
            elevcalc += surf1[iy,ix] #add to calculation of mean elevation
    indsBM = np.array(indsBM) #convert to numpy array
    allindsBM.append(indsBM)
    if len(indsBM)>0:
        allMeanElevs[ii] = elevcalc/len(indsBM) #average elevation
    

### Below: calculation with centroids ###
# allCentrElevs = np.zeros(ctot)
# allCentrBMinds = np.zeros((ctot,2))
# coordsBM0 = np.meshgrid(x1,y1)
# for ii in range(ctot):
#     crd  = allCentroids[ii]
#     absdist = np.sqrt((coordsBM0[0]-crd[0])**2)+np.sqrt((coordsBM0[1]-crd[1])**2)
#     indBM   = np.unravel_index(absdist.argmin(),absdist.shape)
#     allCentrBMinds[ii] = [indBM[1],indBM[0]]
#     allCentrElevs[ii] = surf1[indBM]


### Save variables in csv output ###
if saveGeneralData:
    if os.path.isdir('CatchmentsMouginotInfo/')==False: #make sure directory exists
        os.mkdir('CatchmentsMouginotInfo/')
    # First: save the general data about the catchments
    with open('CatchmentsMouginotInfo/generalCatchmentData.csv','w',newline='') as outputcsv:
        header = ['ID','Mean Elevation [m]','Centroid lat [EPSG:4326]','Centroid lon [EPSG:3413]','Centroid x [EPSG:3413]','Centroid y [EPSG:3413]']
        writer = csv.writer(outputcsv,header)
        writer.writerow(header)
        for ii in range(ctot):
            writer.writerow([allIDs[ii],allMeanElevs[ii],allCentroids_latlon[ii,0],allCentroids_latlon[ii,1],allCentroids[ii,0],allCentroids[ii,1]])
 
if saveCatchmentOutlines: 
    if os.path.isdir('CatchmentsMouginotInfo/')==False: #make sure directory exists
        os.mkdir('CatchmentsMouginotInfo/')
    # Second: save the catchment individual outlines
    for jj,idnb in enumerate(allIDs):
        pathname = f'CatchmentsMouginotInfo/outline_catchment_{idnb}.csv'
        with open(pathname,'w',newline='') as outputcsv:
            header = ['Vertex lat [EPSG:4326]','Vertex lon [EPSG:4326]']
            writer = csv.writer(outputcsv,header)
            writer.writerow(header)
            for ii in range(np.shape(allVertices_latlon[jj])[0]):
                writer.writerow([allVertices_latlon[jj][ii,0],allVertices_latlon[jj][ii,1]])












