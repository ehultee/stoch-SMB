# -*- coding: utf-8 -*-
"""
Plot an example power spectral density of Greenland surface mass balance

Created on 16 Mar 2021

@author: lizz
"""

from netCDF4 import Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyproj as pyproj
from scipy import signal


# print 'Reading in surface mass balance from 1981-2010 climatology'
# gl_smb_path ='/Users/lizz/Documents/GitHub/Data_unsynced/HIRHAM5-SMB/DMI-HIRHAM5_GL2_ERAI_1980_2014_SMB_YM.nc'
# fh2 = Dataset(gl_smb_path, mode='r')
# x_lon = fh2.variables['lon'][:].copy() #x-coord (latlon)
# y_lat = fh2.variables['lat'][:].copy() #y-coord (latlon)
# #zs = fh2.variables['height'][:].copy() #height in m - is this surface elevation or SMB?
# ts = fh2.variables['time'][:].copy()
# smb_raw = fh2.variables['smb'][:].copy()
# fh2.close()

helheim_smb_fpath = '/Users/lizz/Documents/GitHub/Data_unsynced/Helheim-processed/smb_rec._.BN_RACMO2.3p2_ERA5_3h_FGRN055.1km.MM.csv'

helheim_smb = pd.read_csv(helheim_smb_fpath, parse_dates=[0])

f, P = signal.periodogram(helheim_smb[' smb_rec (m3WE)'], scaling='spectrum')
# plt.semilogy(f, P)
plt.loglog(f, P)
# plt.plot(f, P)
plt.xlabel('Frequency [Hz]')
plt.ylabel('PSD [m6WE/Hz]')
plt.show()
