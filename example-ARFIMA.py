#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fractional ARIMA using Kuttruf functions

Created on Tue Mar 30 13:50:53 2021

@author: lizz
"""

from netCDF4 import Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyproj as pyproj
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from scipy import signal


### KUTTRUF FUNCTIONS
def getWeights(d,lags):
    # return the weights from the series expansion of the differencing operator
    # for real orders d and up to lags coefficients
    w=[1]
    for k in range(1,lags):
        w.append(-w[-1]*((d-k+1))/k)
    w=np.array(w).reshape(-1,1) 
    return w

def plotWeights(dRange, lags, numberPlots):
    weights=pd.DataFrame(np.zeros((lags, numberPlots)))
    interval=np.linspace(dRange[0],dRange[1],numberPlots)
    for i, diff_order in enumerate(interval):
        weights[i]=getWeights(diff_order,lags)
    weights.columns = [round(x,2) for x in interval]
    fig=weights.plot()
    plt.legend(title='Order of differencing')
    plt.title('Lag coefficients for various orders of differencing')
    plt.xlabel('lag coefficients')
    #plt.grid(False)
    plt.show()

# plotWeights([0,1],7,6)

def ts_differencing(series, order, lag_cutoff):
    # return the time series resulting from (fractional) differencing
    # for real orders order up to lag_cutoff coefficients
    
    weights=getWeights(order, lag_cutoff)
    res=0
    for k in range(lag_cutoff):
        res += weights[k]*series.shift(k).fillna(0)
    return res[lag_cutoff:] 


def plotMemoryVsCorr(result, seriesName):
    fig, ax = plt.subplots()
    ax2 = ax.twinx()  
    color1='xkcd:deep red'; color2='xkcd:cornflower blue'
    ax.plot(result.order,result['adf'],color=color1)
    ax.plot(result.order, result['5%'], color='xkcd:slate')
    ax2.plot(result.order,result['corr'], color=color2)
    ax.set_xlabel('order of differencing')
    ax.set_ylabel('adf', color=color1);ax.tick_params(axis='y', labelcolor=color1)
    ax2.set_ylabel('corr', color=color2); ax2.tick_params(axis='y', labelcolor=color2)
    plt.title('ADF test statistics and correlation for %s' % (seriesName))
    plt.show()


from statsmodels.tsa.stattools import adfuller 
def MemoryVsCorr(series, dRange, numberPlots, lag_cutoff, seriesName):
    # return a data frame and plot comparing adf statistics and linear correlation
    # for numberPlots orders of differencing in the interval dRange up to a lag_cutoff coefficients
    
    interval=np.linspace(dRange[0], dRange[1],numberPlots)
    result=pd.DataFrame(np.zeros((len(interval),4)))
    result.columns = ['order','adf','corr', '5%']
    result['order']=interval
    for counter,order in enumerate(interval):
        seq_traf=ts_differencing(series,order,lag_cutoff)
        res=adfuller(seq_traf, maxlag=1, regression='c') #autolag='AIC'
        result.loc[counter,'adf']=res[0]
        result.loc[counter,'5%']=res[4]['5%']
        result.loc[counter,'corr']= np.corrcoef(series[lag_cutoff:].fillna(0),seq_traf)[0,1]
    plotMemoryVsCorr(result, seriesName)    
    return result
####


helheim_smb_fpath = '/Users/lizz/Documents/GitHub/Data_unsynced/Helheim-processed/smb_rec._.BN_RACMO2.3p2_ERA5_3h_FGRN055.1km.MM.csv'

helheim_smb = pd.read_csv(helheim_smb_fpath, index_col=0, parse_dates=[0])

helheim_fracdiff = ts_differencing(helheim_smb, order=0.4, lag_cutoff=5)

# ## Fit an AR[n] model
# mod = AutoReg(helheim_smb, 5, seasonal=True)
# res = mod.fit()
# print(res.summary())

# fig = plt.figure(figsize=(16,9))
# fig = res.plot_diagnostics(fig=fig, lags=30)

## Fit an ARIMA model
lag_acf = sm.tsa.stattools.acf(helheim_fracdiff, nlags=20)
lag_pacf = sm.tsa.stattools.pacf(helheim_fracdiff, nlags=20, method='ols')
upper_confidence = 1.96/np.sqrt(len(helheim_fracdiff))
p_candidates = np.argwhere(lag_pacf > upper_confidence).squeeze()
p = p_candidates[p_candidates >0][0] # choose first nonzero value that exceeds upper CI
q_candidates = np.argwhere(lag_acf > upper_confidence).squeeze()
q = p_candidates[p_candidates >0][0] 
d = 0 # no differencing in ARIMA; apply only the differencing above in fracdiff
print('Order of ARIMA model: ({},{},{})'.format(p,d,q))

mod = sm.tsa.arima.ARIMA(helheim_fracdiff, order=(p,d,q), freq='M')
mod_fit = mod.fit()
