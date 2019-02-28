#!/usr/bin/python

# data processing packages
import numpy as np   
import pandas as pd 
import scipy as sp

import pylab

from pandas import *
from numpy import *
from scipy import *

import random
import sys

# machine leanring packages
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import xgboost as xgb

# statiscal models
import statsmodels as sm
from statsmodels.tsa.stattools import acf  
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.api import VAR, DynamicVAR

from statsmodels.stats import diagnostic


# local packages
from utils_data_prep import *
# from ml_models import *
#from utils_keras import *


def stationary_test( arr ):
    # null one unit-root
    print 'ADF test:\n', sm.tsa.stattools.adfuller(arr, regression='c', maxlag=None, store=False, autolag='AIC')
    # kpss
    # null is stationarity, no deterministic trend component
    print 'KPSS test:\n', sm.tsa.stattools.kpss(arr, regression='c', lags=None, store=False)


# Automatic causality discovery
# Steps: 1. stationary check and stationarize if necessary
#        2. apply VAR
#        3. test Granger causality 

def ts_stationarize_diff( ts ):
    
    cur_ts = ts
    p_val = sm.tsa.stattools.adfuller(cur_ts,\
                                      regression='c', \
                                      maxlag=None, store=False)[1]
    diff_cnt = 0 
    while p_val > 0.01:
                
        pre_ts = cur_ts[1:]
        post_ts = cur_ts[:-1]
        
        diff_cnt += 1
        
        tmplen= len(pre_ts)
        
        cur_ts = [ pre_ts[i] - post_ts[i] for i in range(tmplen) ] 
        
        p_val = sm.tsa.stattools.adfuller(cur_ts,\
                                          regression='c', \
                                          maxlag=None, store=False)[1]
        
    return diff_cnt, p_val, cur_ts

def multi_ts_stationarize( dta ):
    
    post_len = []
    post_dta = []
    
    num_ts = len(dta[0])
    len_ts = len(dta)
     
    for i in range(num_ts):
        tmpts = [ dta[j][i] for j in range(len_ts) ]
        tmpts = ts_stationarize_diff( tmpts )
        
        print "stationary prepro: ", tmpts[0], tmpts[1], '\n'
        
        post_len.append( len(tmpts[2]) )
        post_dta.append( tmpts[2] )
        
    min_len = min(post_len)
        
    res_dta = []
    for i in range(num_ts):
        res_dta.append( post_dta[i][ post_len[i]-min_len : post_len[i] ] )
        
    return np.transpose(res_dta,[1, 0])

def causality_VAR(post_ts, max_order):
    
    model =  VAR(post_ts)
    best_lag = model.select_order(max_order, verbose= False)
    
    print 'best lag: ', best_lag
    
    result = model.fit(best_lag['aic'])
    
    return result, best_lag

def causality_pairwise(VAR_res, post_ts):
    
    num_ts = len(post_ts[0])
    
    causing = []
    
    for i in range(num_ts):
        causing.append( [] )
        
        for j in range(num_ts):
            if j!=i:
                tmp_res = VAR_res.test_causality(i, [j], kind='wald', \
                                                verbose=False)
                
                if tmp_res['pvalue']< 0.05:
                    causing[-1].append(j)
    
    return causing

def causality_one(VAR_res, post_ts, ts_id):
    
    num_ts = len(post_ts[0])
    causing = []
        
    for j in range(num_ts):
        if j != ts_id:
            tmp_res = VAR_res.test_causality(ts_id, [j], kind='wald', \
                                                verbose=False)                
            if tmp_res['pvalue']< 0.01:
                causing.append(j)
    
    return causing


# Length of temporal dependence on individual exogenous variable
# via VAR
# argu: preprocessed time series
def temporal_detect_individual(target_idx, dta, maxlag):
    
    num_ts = len(dta[0])
    len_ts = len(dta)
     
    tmp_target = [ dta[j][target_idx] for j in range(len_ts) ] 
    
    res_lag = []
    
    for i in range(num_ts):
        if i != target_idx:
            
            tmp_ts = [ dta[j][i] for j in range(len_ts) ]
            tmp_x = zip(tmp_target, tmp_ts )
            
#             print np.shape(tmp_x)
            
            model =  VAR(tmp_x)
            best_lag = model.select_order(maxlag, verbose= False)
            
            res_lag.append(best_lag)
    
    return res_lag 