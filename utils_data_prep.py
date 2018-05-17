# data processing packages
import numpy as np   
import pandas as pd 

from scipy import stats
from scipy import linalg
from scipy import *

import random

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import utils_libs

    
# ---- data pre-processing ----

# for both univeriate and multi-variate cases
def instance_extraction( list_ts, win_size, is_stateful ):
    
    n = len(list_ts)
    if n < win_size:
        print "ERROR: SIZE"
        return
    
    listX = []
    listY = []
    
    if is_stateful:
        for i in range(win_size, n, win_size):
            listX.append( list_ts[i-win_size:i] )
            listY.append( list_ts[i] )
    else:
        for i in range(win_size, n):
            listX.append( list_ts[i-win_size:i] )
            listY.append( list_ts[i] )
        
    return listX, listY


# for the case of multiple independent and one target series 
# argu: list
# return: list
def instance_extraction_multiple_one( list_target, list_indepen, win_size, is_stateful ):
    
    n = len(list_target)
    if n < win_size:
        print "ERROR: SIZE"
        return
    
    listX = []
    listY = []
    
    if is_stateful:
        
        for i in range(win_size, n, win_size):
            tmp  = list_indepen[i-win_size:i]
            tmp1 = np.expand_dims(list_target[i-win_size:i], axis=1) 
            #tmp1 = np.reshape( list_target[i-win_size:i], [-1,1] )
            tmp  = np.append( tmp, tmp1 , axis = 1 )
                    
            listX.append( tmp )
            listY.append( list_target[i] )
    else:
        for i in range(win_size, n):
            tmp  = list_indepen[i-win_size:i]
            tmp1 = np.expand_dims(list_target[i-win_size:i], axis=1)
            tmp  = np.append( tmp, tmp1 , axis = 1 )
                    
            listX.append( tmp )
            listY.append( list_target[i] )
        
    return listX, listY

def instance_extraction_multiple_one_separate_target( list_target, list_indepen, win_size, is_stateful ):
    
    n = len(list_target)
    if n < win_size:
        print "ERROR: SIZE"
        return
    
    listX = []
    listY_his = []
    listY = []
 
    for i in range(win_size, n):
        tmp  = list_indepen[i-win_size:i+1]
        tmp1 = list_target[i-win_size:i]
                    
        listX.append(tmp)
        listY_his.append(tmp1)
        listY.append(list_target[i])
        
    return listX, listY_his, listY


# Utilities 

# expand y, y_t -> y_1,...y_t
# in x, y is on the last column
# argu: np.matrix
# return: np.matrix
def expand_y( x, y ):
    cnt = len(x)
    expand_y = []
    
    if np.shape(x)[2]==1:
        
        tmpx = list(x)
        tmpy = list(y)
        for i in range(cnt):
            tmp = tmpx[i][1:]
            tmp = np.append( tmp, tmpy[i] )
            expand_y.append( tmp )
    
        return np.array( expand_y )
    
    elif np.shape(x)[2]>1:
        
        tmpx = np.transpose(x,[2,0,1])
        tmpx = tmpx[-1]
        
        tmpy = list(y)
        for i in range(cnt):
            tmp = tmpx[i][1:]
            tmp = np.append( tmp, tmpy[i] )
            expand_y.append( tmp )
    
        return np.array( expand_y )
        

def expand_x_trend( x ):
    
    shape = np.shape(x)
    cnt   = shape[0]
    steps = shape[1]
    
    tmp_x = list(x)
    if len( shape ) == 2:
      
        res_x = []
        for i in range(cnt):
            res_x.append( [ (tmp_x[i][j], tmp_x[i][j] - tmp_x[i][j-1]) for j in range(1,steps)] )          
            
    elif len( shape ) == 3:
        
        n_dim = shape[2]
    
        res_x = []
        for i in range(cnt):
            res_x.append([])
            for j in range(1, steps):
                res_x[-1].append( [ (tmp_x[i][j][k], tmp_x[i][j][k] - tmp_x[i][j-1][k]) for k in range(n_dim)] )    
                        
    res_x = np.array(res_x)
    
    return np.reshape(res_x, [cnt, steps-1, -1])


def conti_normalization_train_dta(dta_df):
    
    return preprocessing.scale(dta_df)

def conti_normalization_test_dta(dta_df, train_df):
    
    mean_dim = np.mean(train_df, axis=0)
    std_dim = np.std(train_df, axis=0)
    
#    print '--test--', mean_dim, std_dim
    
    df=pd.DataFrame()
    cols = train_df.columns
    idx=0
    
#    print '--test--', cols
    
    for i in cols:
        
        df[i] = (dta_df[i]- mean_dim[idx])*1.0/(std_dim[idx]+1e-10)
        idx=idx+1
        
    return df.as_matrix()

# normalize features
def prepare_train_test_data(bool_add_trend, files_list):
    
    PARA_ADD_TREND = bool_add_trend
                      
    xtr = np.load(files_list[0])
    xts = np.load(files_list[1])
    ytr = np.load(files_list[2])
    yts = np.load(files_list[3]) 
                    
    cnt_tr = len(xtr)
    cnt_ts = len(xts)   
    
    original_shape_tr = np.shape(xtr)
    original_shape_ts = np.shape(xts)
                      
    # integrate trends
    if PARA_ADD_TREND == True:
        
        trend_xtrain = expand_x_trend( xtr )
        trend_xtest  = expand_x_trend( xts )
    
        tmp_xtrain = np.reshape( trend_xtrain, [cnt_tr, -1 ] )
        tmp_xtest  = np.reshape( trend_xtest,  [cnt_ts, -1 ] )
    
        xtrain_df = pd.DataFrame( tmp_xtrain )
        xtest_df  = pd.DataFrame( tmp_xtest )
        
    else:
        
        tmp_xtrain = np.reshape( xtr, [cnt_tr, -1 ] )
        tmp_xtest  = np.reshape( xts, [cnt_ts, -1 ] )
    
        xtrain_df = pd.DataFrame( tmp_xtrain )
        xtest_df  = pd.DataFrame( tmp_xtest )
        

    xtest = conti_normalization_test_dta(  xtest_df, xtrain_df )
    xtrain= conti_normalization_train_dta( xtrain_df )
        
    return xtrain, ytr, xtest, yts, original_shape_tr, original_shape_ts




def build_training_testing_data_4learning( dta_df, target_col, indep_col, \
                                para_uni_variate, para_train_test_split, para_win_size, \
                                para_train_range, para_test_range, is_stateful):
        
# univariate
    if para_uni_variate == True:
        
        x_all, y_all = instance_extraction( \
                       list(dta_df[target_col][ para_train_range[0]:para_train_range[1] ]), \
                                           para_win_size, is_stateful )

# multiple independent and one target series
    else:
        dta_mat = dta_df[ indep_col ][ para_train_range[0]:para_train_range[1] ].as_matrix()
        
        x_all, y_all = instance_extraction_multiple_one( \
                        list(dta_df[ target_col ][ para_train_range[0]:para_train_range[1] ]),\
                                                 list(dta_mat),para_win_size, is_stateful )
# multivariate 
# x_all, y_all = instance_extraction( list(dta_df[['Open','High','Low','Volume']][:4000]), 100 )


# downsample the whole data
    total_idx = range(len(x_all))
    np.random.shuffle(total_idx)

    x_all = np.array(x_all)
    y_all = np.array(y_all)

    x_all = x_all[ total_idx[: int(1.0*len(total_idx))] ]
    y_all = y_all[ total_idx[: int(1.0*len(total_idx))] ]
    
    
    tmp_x_train, tmp_x_test, tmp_y_train, tmp_y_test = \
    train_test_split( x_all, y_all, test_size = 0.2, random_state = 20)

    
# training and testing data

# by extracting from subsequent data
    if para_train_test_split == False:
        
#       for test data, no needs of stateful test instances 
        if para_uni_variate == False:
            
            dta_mat = dta_df[ indep_col ][ para_test_range[0]:para_test_range[1] ].as_matrix()

            x_test, y_test = instance_extraction_multiple_one(\
                             list(dta_df[ target_col ][ para_test_range[0]:para_test_range[1] ]),\
                                                           list(dta_mat),para_win_size, False )
        else:
            x_test, y_test = instance_extraction( \
                             list(dta_df[ target_col ][ para_test_range[0]:para_test_range[1] ]), \
                                                 para_win_size, False )
            
        x_train = np.array(x_all)
        y_train = np.array(y_all)
        x_test = np.array(x_test)
        y_test = np.array(y_test)
        
#       return all data in the train_range  
        return x_train, x_test, y_train, y_test

# by randomly split
    else:
        return  tmp_x_train, tmp_x_test, tmp_y_train, tmp_y_test



def build_training_testing_data_4statistics( dta_df, target_col, indep_col, \
                                para_uni_variate, \
                                para_train_range, para_test_range):
# univariate
    if para_uni_variate == True:
        
        x_train = dta_df[target_col][ para_train_range[0]:para_train_range[1] ].as_matrix()
        
        x_test  = dta_df[target_col][ para_test_range[0]:para_test_range[1] ].as_matrix()

# multiple independent and one target series
    else:
        
        x_train = dta_df[ indep_col+[target_col] ][ para_train_range[0]:para_train_range[1] ].as_matrix()
        
        x_test  = dta_df[ indep_col+[target_col] ][ para_test_range[0]:para_test_range[1] ].as_matrix()
        
    return x_train, x_test



# ---- RETAIN baseline in NIPS paper ----   
def prepare_train_test_RETAIN(files_list, dump_path, dump_prefix):
    
    # normalize features
    xtrain, ytrain, xtest, ytest, tr_shape, ts_shape = prepare_train_test_data(False, files_list)
    #print np.shape(xtrain), np.shape(ytrain), np.shape(xtest), np.shape(ytest)
    
    # automatically format the dimensions for univairate or multi-variate cases 
    # always in formt [#, time_steps, data dimension]
    
    if len(tr_shape)==2:
        xtrain = np.expand_dims( xtrain, 2 )
        xtest  = np.expand_dims( xtest,  2 )
        
    elif len(tr_shape)==3:
        xtrain = np.reshape( xtrain, tr_shape )
        xtest  = np.reshape( xtest,  ts_shape )

    #ytrain = np.expand_dims( ytrain, 1 ) 
    #ytest  = np.expand_dims( ytest,  1 )
        
    print np.shape(xtrain), np.shape(ytrain), np.shape(xtest), np.shape(ytest)
    
    xtrain.dump(dump_path + dump_prefix + "_xtrain_nips.dat")
    xtest.dump(dump_path + dump_prefix + "_xtest_nips.dat")
    
    ytrain.dump(dump_path + dump_prefix + "_ytrain_nips.dat")
    ytest.dump(dump_path + dump_prefix + "_ytest_nips.dat")
    
    
# ---- Dual-RNN baseline in IJCAI paper ---- 
#def prepare_train_test_DualRNN(dta_df, target_col, feature_cols, train_range, test_range, dump_path, dump_prefix,\
#                              win_size, bool_stateful, bool_log_y):

def prepare_train_test_DualRNN( xtr, ytr, xts, yts, dump_path, dump_prefix,\
                                win_size, bool_stateful, bool_log_y):
    
    #dta_mat = dta_df[ feature_cols ][ train_range[0]:train_range[1] ].as_matrix()

    #x_train, yhis_train, y_train = instance_extraction_multiple_one_separate_target( \
    #                    list(dta_df[ target_col ][ train_range[0]:train_range[1] ]),\
    #                                             list(dta_mat), win_size, bool_stateful )

    #dta_mat = dta_df[ feature_cols ][ test_range[0]:test_range[1] ].as_matrix()

    #x_test, yhis_test, y_test = instance_extraction_multiple_one_separate_target( \
    #                    list(dta_df[ target_col ][ test_range[0]:test_range[1] ]),\
    #                                             list(dta_mat), win_size, bool_stateful )
    
    tmpshape = np.shape(xtr)[2]
    
    x_train, yhis = np.split(xtr, [tmpshape-1,], axis = 2)
    yhis_train = np.squeeze(yhis, [2])
    
    x_test, yhis = np.split(xts, [tmpshape-1,], axis = 2)
    yhis_test = np.squeeze(yhis, [2])
    
    y_train = ytr
    y_test = yts
    

    # -- normalization
    # X
    cnt_tr = len(x_train)
    cnt_ts = len(x_test)

    tmp_xtrain = np.reshape( x_train, [cnt_tr, -1 ] )
    tmp_xtest  = np.reshape( x_test,  [cnt_ts, -1 ] )
    
    xtrain_df = pd.DataFrame( tmp_xtrain )
    xtest_df  = pd.DataFrame( tmp_xtest )

    xtest = conti_normalization_test_dta(  xtest_df, xtrain_df )
    xtrain= conti_normalization_train_dta( xtrain_df )

    xtrain = np.reshape(xtrain, np.shape(x_train))
    xtest = np.reshape(xtest, np.shape(x_test))

    # Y
    yhtrain_df = pd.DataFrame( yhis_train )
    yhtest_df  = pd.DataFrame( yhis_test )

    yhtest = conti_normalization_test_dta(  yhtest_df, yhtrain_df )
    yhtrain= conti_normalization_train_dta( yhtrain_df )

    # ---
    if bool_log_y == True:
        ytrain = log(np.asarray(y_train)+1e-5)
        ytest  = log(np.asarray(y_test)+1e-5)
        
    else:
        ytrain = np.asarray(y_train)
        ytest = np.asarray(y_test)

    print np.shape(xtrain), np.shape(yhtrain), np.shape(ytrain)
    print np.shape(xtest), np.shape(yhtest), np.shape(ytest)
    
    xtrain.dump(dump_path + dump_prefix + "_xtrain_dual.dat")
    ytrain.dump(dump_path + dump_prefix + "_ytrain_dual.dat")
    xtest.dump(dump_path + dump_prefix + "_xtest_dual.dat")
    ytest.dump(dump_path + dump_prefix + "_ytest_dual.dat")

    yhtrain.dump(dump_path + dump_prefix + "_hytrain_dual.dat")
    yhtest.dump(dump_path + dump_prefix + "_hytest_dual.dat")



###############################################################################################33
#TO DO

def prepare_trend_train_test_data( steps, bool_add_trend, xtrain_df, xtest_df, ytrain_df, ytest_df):
    
    PARA_STEPS = steps
    PARA_ADD_TREND = bool_add_trend
    
    # integrate trends
    if PARA_ADD_TREND == True:
        
        trend_xtrain = expand_x_trend( xtrain_df.as_matrix() )
        trend_xtest  = expand_x_trend( xtest_df.as_matrix() )
    
        tmp_xtrain = np.reshape( trend_xtrain, [-1, (PARA_STEPS-1)*2 ] )
        tmp_xtest  = np.reshape( trend_xtest,  [-1, (PARA_STEPS-1)*2 ] )
    
        xtrain_df = pd.DataFrame( tmp_xtrain )
        xtest_df  = pd.DataFrame( tmp_xtest )
    
#   normalize x in training and testing datasets
        xtest = conti_normalization_test_dta(xtest_df, xtrain_df)
        xtrain= conti_normalization_train_dta(xtrain_df)

#   trend enhanced
        xtest  = np.reshape( xtest,  [-1, (PARA_STEPS-1), 2 ] )
        xtrain = np.reshape( xtrain, [-1, (PARA_STEPS-1), 2 ] )
        
    else:
#   normalize x in training and testing datasets
        xtest = conti_normalization_test_dta(xtest_df, xtrain_df)
        xtrain= conti_normalization_train_dta(xtrain_df)

    ytrain = ytrain_df.as_matrix()
    ytest  = ytest_df.as_matrix()
        
    return xtrain, ytrain, xtest, ytest


def prepare_lastPoint_train_test_data( steps, bool_add_trend, xtrain_df, xtest_df, ytrain_df, ytest_df):
    
    PARA_STEPS = steps
    PARA_ADD_TREND = bool_add_trend
    
    # integrate trends
    if PARA_ADD_TREND == True:
        
        trend_xtrain = expand_x_trend( xtrain_df.as_matrix() )
        trend_xtest  = expand_x_trend( xtest_df.as_matrix() )
    
        tmp_xtrain = np.reshape( trend_xtrain, [-1, (PARA_STEPS-1)*2 ] )
        tmp_xtest  = np.reshape( trend_xtest,  [-1, (PARA_STEPS-1)*2 ] )
    
        xtrain_df = pd.DataFrame( tmp_xtrain )
        xtest_df  = pd.DataFrame( tmp_xtest )
    
#   normalize x in training and testing datasets
        xtest = conti_normalization_test_dta(xtest_df, xtrain_df)
        xtrain= conti_normalization_train_dta(xtrain_df)

#   trend enhanced
        xtest  = np.reshape( xtest,  [-1, (PARA_STEPS-1), 2 ] )
        xtrain = np.reshape( xtrain, [-1, (PARA_STEPS-1), 2 ] )
        
    else:
#   normalize x in training and testing datasets
        xtest = conti_normalization_test_dta(xtest_df, xtrain_df)
        xtrain= conti_normalization_train_dta(xtrain_df)

    ytrain = ytrain_df.as_matrix()
    ytest  = ytest_df.as_matrix()
        
    return xtrain, ytrain, xtest, ytest




def flatten_features(dta):
    tmplen = np.shape(dta)[0]
    return np.reshape(dta, [tmplen,-1])