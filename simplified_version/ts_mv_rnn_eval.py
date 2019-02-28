import sys

import collections
import hashlib
import numbers

from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops.rnn_cell_impl import *

# local 
from utils_libs import *
from ts_mv_rnn import *

# ---- model restore and testing ----

def hyper_para_selection(hpara, error_log, val_epoch_num, test_epoch_num):
    
    val_err = []
    
    for hp_error in error_log:
        
        # based on RMSE
        val_err.append( mean([k[3] for k in hp_error[:val_epoch_num]]) )
    
    idx = val_err.index(min(val_err))
    
    return hpara[idx], [i[0] for i in error_log[idx]][:test_epoch_num], min(val_err)

def test_nn(epoch_samples, x_test, y_test, model_path, method_str, dataset_str):
    
    for idx in epoch_samples:
        
        # path of the stored models 
        tmp_meta = model_path + method_str + '_' + dataset_str + '_' + str(idx) + '.meta'
        tmp_data = model_path + method_str + '_' + dataset_str + '_' + str(idx)
        
        # clear graph
        tf.reset_default_graph()
        
        with tf.device('/device:GPU:0'):
            
            config = tf.ConfigProto()
        
            config.allow_soft_placement = True
            config.gpu_options.allow_growth = True
        
            sess = tf.Session(config = config)
            
            if method_str == 'plain':
                
                # restore the model
                reg = tsLSTM_plain(sess)
                
            elif method_str == 'mv_tensor' or method_str == 'mv_full':
                
                reg = tsLSTM_mv(sess)
                      
                
            reg.pre_train_restore_model(tmp_meta, tmp_data)
            # testing using the restored model
            yh, rmse, mae, mape = reg.pre_train_inference(x_test, y_test, 1.0)
                
    return yh, rmse, mae, mape
