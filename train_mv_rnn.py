#!/usr/bin/python
import sys
import os
import collections
import hashlib
import numbers
import pickle

from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops.rnn_cell_impl import *

import time
import json

# local 
from utils_libs import *
from mv_rnn_cell import *
from ts_mv_rnn import *
from config_hyper_para import *

# fix the random seed to reproduce the results
np.random.seed(1)
tf.set_random_seed(1)

''' 
Arguments:

dataset_str: name of the dataset
method_str: name of the neural network 

'''

# ------ GPU set-up in multi-GPU environment ------

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5" 

# ------ load data ------

# parameters from command line
dataset_str = str(sys.argv[1])
method_str = str(sys.argv[2])

# parameters from config files
import json
with open('config_data.json') as f:
    file_dict = json.load(f)
    
print(" --- Loading files at", file_dict[dataset_str]) 

files_list = file_dict[dataset_str]    
xtrain = np.load(files_list[0], encoding = 'latin1')
xval = np.load(files_list[1], encoding = 'latin1')
ytrain = np.load(files_list[2], encoding = 'latin1')
yval = np.load(files_list[3], encoding = 'latin1')
    
# align the dimension    
ytrain = np.expand_dims(ytrain, 1) 
yval  = np.expand_dims(yval,  1)

# fixed
para_input_dim = np.shape(xtrain)[-1]
para_win_size = np.shape(xtrain)[1]

print(" --- Data shapes: ", np.shape(xtrain), np.shape(ytrain), np.shape(xval), np.shape(yval))


# ------ model set-up ------

# if residual layers are used, keep size of layers the same 
para_bool_residual = False

# regularization
para_dense_regul_type_mv= 'l2'  # l1, l2
para_l2_att_mv = 0.00001

para_bool_regular_lstm = True
para_bool_regular_attention = False
para_bool_regular_dropout_output = False

# layer normalization
para_layer_norm = ''

# learning rate, convergence
para_n_epoch = 100
para_lr_mv = lr_dic[dataset_str]
para_batch_size_mv = batch_size_dic[dataset_str]
para_decay_step = 1000000

# multi-variable architecture
para_rnn_gate_type = "full" if method_str == 'mv_full' else 'tensor'
para_lstm_dims_mv = hidden_dim_dic[dataset_str] 

# attention
para_attention_mv = attention_dic[method_str] # temp, var, var-pre, both-att, both-pool, vari-mv-output'
para_temp_attention_type = 'temp_loc' # loc, concate
para_temp_decay_type = ''  # cutoff
para_vari_attention_type = 'vari_loc_all'
para_vari_attention_after_mv_desne = False

para_loss_type = 'lk'
# mse, lk: likelihood, pseudo_lk 
para_ke_type = 'aggre_posterior' # base_posterior, base_prior


# ------ utility functions ------

def train_nn(num_dense, l2_dense, dropout_keep_prob, log_file, ke_pickle, pred_pickle):   
    
    # ---- train and evaluate the model ----
    
    # clear graph
    tf.reset_default_graph()
    
    # fix the random seed to stabilize the network 
    np.random.seed(1)
    tf.set_random_seed(1)
    
    with tf.device('/device:GPU:0'):
        
        # device_count={'GPU': }
        config = tf.ConfigProto()
        
        config.allow_soft_placement = True
        config.gpu_options.allow_growth = True
        
        sess = tf.Session(config = config)
                
        # fix the random seed to stabilize the network 
        # np.random.seed(1)
        # tf.set_random_seed(1)
        
        # apply max_norm contraint only when dropout is used
        para_keep_prob = dropout_keep_prob
        para_max_norm = maxnorm_dic[dataset_str] if dropout_keep_prob < 1.0 else 0.0
       
        if method_str == 'mv_full':
            
            print('\n\n --- ', method_str, ' parameter: ',\
                  ' num of dense-', num_dense,\
                  ' l2-', l2_dense,\
                  ' dropout-', para_keep_prob,\
                  ' maxnorm-', para_max_norm)
            
            reg = tsLSTM_mv(para_lstm_dims_mv, para_win_size, para_input_dim, sess, \
                            para_lr_mv, para_max_norm, para_bool_residual,\
                            para_attention_mv, para_temp_decay_type, para_temp_attention_type,\
                            l2_dense, para_l2_att_mv, para_vari_attention_type,\
                            para_loss_type, para_dense_regul_type_mv,\
                            para_layer_norm, num_dense, para_ke_type, "full",
                            para_bool_regular_lstm,
                            para_bool_regular_attention,
                            para_bool_regular_dropout_output,
                            para_vari_attention_after_mv_desne
                            )
            
        elif method_str == 'mv_tensor':
            
            print('\n\n --- ', method_str, ' parameter: ', \
                  ' num of dense-', num_dense, \
                  ' l2-', l2_dense, \
                  ' dropout-', para_keep_prob, \
                  ' maxnorm-', para_max_norm)
            
            reg = tsLSTM_mv(para_lstm_dims_mv, para_win_size, para_input_dim, sess, \
                            para_lr_mv, para_max_norm, para_bool_residual,\
                            para_attention_mv, para_temp_decay_type, para_temp_attention_type,\
                            l2_dense, para_l2_att_mv, para_vari_attention_type,\
                            para_loss_type, para_dense_regul_type_mv,\
                            para_layer_norm, num_dense, para_ke_type, "tensor",
                            para_bool_regular_lstm,
                            para_bool_regular_attention,
                            para_bool_regular_dropout_output,
                            para_vari_attention_after_mv_desne
                            )
            
        else:
            print('\n\n [ ERROR] method_str \n\n')
            
        
        # initialize the network
        reg.train_ini()
        reg.inference_ini()
        
        # perpare for data shuffling
        total_cnt = np.shape(xtrain)[0]
        iter_per_epoch = int(total_cnt/para_batch_size_mv) + 1
        total_idx = list(range(total_cnt))
        
        # set up model saver
        # saver = tf.train.Saver(max_to_keep = para_n_epoch)
        
        # epoch training and validation errors
        epoch_error = []
        epoch_ke = []
        epoch_att = []
        epoch_prediction = []
        
        # training time counter 
        st_time = time.time()
        
        # training epoches 
        for epoch in range(para_n_epoch):
            
            st_time_epoch = time.time()
            
            loss_epoch = 0.0
            err_sum_epoch = 0.0
            
            # -- batch training
            
            np.random.shuffle(total_idx)
            
            for i in range(iter_per_epoch):
                
                # shuffle training data
                batch_idx = total_idx[ i*para_batch_size_mv: min((i+1)*para_batch_size_mv, total_cnt) ] 
                batch_x = xtrain[ batch_idx ]
                batch_y = ytrain[ batch_idx ]            
                
                if method_str == 'mv_full' or method_str == 'mv_tensor':
                    
                    # learning rate decay
                    if (i + iter_per_epoch*epoch) != 0 and (i + iter_per_epoch*epoch)%para_decay_step == 0:
                        tmp_loss, tmp_err = reg.train_batch(batch_x, \
                                                        batch_y, \
                                                        para_keep_prob, \
                                                        True, \
                                                        para_lr_mv*(0.96)**((i + iter_per_epoch*epoch)/para_decay_step))
                    else:
                        tmp_loss, tmp_err = reg.train_batch(batch_x, \
                                                        batch_y, \
                                                        para_keep_prob, \
                                                        False, \
                                                        0.0)
                else:
                    tmp_loss, tmp_err = reg.train_batch(batch_x, batch_y, para_keep_prob)
                
                loss_epoch += tmp_loss
                err_sum_epoch += tmp_err
                       
            # -- epoch-wise evaluation
            
            if para_attention_mv == 'both-att':
                
                # [self.y_hat, self.rmse, self.mae, self.mape]
                # [B V T-1], [B V]
                # dropout probability set to 1.0
                yh, rmse_epoch, mae_epoch, mape_epoch, vari_impt = reg.inference(xval, yval, 1.0)
                    
                # knowledge extraction
                # dropout probability set to 1.0
                test_w, att_temp, att_prior, att_poster, importance_vari_temp, importance_vari_prior,\
                importance_vari_posterior = reg.knowledge_extraction(xtrain, ytrain, 1.0)
                    
                # epoch_att.append([att_temp, att_prior, att_poster])
                epoch_ke.append([importance_vari_temp, importance_vari_prior, importance_vari_posterior, vari_impt])
            
            ed_time_epoch = time.time()
            
            train_rmse_epoch = sqrt(1.0*err_sum_epoch/total_cnt)
            
            epoch_prediction.append(yh)
            epoch_error.append([epoch, \
                                loss_epoch*1.0/iter_per_epoch, \
                                train_rmse_epoch, \
                                rmse_epoch, \
                                mae_epoch, \
                                mape_epoch])
            # epoch-wise 
            print("\n --- At epoch %d : \n    %s, %d "%(epoch, str(epoch_error[-1][1:]), ed_time_epoch - st_time_epoch))
            
            # ?
            print("\n                 :  %s  "%(str(vari_impt)))
            
            
            with open(log_file, "a") as text_file:
                text_file.write("%s\n"%(str(epoch_error[-1])[1:-1]))
            
        ed_time = time.time()
        
        print("Optimization Finished!") 
        
        # ---- dump epoch-wise results
        
        if (method_str == 'mv_full' or method_str == 'mv_tensor') and para_attention_mv == 'both-att':            
            
            pickle.dump(epoch_ke, open(ke_pickle + ".p", "wb"))
            
            best_epoch = min(epoch_error, key = lambda x:x[3])[0]
            pickle.dump(list(zip(np.squeeze(yval), np.squeeze(epoch_prediction[best_epoch]))), \
                        open(pred_pickle + ".p", "wb"))
            
        return min(epoch_error, key = lambda x: x[3]), 1.0*(ed_time - st_time)/para_n_epoch
    

def log_func(text_file):
    
    text_file.write("\n---- dataset: %s \n"%(dataset_str))
    text_file.write("dataset shape: %s \n"%(str(np.shape(xtrain))))
    text_file.write("method: %s, %s  \n"%(method_str, attention_dic[method_str]))
    text_file.write("MV layer size: %s \n"%(str(hidden_dim_dic[dataset_str])))
    text_file.write("lr: %s \n"%(str(lr_dic[dataset_str])))
    text_file.write("learnign rate decay iterations : %d \n"%(para_decay_step))
    text_file.write("attention: %s, %s \n"%(para_temp_attention_type, para_vari_attention_type))
    text_file.write("loss type: %s \n"%(para_loss_type))
    text_file.write("batch size: %s \n"%(str(para_batch_size_mv)))
    text_file.write("knowledge extraction type : %s \n"%(para_ke_type))
    text_file.write("rnn gate type : %s \n"%(para_rnn_gate_type))
    text_file.write("maximum norm constraint : %f \n"%(maxnorm_dic[dataset_str]))
    text_file.write("number of epoch : %d \n"%(para_n_epoch))
    text_file.write("regularization on LSTM weights : %s \n"%(para_bool_regular_lstm))
    text_file.write("regularization on attention : %s \n"%(para_bool_regular_attention))
    text_file.write("dropout before the outpout layer : %s \n"%(para_bool_regular_dropout_output))
    text_file.write("variable attention after mv_desne : %s \n"%(para_vari_attention_after_mv_desne))
    
    return

# ------ main train and validation process ------

'''
Log and dump files:

ts_rnn.txt: overall errors, all method, all set-up

log_method_dataset: epoch level training errors, method dataset wise

ke_pickle: only for MV-RNN, set-up wise

pred_pickle: only for MV-RNN, set-up wise

'''

if __name__ == '__main__':
    
    # log: overall erros, hyperparameter
    with open("../../ts_results/ts_rnn.txt", "a") as text_file:
        log_func(text_file)
        
    # log: epoch files
    log_epoch_file = "../../ts_results/log_" + method_str + "_" + dataset_str + ".txt"
    
    with open(log_epoch_file, "a") as text_file:
        log_func(text_file)
        
    # grid search process
    validate_tuple = []
    
    #for para_lr_mv in [0.001, 0.002, 0.005]
    for tmp_num_dense in [0, 1]:
        for tmp_keep_prob in [1.0, 0.8, 0.5]:
            for tmp_l2 in [0.00001, 0.0001, 0.001, 0.01]:
                
                with open(log_epoch_file, "a") as text_file:
                    text_file.write("\n num_dense: %d, keep_prob: %f, l2: %f \n"%(tmp_num_dense, tmp_keep_prob, tmp_l2))
                
                # pickle: ke - knowledge extraction
                ke_file = "../../ts_results/ke_" + \
                          str(method_str) + "_" \
                          + str(dataset_str) + "_" \
                          + str(tmp_num_dense) + \
                            str(tmp_keep_prob) + \
                            str(tmp_l2) + "_"
                            
                # pickle: predictions            
                pred_file = "../../ts_results/pred_" + \
                            str(method_str) + "_" \
                            + str(dataset_str) + "_" \
                            + str(tmp_num_dense) + \
                              str(tmp_keep_prob) + \
                            str(tmp_l2) + "_"
             
                # -- training
                
                error_tuple, epoch_time = train_nn(tmp_num_dense, 
                                                   tmp_l2, 
                                                   tmp_keep_prob, 
                                                   log_epoch_file,
                                                   ke_file,
                                                   pred_file)
                
                validate_tuple.append(error_tuple) 
                
                print('\n --- current running: ', tmp_num_dense, tmp_keep_prob, tmp_l2, validate_tuple[-1], '\n')
                
                # log: overall errors, performance for one hyperparameter set-up
                with open("../../ts_results/ts_rnn.txt", "a") as text_file:
                    text_file.write( "%f %f %f %s %s \n"%(tmp_num_dense, 
                                                          tmp_keep_prob, 
                                                          tmp_l2, 
                                                          str(validate_tuple[-1]), 
                                                          str(epoch_time)))
                    
    with open("../../ts_results/ts_rnn.txt", "a") as text_file:
        text_file.write( "\n  ") 
