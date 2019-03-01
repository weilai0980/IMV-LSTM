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
from ts_mv_rnn_eval import *
from config_hyper_para_mv import *

''' 
Arguments:

dataset_str: name of the dataset
method_str: name of the neural network
impt_str: variable importance learning 

'''

# ------ GPU set-up in multi-GPU environment ------

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5" 

# ------ arguments ------
    
# from command line
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', '-d', help = "dataset", type = str)
parser.add_argument('--model', '-m', help = "model", type = str, default = 'mv_full')
parser.add_argument('--importance', '-i', help = "importance learning method", type = str, default = '')
 
args = parser.parse_args()
print(args)  

dataset_str = args.dataset
method_str = args.model
impt_str = args.importance

# from config.json
import json
with open('config_data.json') as f:
    file_dict = json.load(f)
    
    
path_result = "../../ts_results/"
path_hyper_para =  "../../ts_results/hyper_para/"
path_model = "../../ts_results/model/"

# ------ load data ------

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


# ------ hyperparameter set-up ------

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

para_lr_decay = False
para_lr_decay_iter = 1000000

# multi-variable architecture
para_rnn_gate_type = "full" if method_str == 'mv_full' else 'tensor'
para_lstm_dims_mv = hidden_dim_dic[dataset_str] 

# attention
para_attention_mv = "both-att" # "both-fusion", "both-att"
para_temp_attention_type = 'temp_loc' # loc, concate
para_vari_attention_type = 'vari_loc_all'
para_vari_attention_after_mv_desne = False

para_vari_impt_learning = impt_str

para_loss_type = 'mse'
# lk: likelihood, mse
para_ke_type = 'aggre_posterior' 
# base_posterior, base_prior

# epoch sample
para_val_epoch_num = max(1, int(0.05 * para_n_epoch)) # how many epoch results to average in validation
para_test_epoch_num = 1 # how many epoch model snapshots to store for use in testing


# ------ utility functions ------

def train_nn(num_dense, 
             l2_dense, 
             dropout_keep_prob, 
             log_file, 
             epoch_set):
    
    # log: epoch errors
    with open(log_file, "a") as text_file:
        text_file.write("\n num_dense: %d, keep_prob: %f, l2: %f \n"%(num_dense, dropout_keep_prob, l2_dense))
    
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
        np.random.seed(1)
        tf.set_random_seed(1)
        
        # apply max_norm contraint only when dropout is used
        para_max_norm = maxnorm_dic[dataset_str] if dropout_keep_prob < 1.0 else 0.0
        
        print('\n\n --- ', method_str, 
              ' parameter: ',
              ' num of dense-', num_dense,
              ' dropout keep prob-', dropout_keep_prob,
              ' l2-', l2_dense,
              ' maxnorm-', para_max_norm, 
              '\n')
        
        reg = tsLSTM_mv(sess)
        
        if method_str == 'mv_full':
            
            reg.network_ini(para_lstm_dims_mv, 
                            para_win_size, 
                            para_input_dim,
                            para_lr_mv, 
                            para_max_norm, 
                            para_bool_residual,
                            para_attention_mv, 
                            para_temp_attention_type,
                            l2_dense, 
                            para_l2_att_mv, 
                            para_vari_attention_type,
                            para_loss_type, 
                            para_dense_regul_type_mv,
                            para_layer_norm, 
                            num_dense, 
                            para_ke_type, 
                            "full",
                            para_bool_regular_lstm,
                            para_bool_regular_attention,
                            para_bool_regular_dropout_output,
                            para_vari_attention_after_mv_desne,
                            para_vari_impt_learning,
                            para_lr_decay
                            )
            
        elif method_str == 'mv_tensor':
            
            reg.network_ini(para_lstm_dims_mv, 
                            para_win_size, 
                            para_input_dim,
                            para_lr_mv, 
                            para_max_norm, 
                            para_bool_residual,
                            para_attention_mv, 
                            para_temp_attention_type,
                            l2_dense, 
                            para_l2_att_mv, 
                            para_vari_attention_type,
                            para_loss_type, 
                            para_dense_regul_type_mv,
                            para_layer_norm, 
                            num_dense, 
                            para_ke_type, 
                            "tensor",
                            para_bool_regular_lstm,
                            para_bool_regular_attention,
                            para_bool_regular_dropout_output,
                            para_vari_attention_after_mv_desne,
                            para_vari_impt_learning,
                            para_lr_decay
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
        
        # model saver
        saver = tf.train.Saver()
        
        # epoch training and validation errors
        epoch_error = []
        
        # training time counter 
        st_time = time.time()
        
        # training epoches 
        for epoch in range(para_n_epoch):
            
            st_time_epoch = time.time()
            
            loss_epoch = 0.0
            err_sum_epoch = 0.0
            
            # -- batch training
            
            # re-shuffle training data
            np.random.shuffle(total_idx)
            
            for i in range(iter_per_epoch):
                
                # batch training data
                batch_idx = total_idx[ i*para_batch_size_mv: min((i+1)*para_batch_size_mv, total_cnt) ] 
                batch_x = xtrain[ batch_idx ]
                batch_y = ytrain[ batch_idx ]
                
                # learning rate decay
                if para_lr_decay == True and iter_per_epoch*epoch != 0 and (i + iter_per_epoch*epoch)%para_lr_decay_iter == 0:
                    
                    tmp_loss, tmp_err = reg.train_batch(batch_x,
                                                        batch_y,
                                                        dropout_keep_prob,
                                                        True,
                                                        para_lr_mv*(0.96)**((i + iter_per_epoch*epoch)/para_lr_decay_iter))
                
                else:
                    tmp_loss, tmp_err = reg.train_batch(batch_x,
                                                        batch_y,
                                                        dropout_keep_prob,
                                                        False,
                                                        0.0)
                
                loss_epoch += tmp_loss
                err_sum_epoch += tmp_err
            
            # -- epoch-wise evaluation
            
            # [self.y_hat, self.rmse, self.mae, self.mape]
            # [B V T-1], [B V]
            # dropout probability set to 1.0
            yh, rmse_epoch, mae_epoch, mape_epoch = reg.inference(xval, 
                                                                  yval, 
                                                                  1.0)
            
            ed_time_epoch = time.time()
            
            # epoch, loss, train rmse, vali. rmse, vali. mae, vali. mape
            epoch_error.append([epoch,
                                loss_epoch*1.0/iter_per_epoch,
                                sqrt(1.0*err_sum_epoch/total_cnt),
                                rmse_epoch,
                                mae_epoch,
                                mape_epoch])
            
            
            print("\n --- At epoch %d : \n    %s, %d "%(epoch, str(epoch_error[-1][1:]), ed_time_epoch - st_time_epoch))
            
            with open(log_file, "a") as text_file:
                text_file.write("%s\n"%(str(epoch_error[-1])[1:-1]))
                
            # save the model w.r.t. the epoch in epoch_set
            if epoch in epoch_set:
                
                saver.save(sess, path_model + method_str + '-' + str(epoch))
                print("    [MODEL SAVED] \n")
                
                
        ed_time = time.time()
        
        print("Optimization Finished!") 
        
        return sorted(epoch_error, key = lambda x: x[3]), 1.0*(ed_time - st_time)/para_n_epoch
    

def log_train(text_env):
    
    text_env.write("\n---- dataset: %s \n"%(dataset_str))
    text_env.write("dataset shape: %s \n"%(str(np.shape(xtrain))))
    text_env.write("method: %s, %s  \n"%(method_str, para_attention_mv))
    text_env.write("MV layer size: %s \n"%(str(hidden_dim_dic[dataset_str])))
    text_env.write("lr: %s \n"%(str(lr_dic[dataset_str])))
    text_env.write("learnign rate decay : %s, %d \n"%(para_lr_decay, para_lr_decay_iter))
    text_env.write("attention: %s, %s \n"%(para_temp_attention_type, para_vari_attention_type))
    text_env.write("loss type: %s \n"%(para_loss_type))
    text_env.write("batch size: %s \n"%(str(para_batch_size_mv)))
    text_env.write("knowledge extraction type : %s \n"%(para_ke_type))
    text_env.write("rnn gate type : %s \n"%(para_rnn_gate_type))
    text_env.write("maximum norm constraint : %f \n"%(maxnorm_dic[dataset_str]))
    text_env.write("number of epoch : %d \n"%(para_n_epoch))
    text_env.write("regularization on LSTM weights : %s \n"%(para_bool_regular_lstm))
    text_env.write("regularization on attention : %s \n"%(para_bool_regular_attention))
    text_env.write("dropout before the outpout layer : %s \n"%(para_bool_regular_dropout_output))
    text_env.write("variable attention after mv_desne : %s \n"%(para_vari_attention_after_mv_desne))
    text_env.write("variable importance learning : %s \n"%(para_vari_impt_learning))
    
    text_env.write("epoch num in validation : %s \n"%(para_val_epoch_num))
    text_env.write("epoch ensemble num in testing : %s \n\n"%(para_test_epoch_num))
    
    return

def log_val(text_env, best_hpara, epoch_set, best_val_err):
    
    text_env.write("\n best hyper parameters: %s %s \n"%(str(best_hpara), str(epoch_set)))
    text_env.write(" best validation errors: %s \n"%(str(best_val_err)))
    
    return

def log_test(text_env, errors):
    
    text_env.write("\n testing error: %s \n\n"%(errors))
    
    return

# ------ main process ------

if __name__ == '__main__':
    
    # log: overall erros, hyperparameter
    log_err_file = path_result + "ts_mv.txt"
    with open(log_err_file, "a") as text_file:
        log_train(text_file)
        
    # log: epoch files
    log_epoch_file = path_result + "log_" + method_str + "_" + dataset_str + ".txt"
    with open(log_epoch_file, "a") as text_file:
        log_train(text_file)
    
    # fix the random seed to reproduce the results
    np.random.seed(1)
    tf.set_random_seed(1)

    # ------ training and validation
    
    # grid search process
    hpara = []
    hpara_err = []
    
    #for para_lr_mv in [0.001, 0.002, 0.005]
    for tmp_num_dense in [0, 1]:
        for tmp_keep_prob in [1.0, 0.8]:
            for tmp_l2 in [0.00001, 0.0001, 0.001, 0.01]:
                
                # -- training
                
                error_epoch_log, epoch_time = train_nn(num_dense = tmp_num_dense, 
                                                       l2_dense = tmp_l2, 
                                                       dropout_keep_prob = tmp_keep_prob, 
                                                       log_file = log_epoch_file,
                                                       epoch_set = [])
                
                hpara.append([tmp_num_dense, tmp_keep_prob, tmp_l2])
                hpara_err.append(error_epoch_log)
                
                print('\n --- current running: ', tmp_num_dense, tmp_keep_prob, tmp_l2, error_epoch_log[0], '\n')
                
                # log: overall errors, performance for one hyperparameter set-up
                with open(log_err_file, "a") as text_file:
                    text_file.write( "%f %f %f %s %s \n"%(tmp_num_dense, 
                                                          tmp_keep_prob, 
                                                          tmp_l2, 
                                                          str(error_epoch_log[0]), 
                                                          str(epoch_time)))
                    
    with open(log_err_file, "a") as text_file:
        text_file.write("\n")

    # ------ re-training
    
    # fix the random seed to reproduce the results
    np.random.seed(1)
    tf.set_random_seed(1)
    
    # choose the best hyper-parameters based on RMSE, changeable 
    best_hpara, epoch_sample, best_val_err = hyper_para_selection(hpara = hpara, 
                                                                  error_log = hpara_err, 
                                                                  val_epoch_num = para_val_epoch_num, 
                                                                  test_epoch_num = para_test_epoch_num)
    best_num_dense = best_hpara[0]
    best_keep_prob = best_hpara[1]
    best_l2 = best_hpara[2]
    
    
    # result recording
    print('\n\n ----- re-traning ------ \n')
    
    print('best hyper parameters: ', best_hpara, epoch_sample, '\n')
    print('best validation errors: ', best_val_err, '\n')
    
    with open(log_err_file, "a") as text_file:
        log_val(text_file, best_hpara, epoch_sample, best_val_err)
        
    import json
    with open(path_hyper_para + dataset_str + '_' + method_str + '_' + para_attention_mv + '.json', 'w') as fp:
        
        tmp_hyper_para = {'num_dense':best_hpara[0],
                          'keep_prob':best_hpara[1],
                          'l2':best_hpara[2]
                         }
        
        json.dump(tmp_hyper_para, fp)
    
    # start the re-training
    error_epoch_log, epoch_time = train_nn(num_dense = best_num_dense, 
                                           l2_dense = best_l2, 
                                           dropout_keep_prob = best_keep_prob, 
                                           log_file = log_epoch_file, 
                                           epoch_set = epoch_sample)
    
    # log: overall errors, performance for one hyperparameter set-up
    with open(log_err_file, "a") as text_file:
        text_file.write( "%f %f %f %s %s \n"%(best_num_dense, 
                                              best_keep_prob, 
                                              best_l2, 
                                              str(error_epoch_log[0]), 
                                              str(epoch_time)))
                
    # ------ testing
    
    print('\n\n ----- testing ------ \n')
    
    yh, rmse, mae, mape = test_nn(epoch_samples = epoch_sample, 
                                  x_test = xval, 
                                  y_test = yval, 
                                  model_path = path_model, 
                                  method_str = method_str,
                                  dataset_str = dataset_str)
    
    print('\n\n testing errors: ', rmse, mae, mape)
    
    with open(log_err_file, "a") as text_file:
        log_test(text_file, [rmse, mae, mape])
    
