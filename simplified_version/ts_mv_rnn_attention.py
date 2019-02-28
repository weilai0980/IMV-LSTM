import sys

import collections
import hashlib
import numbers

from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops.rnn_cell_impl import *

# local 
from mv_rnn_cell import *
from utils_libs import *


# ---- Attention for MV-RNN ----

def mv_pooling_temp( h_var, pool_type, step ):
    
    # [V B T D]
    # [V B T-1 D], [V B 1 D]
    tmph_before, tmph_last = tf.split(h_var, [step-1, 1], 2)
    
    if pool_type == 'max':
        tmph_before_reduce = tf.reduce_max(tmph_before, 2)
    elif pool_type == 'average':
        tmph_before_reduce = tf.reduce_mean(tmph_before, 2)
    else:
        print('\n [ERROR] pooling type \n')
            
    #[V B D]
    tmph_last_reduce = tf.squeeze(tmph_last, 2)
    
    #[V B 2D]
    return tf.concat([tmph_before_reduce, tmph_last_reduce], -1)


# variate attention based on the temporal weighted hiddens
def mv_attention_variate( h_temp, var_dim, scope, num_vari, att_type ):
    
    # [V B D]
    with tf.variable_scope(scope):
        
        # on all including the target and independent variables
        if att_type == 'vari_loc_all':
            
            # input [V B D]
            w_var = tf.get_variable('w_var', [var_dim, 1], initializer=tf.contrib.layers.xavier_initializer())
            b_var = tf.Variable(tf.random_normal([1]))
            
            #? bias nonlinear activation ?
            '''
            # [V B 1] = [V B D]*[D 1]
            logits = tf.nn.tanh(tf.transpose(tf.tensordot(h_temp, w_var, axes=1) + b_var, [1, 0, 2]))
            '''
            
            # [1 1 D]
            augment_w_var = tf.expand_dims(tf.transpose(w_var, [1, 0]), 0)
            # [B V 1] <- [V B 1] = [V B D]*[1 1 D]
            logits = tf.nn.tanh( tf.transpose(tf.reduce_sum(h_temp*augment_w_var, 2, keepdims = True) + b_var, [1, 0, 2]) ) 
            
            # [B V 1]
            var_weight = tf.nn.softmax(logits, dim = 1)
            
            # [B V D]
            h_trans = tf.transpose(h_temp, [1, 0, 2])
            h_weighted = tf.reduce_sum(h_trans*var_weight, 1)
            
            # [B D]
            h_res = h_weighted
            
        # on all including the target and independent variables
        elif att_type == 'vari_mlp_all':
            
            # input [V B D]
            w_var = tf.get_variable('w_var', [var_dim, 1], initializer=tf.contrib.layers.xavier_initializer())
            b_var = tf.Variable(tf.random_normal([1]))
            
            w_mul = tf.get_variable('w_mul', [1, num_vari, 1], initializer=tf.contrib.layers.xavier_initializer())
            
            #? bias nonlinear activation ?
            # [B V 1] <- [V B 1] = [V B D]*[D 1]
            logits = w_mul * tf.nn.tanh(tf.transpose(tf.tensordot(h_temp, w_var, axes=1) + b_var, [1, 0, 2]))
            
            # [B V 1]
            var_weight = tf.nn.softmax(logits, dim = 1)
            
            # [B V D]
            h_trans = tf.transpose(h_temp, [1, 0, 2])
            h_weighted = tf.reduce_sum(h_trans*var_weight, 1)
            
            # [B D]
            h_res = h_weighted
        
        else:
            print('\n [ERROR] variable attention type \n')
        
    return h_res, tf.nn.l2_loss(w_var), var_weight, logits



# unified temporal attention 
def mv_attention_temp(h_list, v_dim, scope, n_step, att_type, num_vari):
    
    with tf.variable_scope(scope):
        
        #[V B T D]
        tmph = tf.stack(h_list, 0)
        
        # [V B T-1 D], [V, B, 1, D]
        tmph_before, tmph_last = tf.split(tmph, [n_step - 1, 1], 2)
        
        # -- temporal logits
        if att_type == 'temp_loc':
            
            w_temp = tf.get_variable('w_temp', 
                                     [num_vari, 1, 1, v_dim], 
                                     initializer=tf.contrib.layers.xavier_initializer())
            # ?
            b_temp = tf.Variable(tf.zeros([num_vari, 1, 1]))
            
            # ? bias nonlinear activation ?
            #[V, B, T-1]
            # tf.nn.tanh
            
            temp_logit = tf.nn.tanh(tf.reduce_sum(tmph_before * w_temp, 3) + b_temp)
            
            #temp_logit = tf.reduce_sum(tmph_before * w_temp, 3)
            
            # empty and relu activation 
            # ? use relu if with decay ?
            
            regul = tf.nn.l2_loss(w_temp)
            
        else:
            print('\n [ERROR] temporal attention type \n')
        
        
        # no attention decay
        temp_weight = tf.nn.softmax( temp_logit )
            
        # temp_before [V B T-1 D], temp_weight [V B T-1]
        tmph_cxt = tf.reduce_sum(tmph_before*tf.expand_dims(temp_weight, -1), 2)
        tmph_last = tf.squeeze(tmph_last, [2]) 
            
        # [V B 2D]
        h_temp = tf.concat([tmph_last, tmph_cxt], 2)
        # ?
        #h_temp = tmph_last
            
        return h_temp, regul, temp_weight
