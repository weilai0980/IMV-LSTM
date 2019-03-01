import sys

import collections
import hashlib
import numbers

from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops.rnn_cell_impl import *

# local 
from utils_libs import *
from mv_rnn_cell import *
from ts_mv_rnn import *


# ---- mv dense layers ---- 

def multi_mv_dense(num_layers, keep_prob, h_vari, dim_vari, scope, num_vari, \
                   activation_type, max_norm_regul, regul_type):
    
    in_dim_vari = dim_vari
    out_dim_vari = int(dim_vari/2)
    h_mv_input = h_vari
    
    reg_mv_dense = 0.0
    
    for i in range(num_layers):
        
        with tf.variable_scope(scope+str(i)):
            
            # ? dropout
            h_mv_input = tf.nn.dropout(h_mv_input, keep_prob)
            # h_mv [V B d]
            # ? max norm constrains
            h_mv_input, tmp_regu_dense = mv_dense(h_mv_input, 
                                                  in_dim_vari,
                                                  scope + str(i),
                                                  num_vari, 
                                                  out_dim_vari,
                                                  activation_type, 
                                                  max_norm_regul, 
                                                  regul_type)
            
            reg_mv_dense += tmp_regu_dense
            
            in_dim_vari  = out_dim_vari
            out_dim_vari = int(out_dim_vari/2)
            
    return h_mv_input, reg_mv_dense, in_dim_vari
            

# with max-norm regularization 
def mv_dense(h_vari, dim_vari, scope, num_vari, dim_to, activation_type, max_norm_regul, regul_type):
    
    # argu [V B D]
    
    with tf.variable_scope(scope):
        
        # [V 1 D d]
        w = tf.get_variable('w', [num_vari, 1, dim_vari, dim_to], initializer = tf.contrib.layers.xavier_initializer())
        # [V 1 1 d]
        b = tf.Variable(tf.random_normal([num_vari, 1, 1, dim_to]))
        
        # [V B D 1]
        h_expand = tf.expand_dims(h_vari, -1)
        
        # max-norm regularization 
        if max_norm_regul > 0:
            
            clipped = tf.clip_by_norm(w, clip_norm = max_norm_regul, axes = 2)
            clip_w = tf.assign(w, clipped)
            
            tmp_h = tf.reduce_sum(h_expand*clip_w+b, 2)
            regularization_term = clip_w
            
        else:
            tmp_h = tf.reduce_sum(h_expand*w+b, 2)
            regularization_term = w
            
        # [V B D 1] * [V 1 D d] -> [V B d]
        # ?
        if activation_type == "":
            h = tmp_h
        elif activation_type == "relu":
            h = tf.nn.relu(tmp_h) 
        else:
            print("\n [ERROR] activation in mv_dense \n")
        
        # regularization type
        if regul_type == 'l2':
            return h, tf.nn.l2_loss(regularization_term) 
        
        elif regul_type == 'l1':
            return h, tf.reduce_sum(tf.abs(regularization_term)) 
        
        else:
            return '[ERROR] regularization type'

    
def multi_dense(x, x_dim, num_layers, scope, dropout_keep_prob, max_norm_regul):
    
        in_dim = x_dim
        out_dim = int(in_dim/2)
        
        h = x
        regularization = 0.0
        
        for i in range(num_layers):
            
            with tf.variable_scope(scope+str(i)):
                
                # dropout
                h = tf.nn.dropout(h, dropout_keep_prob)
                
                w = tf.get_variable('w', [ in_dim, out_dim ], dtype=tf.float32,\
                                    initializer = tf.contrib.layers.variance_scaling_initializer())
                                    #initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.zeros( out_dim ))
                
                # max norm constraints 
                if max_norm_regul > 0:
                    
                    clipped = tf.clip_by_norm(w, clip_norm = max_norm_regul, axes = 1)
                    clip_w = tf.assign(w, clipped)
                    
                    h = tf.nn.relu( tf.matmul(h, clip_w) + b )
                    regularization += tf.nn.l2_loss(clip_w)
                    
                else:
                    h = tf.nn.relu( tf.matmul(h, w) + b )
                    regularization += tf.nn.l2_loss(w)
                
                #?
                #regularization += tf.nn.l2_loss(w)
                # regularization += tf.reduce_sum(tf.abs(w))
                
                in_dim = out_dim
                out_dim = int(out_dim/2)
                
        return h, regularization, in_dim    
    
def dense(x, x_dim, out_dim, scope, dropout_keep_prob, max_norm_regul, activation_type):
    
    h = x
    regularization = 0.0
    
    with tf.variable_scope(scope):
        
        # dropout
        h = tf.nn.dropout(h, dropout_keep_prob)
        
        
        w = tf.get_variable('w', 
                            [x_dim, out_dim], 
                            dtype = tf.float32,\
                            initializer = tf.contrib.layers.xavier_initializer())
                                    #variance_scaling_initializer())
                                    #initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(tf.zeros(out_dim))
                
        # max norm constraints 
        if max_norm_regul > 0:
            clipped = tf.clip_by_norm(w, clip_norm = max_norm_regul, axes = 1)
            clip_w = tf.assign(w, clipped)
                    
            tmp_h = tf.matmul(h, clip_w) + b
            regularization = tf.nn.l2_loss(clip_w)
                    
        else:
            tmp_h = tf.matmul(h, w) + b
            regularization = tf.nn.l2_loss(w)
        
        
        if activation_type == "relu":
            h = tf.nn.relu(tmp_h)
            
        elif activation_type == "":
            h = tmp_h
        
        else:
            print("\n [ERROR] activation in dense \n")

        #?
        #regularization = tf.nn.l2_loss(w)
                
    return h, regularization


