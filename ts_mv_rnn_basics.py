import sys

import collections
import hashlib
import numbers

from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops.rnn_cell_impl import *

# local 
from custom_rnn_cell import *
from utils_libs import *


# ---- mv dense layers ---- 

def multi_mv_dense( num_layers, keep_prob, h_vari, dim_vari, scope, num_vari, \
                   bool_no_activation, max_norm_regul, regul_type ):
    
    in_dim_vari = dim_vari
    out_dim_vari = int(dim_vari/2)
    h_mv_input = h_vari
    
    reg_mv_dense = 0.0
    
    for i in range(num_layers):
        
        with tf.variable_scope(scope+str(i)):
            
            # ? dropout
            h_mv_input = tf.nn.dropout(h_mv_input, tf.gather(keep_prob, 0))
            # h_mv [V B d]
            # ? max norm constrains
            h_mv_input, tmp_regu_dense = mv_dense(h_mv_input, \
                                                  in_dim_vari,\
                                                  scope+str(i),\
                                                  num_vari, \
                                                  out_dim_vari,\
                                                  False, max_norm_regul, regul_type)
            
            reg_mv_dense += tmp_regu_dense
            
            in_dim_vari  = out_dim_vari
            out_dim_vari = int(out_dim_vari/2)
            
    return h_mv_input, reg_mv_dense, in_dim_vari
            

# with max-norm regularization 
def mv_dense( h_vari, dim_vari, scope, num_vari, dim_to, bool_no_activation, max_norm_regul, regul_type ):
    
    # argu [V B D]
    
    with tf.variable_scope(scope):
        
        # [V 1 D d]
        w = tf.get_variable('w', [ num_vari, 1, dim_vari, dim_to ], initializer=tf.contrib.layers.xavier_initializer())
        # [V 1 1 d]
        b = tf.Variable( tf.random_normal([ num_vari, 1, 1, dim_to ]) )
        
        # [V B D 1]
        h_expand = tf.expand_dims(h_vari, -1)
        
        if max_norm_regul > 0:
            clipped = tf.clip_by_norm(w, clip_norm = max_norm_regul, axes = 2)
            clip_w = tf.assign(w, clipped)
            
            tmp_h =  tf.reduce_sum(h_expand * clip_w + b, 2)
            
        else:
            tmp_h =  tf.reduce_sum(h_expand * w + b, 2)
            
        # [V B D 1] * [V 1 D d] -> [V B d]
        # ?
        if bool_no_activation == True:
            h = tmp_h
        else:
            h = tf.nn.relu( tmp_h ) 
            
        if regul_type == 'l2':
            return h, tf.nn.l2_loss(w) 
        
        elif regul_type == 'l1':
            return h, tf.reduce_sum( tf.abs(w) ) 
        
        else:
            return '[ERROR] regularization type'

# with max-norm regularization 
def mv_dense_share( h_vari, dim_vari, scope, num_vari, dim_to, bool_no_activation, max_norm_regul, regul_type ):
    
    # argu [V B D]
    
    with tf.variable_scope(scope):
        
        # [D d]
        w = tf.get_variable('w', [ dim_vari, dim_to ], initializer=tf.contrib.layers.xavier_initializer())
        # [ d]
        b = tf.Variable( tf.random_normal([ dim_to ]) )
        
        
        if max_norm_regul > 0:
            clipped = tf.clip_by_norm(w, clip_norm = max_norm_regul, axes = 0)
            clip_w = tf.assign(w, clipped)
            
            # [V B d]
            tmp_h = tf.tensordot(h_vari, w, 1) + b
            #tmp_h = tf.reduce_sum(h_expand * clip_w + b, 2)
            
        else:
            tmp_h = tf.tensordot(h_vari, w, 1) + b
            #tmp_h =  tf.reduce_sum(h_expand * w + b, 2)
            
        # [V B D 1] * [V 1 D d] -> [V B d]
        # ?
        if bool_no_activation == True:
            h = tmp_h
        else:
            h = tf.nn.relu( tmp_h ) 
            
        if regul_type == 'l2':
            return h, tf.nn.l2_loss(w) 
        
        elif regul_type == 'l1':
            return h, tf.reduce_sum( tf.abs(w) ) 
        
        else:
            return '[ERROR] regularization type'



# ---- residual and plain dense layers ----  
    
def res_lstm(x, hidden_dim, n_layers, scope, dropout_keep_prob):
    
    #dropout
    #x = tf.nn.dropout(x, dropout_keep_prob)
    
    with tf.variable_scope(scope):
            #Deep lstm: residual or highway connections 
            lstm_cell = tf.nn.rnn_cell.LSTMCell(hidden_dim, \
                                                initializer= tf.contrib.keras.initializers.glorot_normal())
            hiddens, state = tf.nn.dynamic_rnn(cell = lstm_cell, inputs = x, dtype = tf.float32)
            
    for i in range(1, n_layers):
        
        with tf.variable_scope(scope+str(i)):
            
            tmp_h = hiddens
            
            lstm_cell = tf.nn.rnn_cell.LSTMCell(hidden_dim, \
                                                    initializer= tf.contrib.keras.initializers.glorot_normal())
            hiddens, state = tf.nn.dynamic_rnn(cell = lstm_cell, inputs = hiddens, dtype = tf.float32)
            hiddens = hiddens + tmp_h 
             
    return hiddens, state

def plain_lstm(x, dim_layers, scope, dropout_keep_prob):
    
    #dropout
    #x = tf.nn.dropout(x, dropout_keep_prob)
    
    with tf.variable_scope(scope):
        
            tmp_cell = tf.nn.rnn_cell.LSTMCell(dim_layers[0], \
                                               initializer= tf.contrib.keras.initializers.glorot_normal())
            
            # dropout
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(tmp_cell, state_keep_prob = tf.gather(dropout_keep_prob, 0))
            
            hiddens, state = tf.nn.dynamic_rnn(cell = lstm_cell, inputs = x, dtype = tf.float32)
        
        
    for i in range(1,len(dim_layers)):
        with tf.variable_scope(scope+str(i)):
            tmp_cell = tf.nn.rnn_cell.LSTMCell(dim_layers[i], \
                                                    initializer= tf.contrib.keras.initializers.glorot_normal())
            
            # dropout
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(tmp_cell, state_keep_prob = tf.gather(dropout_keep_prob, 0))
            
            hiddens, state = tf.nn.dynamic_rnn(cell = lstm_cell, inputs = hiddens, dtype = tf.float32)
                
    return hiddens, state 

    
def res_dense(x, x_dim, hidden_dim, n_layers, scope, dropout_keep_prob):
    
        #dropout
        x = tf.nn.dropout(x, dropout_keep_prob)
        
        with tf.variable_scope(scope):
                # initilization
                w = tf.get_variable('w', [x_dim, hidden_dim], dtype = tf.float32,
                                         initializer = tf.contrib.layers.variance_scaling_initializer())
                                         #initializer = tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.zeros([hidden_dim]))
                h = tf.nn.relu(tf.matmul(x, w) + b )

                regularization = tf.nn.l2_loss(w)
        #dropout
        #h = tf.nn.dropout(h, dropout_keep_prob)
        
        for i in range(1, n_layers):
            
            with tf.variable_scope(scope+str(i)):
                w = tf.get_variable('w', [hidden_dim, hidden_dim], \
                                    initializer = tf.contrib.layers.variance_scaling_initializer())
                                    #initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.zeros( hidden_dim ))
                
                # residual connection
                tmp_h = h
                h = tf.nn.relu( tf.matmul(h, w) + b )
                h = tmp_h + h
                
                regularization += tf.nn.l2_loss(w)
        
        return h, regularization
    
def plain_dense(x, x_dim, dim_layers, scope, dropout_keep_prob, max_norm_regul):
    
        #dropout
        x = tf.nn.dropout(x, dropout_keep_prob)
        
        with tf.variable_scope(scope):
                # initilization
                w = tf.get_variable('w', [x_dim, dim_layers[0]], dtype=tf.float32,\
                                    initializer = tf.contrib.layers.variance_scaling_initializer())
                                    #initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.zeros([dim_layers[0]]))
                
                # max norm constraints
                if max_norm_regul > 0:
                    
                    clipped = tf.clip_by_norm(w, clip_norm = max_norm_regul, axes = 1)
                    clip_w = tf.assign(w, clipped)
                    
                    h = tf.nn.relu( tf.matmul(x, clip_w) + b )
                    
                else:
                    h = tf.nn.relu( tf.matmul(x, w) + b )
                    
                #?
                regularization = tf.nn.l2_loss(w)
                #regularization = tf.reduce_sum(tf.abs(w))
                
        #dropout
        h = tf.nn.dropout(h, dropout_keep_prob)
        
        for i in range(1, len(dim_layers)):
            
            with tf.variable_scope(scope+str(i)):
                w = tf.get_variable('w', [dim_layers[i-1], dim_layers[i]], dtype=tf.float32,\
                                    initializer = tf.contrib.layers.variance_scaling_initializer())
                                    #initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.zeros( dim_layers[i] ))
                
                # max norm constraints 
                if max_norm_regul > 0:
                    
                    clipped = tf.clip_by_norm(w, clip_norm = max_norm_regul, axes = 1)
                    clip_w = tf.assign(w, clipped)
                    
                    h = tf.nn.relu( tf.matmul(h, clip_w) + b )
                    
                else:
                    h = tf.nn.relu( tf.matmul(h, w) + b )
                
                #?
                regularization += tf.nn.l2_loss(w)
                #regularization += tf.reduce_sum(tf.abs(w))
                
        return h, regularization

def plain_dense_leaky(x, x_dim, dim_layers, scope, dropout_keep_prob, alpha):
    
        #dropout
        x = tf.nn.dropout(x, dropout_keep_prob)
        
        with tf.variable_scope(scope):
                # initilization
                w = tf.get_variable('w', [x_dim, dim_layers[0]], dtype=tf.float32,\
                                    initializer = tf.contrib.layers.variance_scaling_initializer())
                                    #initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.zeros([dim_layers[0]]))
                
                # ?
                tmp_h = tf.matmul(x, w) + b 
                h = tf.maximum( alpha*tmp_h, tmp_h )

                #?
                regularization = tf.nn.l2_loss(w)
                #regularization = tf.reduce_sum(tf.abs(w))
                
        #dropout
        #h = tf.nn.dropout(h, dropout_keep_prob)
        
        for i in range(1, len(dim_layers)):
            
            with tf.variable_scope(scope+str(i)):
                w = tf.get_variable('w', [dim_layers[i-1], dim_layers[i]], dtype=tf.float32,\
                                    initializer = tf.contrib.layers.variance_scaling_initializer())
                                    #initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.zeros(dim_layers[i]))
                
                # ?
                tmp_h = tf.matmul(h, w) + b 
                h = tf.maximum( alpha*tmp_h, tmp_h )
                
                #?
                regularization += tf.nn.l2_loss(w)
                #regularization += tf.reduce_sum(tf.abs(w))
                
        return h, regularization
    
    
def multi_dense(x, x_dim, num_layers, scope, dropout_keep_prob, max_norm_regul):
    
        in_dim = x_dim
        out_dim = int(in_dim/2)
        
        h = x
        regularization = 0.0
        
        for i in range(num_layers):
            
            with tf.variable_scope(scope+str(i)):
                
                #dropout
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
                    
                else:
                    h = tf.nn.relu( tf.matmul(h, w) + b )
                
                #?
                regularization += tf.nn.l2_loss(w)
                #regularization += tf.reduce_sum(tf.abs(w))
                
                in_dim = out_dim
                out_dim = int(out_dim/2)
                
        return h, regularization, in_dim    