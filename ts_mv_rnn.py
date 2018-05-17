#!/usr/bin/python

import sys

import collections
import hashlib
import numbers

from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops.rnn_cell_impl import *

# local 
from mv_rnn_cell import *
from utils_libs import *
from ts_mv_rnn_attention import *
from ts_mv_rnn_basics import *

# ---- plain RNN ----

class tsLSTM_plain():
    
    def __init__(self, n_dense_dim_layers, n_lstm_dim_layers, n_steps, n_data_dim, session,\
                 lr, l2_dense, max_norm , n_batch_size, bool_residual, att_type, l2_att):
        
        self.LEARNING_RATE = lr
        self.L2 =  l2
        
        self.n_lstm_dim_layers = n_lstm_dim_layers
        
        self.N_STEPS    = n_steps
        self.N_DATA_DIM = n_data_dim
        
        self.n_dense_dim_layers = n_dense_dim_layers
        self.n_batch_size       = n_batch_size
        
        self.att_type = att_type
        
        # placeholders
        self.x = tf.placeholder(tf.float32, [None, self.N_STEPS, self.N_DATA_DIM])
        self.y = tf.placeholder(tf.float32, [None, 1])
        self.keep_prob = tf.placeholder(tf.float32, [None])
        
        # begin to build the graph
        self.sess = session
        
        h, _ = plain_lstm( self.x, n_lstm_dim_layers, 'lstm', self.keep_prob )
        
        
        if att_type == 'temp':
            
            print ' --- Plain RNN using temporal attention:  '
                
            h, self.att, regu_att = attention_temp_logit( h, n_lstm_dim_layers[-1], 'att', self.N_STEPS )
            
            # dropout
            h, regu_dense = plain_dense( h, n_lstm_dim_layers[-1]*2, n_dense_dim_layers, 'dense', \
                                        tf.gather(self.keep_prob, 0), max_norm )
            #?
            self.regularization = l2_dense*regu_dense + l2_att*regu_att
            
        else:
            
            print ' --- Plain RNN using no attention:  '
            
            # obtain the last hidden state
            tmp_hiddens = tf.transpose( h, [1,0,2] )
            h = tmp_hiddens[-1]
            
            # dropout
            h, regu_dense = plain_dense( h, n_lstm_dim_layers[-1], n_dense_dim_layers, 'dense', \
                                        tf.gather(self.keep_prob, 0), max_norm )
            
            #?
            self.regularization = l2_dense*regu_dense
            
            
        #dropout
        #h = tf.nn.dropout(h, tf.gather(self.keep_prob, 1))
        
        with tf.variable_scope("output"):
            
            w = tf.get_variable('w', shape=[n_dense_dim_layers[-1], 1],\
                                     initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.zeros( [ 1 ] ))
            
            self.py = tf.matmul(h, w) + b
            
            # regularization
            # ?
            self.regularization += l2_dense*tf.nn.l2_loss(w)
            
            
    def train_ini(self):
        
        # loss function 
        self.error_mse = tf.reduce_mean( tf.square(self.y - self.py) )
        self.error_sqsum  = tf.reduce_sum( tf.square(self.y - self.py) )
        
        # ?
        self.loss = self.error_mse + self.regularization
        
        # optimizer 
        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.LEARNING_RATE).minimize(self.loss)  
        
        # initilization 
        self.init = tf.global_variables_initializer()
        self.sess.run( [self.init] )
        
    def train_batch(self, x_batch, y_batch, keep_prob ):
        
        _, c, err_sum = self.sess.run([self.optimizer, self.loss, self.error_sqsum ],\
                                      feed_dict = {self.x:x_batch, self.y:y_batch, self.keep_prob:keep_prob })
        return c, err_sum

#   initialize inference         
    def inference_ini(self):

        # error metric
        self.rmse = tf.sqrt( tf.reduce_mean(tf.square(self.y - self.py)) )
        self.mae =  tf.reduce_mean( tf.abs(self.y - self.py) )
        self.mape = tf.reduce_mean( tf.abs((self.y - self.py)*1.0/(self.y+1e-5)) )

#   infer givn testing data
    def inference(self, x_test, y_test, keep_prob):
        
        if self.att_type == '':
            return self.sess.run([self.py, self.rmse, self.mae, self.mape], \
                             feed_dict = {self.x:x_test, self.y:y_test, self.keep_prob:keep_prob})
        
        else:
            return self.sess.run([self.att, self.py, self.rmse, self.mae, self.mape], \
                             feed_dict = {self.x:x_test, self.y:y_test, self.keep_prob:keep_prob})
    
    def predict(self, x_test, y_test, keep_prob):
        return self.sess.run([self.py], feed_dict = {self.x:x_test, self.y:y_test, self.keep_prob:keep_prob})
    
    def test_attention(self, x_test, y_test, keep_prob):
        return self.sess.run( self.att,  feed_dict = {self.x:x_test, self.y:y_test, self.keep_prob:keep_prob})
    
    
# ---- separate RNN ----

class tsLSTM_seperate():
    
    def __init__(self, n_dense_dim_layers, n_lstm_dim_layers, n_steps, n_data_dim, session,\
                 lr, l2_dense, max_norm , n_batch_size, bool_residual, att_type, temp_attention_type, vari_attention_type,\
                 dense_regul_type, l2_att):
        
        self.LEARNING_RATE = lr
        self.L2 =  l2_dense
        
        self.n_lstm_dim_layers = n_lstm_dim_layers
        
        self.N_STEPS    = n_steps
        self.N_DATA_DIM = n_data_dim
        
        self.x = tf.placeholder(tf.float32, [None, self.N_STEPS, self.N_DATA_DIM])
        self.y = tf.placeholder(tf.float32, [None, 1])
        
        self.keep_prob = tf.placeholder(tf.float32)
        
        self.n_dense_dim_layers = n_dense_dim_layers
        self.n_batch_size       = n_batch_size
        
        self.att_type = att_type
        
        self.sess = session
        
        indivi_ts = tf.split(self.x, num_or_size_splits = self.N_DATA_DIM, axis = 2)
        concat_h  = []
        
        for i in range( self.N_DATA_DIM ):
            
            current_x = indivi_ts[i]
            
            if bool_residual == True:
                h, _  = res_lstm( current_x, n_lstm_dim_layers[0], len(n_lstm_dim_layers), 'lstm'+str(i), self.keep_prob)
            else:
                h, _  = plain_lstm( current_x, n_lstm_dim_layers, 'lstm'+str(i), self.keep_prob)
                
            if att_type != '':
                
                concat_h.append(h)
            
            else:
                # obtain the last hidden state    
                tmp_hiddens = tf.transpose( h, [1,0,2]  )
                h = tmp_hiddens[-1]
                
                # [V B T D]
                concat_h.append(h)
        
        
        # no attention
        if att_type == '': 
            
            # hidden space merge
            h = tf.concat(concat_h, 1)
            h, regul = plain_dense(h, n_lstm_dim_layers[-1]*self.N_DATA_DIM, n_dense_dim_layers, 'dense', self.keep_prob)
            
            self.regularization = regul
            
            #dropout
            #h = tf.nn.dropout(h, self.keep_prob)
            with tf.variable_scope("output"):
                w = tf.get_variable('w', shape=[n_dense_dim_layers[-1], 1],\
                                     initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.zeros([ 1 ]))
            
                self.py = tf.matmul(h, w) + b
                self.regularization += tf.nn.l2_loss(w)
            
        
        elif att_type == 'temp':
            
            h, att_regu, self.att = sep_attention_temp_logit( concat_h, n_lstm_dim_layers[-1], 'attention', self.N_STEPS )
            
            # dense layers 
            h, regul = plain_dense(h, n_lstm_dim_layers[-1]*self.N_DATA_DIM*2, n_dense_dim_layers, 'dense', self.keep_prob)
            self.regularization = regul + att_regu
            
            #dropout
            #h = tf.nn.dropout(h, self.keep_prob)
            with tf.variable_scope("output"):
                w = tf.get_variable('w', shape=[n_dense_dim_layers[-1], 1],\
                                     initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.zeros([ 1 ]))
            
                self.py = tf.matmul(h, w) + b
                self.regularization += tf.nn.l2_loss(w)
            
        elif att_type == 'both-att':
            
            # [V B T D]
            h_list = concat_h 
            #tf.split(h, [int(n_lstm_dim_layers[0]/self.N_DATA_DIM)]*self.N_DATA_DIM, 2)
                
                
            # --- temporal and variate attention 
                
            # ? dropout
            # h_att_temp_input = tf.nn.dropout(h_list, tf.gather(self.keep_prob,1))
            h_att_temp_input = h_list
                
            # temporal attention
            # h_temp [V B 2D]
            h_temp, regu_att_temp, self.att_temp = sep_attention_temp(h_att_temp_input,\
                                                                      n_lstm_dim_layers[-1],\
                                                                      'att_temp',\
                                                                      self.N_STEPS,\
                                                                      temp_attention_type,\
                                                                      self.N_DATA_DIM)
            # ? dropout
            # h_att_vari_input = tf.nn.dropout(h_temp, tf.gather(self.keep_prob,1))
            h_att_vari_input = h_temp
                
                
            # variable attention 
            _, regu_att_vari, self.att_vari = mv_attention_variate(h_att_vari_input,\
                                                                   2*int(n_lstm_dim_layers[-1]),\
                                                                   'att_vari',\
                                                                   self.N_DATA_DIM,\
                                                                   vari_attention_type)
            self.att = self.att_vari
            #[self.att_temp, self.att_vari]
                
                
            # --- multiple mv-dense layers
                
            # [V B 2D] - [V B d]
            interm_var_dim = int(n_lstm_dim_layers[-1])
                
            # ? dropout
            h_mv_input = tf.nn.dropout(h_temp, tf.gather(self.keep_prob, 0))
            # h_mv [V B d]
            # ? max norm constrains
            h_mv1, regu_mv_dense1 = mv_dense( h_mv_input, 2*int(n_lstm_dim_layers[-1]), 'mv_dense1',\
                                              self.N_DATA_DIM, interm_var_dim, False, max_norm, dense_regul_type )
                
            # ? dropout
            h_mv1 = tf.nn.dropout(h_mv1, tf.gather(self.keep_prob, 1))
            # ? max norm constrains
            h_mv2, regu_mv_dense2 = mv_dense( h_mv1, interm_var_dim, 'mv_dense2', \
                                              self.N_DATA_DIM, interm_var_dim/2, False, max_norm, dense_regul_type )
                
            # ? dropout
            h_mv2 = tf.nn.dropout(h_mv2, tf.gather(self.keep_prob,1))
            # outpout layer without max-norm regularization
            h_mv, regu_mv_dense = mv_dense( h_mv2, interm_var_dim/2, 'mv_dense3', \
                                            self.N_DATA_DIM, 2, True, 0.0, dense_regul_type )
                
            #[V B 1], [V B 1] 
            h_mean, h_var = tf.split(h_mv, [1, 1], 2)
            h_var = tf.square(h_var)
            
            
            # --- mixture prediction 
                
            # [V B d] - [B V d]
            h_mv_trans = tf.transpose(h_mean, [1, 0, 2])
            # [B V d]*[B V 1]
            h_mv_weighted = tf.reduce_sum( h_mv_trans*self.att_vari, 1 )
                
            # [V 1]
            self.py = h_mv_weighted
                    
            # --- regularization
            # ?
            self.regularization = l2_dense*regu_mv_dense + l2_dense*regu_mv_dense1 + l2_dense*regu_mv_dense2 + \
                                  l2_att*(regu_att_temp + regu_att_vari) 
            
    
    def train_ini(self):
        
        # loss function 
        self.error_mse = tf.reduce_mean( tf.square(self.y - self.py) )
        self.error_sqsum  = tf.reduce_sum( tf.square(self.y - self.py) )
        
        # ?
        self.loss = self.error_mse + self.regularization
        
        # optimizer 
        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.LEARNING_RATE).minimize(self.loss)  
        
        # initilization 
        self.init = tf.global_variables_initializer()
        self.sess.run( [self.init] )
        
    def train_batch(self, x_batch, y_batch, keep_prob ):
        
        _, c, err_sum = self.sess.run([self.optimizer, self.loss, self.error_sqsum ],\
                                      feed_dict = {self.x:x_batch, self.y:y_batch, self.keep_prob:keep_prob })
        return c, err_sum

#   initialize inference         
    def inference_ini(self):

        # error metric
        self.rmse = tf.sqrt( tf.reduce_mean(tf.square(self.y - self.py)) )
        self.mae =  tf.reduce_mean( tf.abs(self.y - self.py) )
        self.mape = tf.reduce_mean( tf.abs((self.y - self.py)*1.0/(self.y+1e-5)) )

#   infer givn testing data
    def inference(self, x_test, y_test, keep_prob):
        
        if self.att_type == '':
            return self.sess.run([self.py, self.rmse, self.mae, self.mape], \
                             feed_dict = {self.x:x_test, self.y:y_test, self.keep_prob:keep_prob})
        
        else:
            return self.sess.run([self.att, self.py, self.rmse, self.mae, self.mape], \
                             feed_dict = {self.x:x_test, self.y:y_test, self.keep_prob:keep_prob})


# ---- multi-variate RNN ----
'''
RNN Regularization:

   Batch norm, Layer norm, Batch Normalized Recurrent Neural Networks
   
   DropConnect
    
   Regularization of Deep Neural Networks with Spectral Dropout
   
   dropout on input-hidden, hidden-hidden, hidden-output 

   RNN dropout without memory loss

   max norm regularization
'''

class tsLSTM_mv():
    
    def __init__(self, n_dense_dim_layers, n_lstm_dim_layers, n_steps, n_data_dim, session,\
                 lr, max_norm , n_batch_size, bool_residual, \
                 bool_att, temp_decay, temp_attention_type, \
                 l2_dense, l2_att, vari_attention_type, loss_type, dense_regul_type, layer_norm, num_mv_dense ):
        
        self.LEARNING_RATE = lr
        
        self.n_lstm_dim_layers = n_lstm_dim_layers
        
        self.N_STEPS    = n_steps
        self.N_DATA_DIM = n_data_dim
        
        self.n_dense_dim_layers = n_dense_dim_layers
        self.n_batch_size       = n_batch_size
        
        self.loss_type = loss_type
        self.att_type = bool_att
        
        
        # placeholders
        self.x = tf.placeholder(tf.float32, [None, self.N_STEPS, self.N_DATA_DIM])
        self.y = tf.placeholder(tf.float32, [None, 1])
        #self.keep_prob = tf.placeholder(tf.float32)
        self.keep_prob = tf.placeholder(tf.float32, [None])
        
        steps = tf.constant( range(self.N_STEPS-2, -1, -1), dtype=tf.float32 )
        alpha = tf.constant( 0.2, dtype=tf.float32 )
        
        # begin to build the graph
        self.sess = session
        
        # residual connections
        if bool_residual == True:
            
            print '--- [ERROR] Wrongly use residual connection'
            
        # no residual connection 
        else:
            
            # no attention
            if bool_att == '':
                
                print ' --- MV-RNN using no attention: '
                
                with tf.variable_scope('lstm'):
                    lstm_cell = MvLSTMCell( n_lstm_dim_layers[0], n_var = n_data_dim, \
                                            initializer=tf.contrib.layers.xavier_initializer() )
                    h, state = tf.nn.dynamic_rnn(cell = lstm_cell, inputs = self.x, dtype = tf.float32)
            
                # obtain the last hidden state    
                h_temp = tf.transpose( h, [1,0,2] )
                h_last = h_temp[-1]
                
                h = h_last
                
                self.test = tf.shape(h_last)
                
                # ?
                h, regu_dense = plain_dense( h, n_lstm_dim_layers[-1], n_dense_dim_layers, 'dense', self.keep_prob )
                regu_pre_out = l2_dense*regu_dense
                
                #dropout
                #last_hidden = tf.nn.dropout(last_hidden, self.keep_prob)
                with tf.variable_scope("output"):
                    w = tf.get_variable('w', shape=[n_dense_dim_layers[-1], 1],\
                                     initializer=tf.contrib.layers.xavier_initializer())
                    b = tf.Variable(tf.zeros([1]))
            
                self.py = tf.matmul(h, w) + b
                self.regularization = regu_pre_out + l2_dense*tf.nn.l2_loss(w)
                
            
            elif bool_att == 'temp':
                
                print ' --- MV-RNN using only temporal attention: '

                with tf.variable_scope('lstm'):
                    lstm_cell = MvLSTMCell( n_lstm_dim_layers[0], n_var = n_data_dim ,\
                                            initializer = tf.contrib.layers.xavier_initializer() )
                    h, state = tf.nn.dynamic_rnn(cell = lstm_cell, inputs = self.x, dtype = tf.float32)
                
                # [V B T D]
                h_list = tf.split(h, num_or_size_splits = self.N_DATA_DIM, axis = 2)

                
                # --- apply temporal attention 
                
                # [V B T D] - [V B 2D]
                # shape of h_temp [V B 2D]
                h_temp, regu_att, self.att = mv_attention_temp( h_list, int(n_lstm_dim_layers[-1]/self.N_DATA_DIM),\
                                                       'att', self.N_STEPS, steps, temp_decay, temp_attention )
               
                # test
                self.test = tf.shape(self.att)
                
                # reshape to [B 2H]
                h_tmp = tf.split(h_temp, num_or_size_splits = n_data_dim, axis = 0) 
                h_att = tf.squeeze(tf.concat(h_tmp, 2), [0])
                
                # ---
                
                # ?
                h, regu_dense = plain_dense_leaky( h_att, 2*n_lstm_dim_layers[-1], n_dense_dim_layers, 'dense', \
                                                   self.keep_prob, alpha )
                
                # ?
                if temp_decay == '' or temp_decay == 'cutoff' :
                    regu_pre_out = l2_dense*(regu_dense) + l2_att*regu_att
                else:
                    regu_pre_out = l2_dense*regu_dense + l2_att*(regu_att[0] + regu_att[1]) 
                    
                    
                #dropout
                #last_hidden = tf.nn.dropout(last_hidden, self.keep_prob)
                with tf.variable_scope("output"):
                    w = tf.get_variable('w', shape=[n_dense_dim_layers[-1], 1],\
                                     initializer=tf.contrib.layers.xavier_initializer())
                    b = tf.Variable(tf.zeros([1]))
            
            
                self.py = tf.matmul(h, w) + b
                self.regularization = regu_pre_out + l2_dense*tf.nn.l2_loss(w)
                
            
            elif bool_att == 'vari-direct':
                
                print ' --- MV-RNN using only variate attention with direct output : '
                
                with tf.variable_scope('lstm'):
                    lstm_cell = MvLSTMCell( n_lstm_dim_layers[0], n_var = n_data_dim ,\
                                           initializer = tf.contrib.layers.xavier_initializer() )
                    h, state = tf.nn.dynamic_rnn(cell = lstm_cell, inputs = self.x, dtype = tf.float32)
                
                # [B T D*V] - [T B D*V]
                h_temp = tf.transpose(h, [1, 0, 2])
                # [B D*V]
                h_last = h_temp[-1]
                # [V B D]
                h_last_vari = tf.split(h_last, [int(n_lstm_dim_layers[0]/self.N_DATA_DIM)]*self.N_DATA_DIM, 1)

                
                # --- apply variate attention 
                
                # ? shape h_att [B H] or [B 2D]
                h_att, regu_att_vari, self.att = mv_attention_variate(h_last_vari,\
                                                                      int(n_lstm_dim_layers[0]/self.N_DATA_DIM),\
                                                                      'att_vari', self.N_DATA_DIM, vari_attention)
                self.test = tf.shape(self.att)
                # ---
                
                if vari_attention in [ 'vari_sigmoid' ]:
                    h, regu_dense = plain_dense( h_att, n_lstm_dim_layers[-1], n_dense_dim_layers, 'dense', self.keep_prob )
                
                elif vari_attention in ['vari_softmax_indepen', 'vari_mlp']:
                    h, regu_dense = plain_dense( h_att, n_lstm_dim_layers[-1]/self.N_DATA_DIM*2, n_dense_dim_layers, 'dense',\
                                                self.keep_prob )
                
                elif vari_attention in ['vari_softmax_all']:
                    h, regu_dense = plain_dense( h_att, n_lstm_dim_layers[-1]/self.N_DATA_DIM, n_dense_dim_layers, 'dense',\
                                                self.keep_prob )
                
                else:
                    print '--- [ERROR] variate attention type'
                    
                # ?
                regu_pre_out = l2_dense*regu_dense + l2_att*regu_att_vari
                
                
                #dropout
                #last_hidden = tf.nn.dropout(last_hidden, self.keep_prob)
                with tf.variable_scope("output"):
                    w = tf.get_variable('w', shape=[n_dense_dim_layers[-1], 1],\
                                     initializer=tf.contrib.layers.xavier_initializer())
                    b = tf.Variable(tf.zeros([1]))
            
            
                self.py = tf.matmul(h, w) + b
                self.regularization = regu_pre_out + l2_dense*tf.nn.l2_loss(w)
             
            
            elif bool_att == 'vari-mixture':
                
                print ' --- MV-RNN using only variate attention with mv-dense: '
                
                with tf.variable_scope('lstm'):
                    lstm_cell = MvLSTMCell( n_lstm_dim_layers[0], n_var = n_data_dim ,\
                                           initializer = tf.contrib.layers.xavier_initializer() )
                    h, state = tf.nn.dynamic_rnn(cell = lstm_cell, inputs = self.x, dtype = tf.float32)
                
                # [B T D*V]
                h_temp = tf.transpose(h, [1, 0, 2])
                # [B D*V]
                h_last = h_temp[-1]
                
                #dropout
                h_last = tf.nn.dropout(h_last, self.keep_prob)
                
                # [V B D]
                h_last_vari = tf.split(h_last, [int(n_lstm_dim_layers[0]/self.N_DATA_DIM)]*self.N_DATA_DIM, 1)
                
                # --- intermediate multiple mv-dense layers
                # [V B D] -> [V B d]
                
                interm_var_dim = int(n_lstm_dim_layers[-1]/self.N_DATA_DIM)*2
                
                # h_mv [V B d]
                h_mv1, regu_mv_dense1 = mv_dense( h_last_vari, int(n_lstm_dim_layers[-1]/self.N_DATA_DIM), 'intermediate1',\
                                                self.N_DATA_DIM, interm_var_dim, False, 0.0 )
                
                h_mv, regu_mv_dense = mv_dense( h_mv1, interm_var_dim, 'intermediate',\
                                                self.N_DATA_DIM, 1, True, 0.0 )
                
                # --- derive variate attention 
                
                _, regu_att_vari, self.att = mv_attention_variate(h_last_vari,\
                                                                      int(n_lstm_dim_layers[0]/self.N_DATA_DIM),\
                                                                      'att_vari', self.N_DATA_DIM, vari_attention)
                self.test = tf.shape(self.att)
                
                # ? shape self.att [B V-1 1]
                #self.att, regu_att_vari = mv_attention_variate( h_last_vari, int(n_lstm_dim_layers[-1]/self.N_DATA_DIM),\
                #                                               'att_vari', self.N_DATA_DIM, vari_attention )
                
                # --- mixture output 
                
                # partial attention
                '''
                # [V-1 B d], [1 B d]
                h_mv_indep, h_mv_tar = tf.split(h_mv, [self.N_DATA_DIM-1, 1], 0)
                # [B V-1 d]
                h_mv_indep_trans = tf.transpose(h_mv_indep, [1, 0, 2])
                # [B, 1]
                h_mv_indep_weighted = tf.reduce_sum( h_mv_indep_trans*self.att, 1 )
                
                self.py = h_mv_indep_weighted + tf.squeeze(h_mv_tar, [0]) 
                '''
                
                # full attention
                # [V B d] - [B V d]
                h_mv_trans = tf.transpose(h_mv, [1, 0, 2])
                # [B V d]*[B V 1]
                h_mv_weighted = tf.reduce_sum( h_mv_trans*self.att, 1 )
                
                self.py = h_mv_weighted
                
                # --- regularization
                self.regularization = l2_dense*regu_mv_dense + l2_dense*regu_mv_dense1 + l2_att*regu_att_vari
                
            
            elif bool_att == 'both-att':
                
                print ' --- MV-RNN using both temporal and variate attention in ', bool_att
                
                with tf.variable_scope('lstm'):
                    
                    # ? layer_norm
                    # ? ? ? no-memory-loss dropout: off
                    mv_cell = MvLSTMCell(n_lstm_dim_layers[0], \
                                         n_var = n_data_dim ,\
                                         initializer = tf.contrib.layers.xavier_initializer(),\
                                         memory_update_keep_prob = tf.gather(self.keep_prob, 0),\
                                         layer_norm = layer_norm)
                    
                    # ? dropout for input-hidden, hidden-hidden, hidden-output 
                    # ? variational dropout
                    drop_mv_cell = tf.nn.rnn_cell.DropoutWrapper(mv_cell, \
                                                                 state_keep_prob = tf.gather(self.keep_prob, 0))
                    
                    h, state = tf.nn.dynamic_rnn(cell = drop_mv_cell, inputs = self.x, dtype = tf.float32)
                
                
                # stacked mv-lstm
                for i in range(1, len(n_lstm_dim_layers)):
                    with tf.variable_scope('lstm'+str(i)):
                        
                        mv_cell = MvLSTMCell(n_lstm_dim_layers[i], \
                                             n_var = n_data_dim ,\
                                             initializer = tf.contrib.layers.xavier_initializer(),\
                                             memory_update_keep_prob = tf.gather(self.keep_prob, 0),\
                                             layer_norm = layer_norm)
                    
                        # ? dropout for input-hidden, hidden-hidden, hidden-output 
                        # ? variational dropout
                        drop_mv_cell = tf.nn.rnn_cell.DropoutWrapper(mv_cell, \
                                                                     state_keep_prob = tf.gather(self.keep_prob, 1))
                    
                        h, state = tf.nn.dynamic_rnn(cell = drop_mv_cell, inputs = h, dtype = tf.float32)
                
                
                # [V B T D]
                h_list = tf.split(h, [int(n_lstm_dim_layers[-1]/self.N_DATA_DIM)]*self.N_DATA_DIM, 2)
                
                
                # --- temporal and variate attention 
                
                # ? dropout
                # h_att_temp_input = tf.nn.dropout(h_list, tf.gather(self.keep_prob,1))
                h_att_temp_input = h_list
                
                # temporal attention
                # h_temp [V B 2D]
                h_temp, regu_att_temp, self.att_temp = mv_attention_temp(h_att_temp_input,\
                                                                         int(n_lstm_dim_layers[-1]/self.N_DATA_DIM),\
                                                                         'att_temp', self.N_STEPS, steps, temp_decay,\
                                                                         temp_attention_type, self.N_DATA_DIM)
                # ? dropout
                # h_att_vari_input = tf.nn.dropout(h_temp, tf.gather(self.keep_prob,1))
                h_att_vari_input = h_temp
                
                # variable attention 
                _, regu_att_vari, self.att_vari = mv_attention_variate(h_att_vari_input,\
                                                                       2*int(n_lstm_dim_layers[-1]/self.N_DATA_DIM),\
                                                                       'att_vari', self.N_DATA_DIM, vari_attention_type)
                self.att = self.att_vari
                
                
                # --- multiple mv-dense layers
                # [V B 2D] - [V B d]
                
                # mv-dense layers before the output layer
                h_mv, tmp_regu, dim_vari = multi_mv_dense( num_mv_dense, self.keep_prob, h_temp,\
                                                           2*int(n_lstm_dim_layers[-1]/self.N_DATA_DIM), \
                                                           'mv_dense', self.N_DATA_DIM, \
                                                           False, max_norm, dense_regul_type )
                
                # ? dropout
                h_mv = tf.nn.dropout(h_mv, tf.gather(self.keep_prob,1))
                # outpout layer without max-norm regularization
                h_mv, tmp_regu1 = mv_dense( h_mv, dim_vari, 'output', \
                                            self.N_DATA_DIM, 2, True, 0.0, dense_regul_type )
                
                mv_dense_regu = tmp_regu + tmp_regu1
                
                
                # --- mixture prediction 
                
                #[V B 1], [V B 1] 
                h_mean, h_var = tf.split(h_mv, [1, 1], 2)
                h_var = tf.square(h_var)
                
                # [V B 1] - [B V 1]
                h_mv_trans = tf.transpose(h_mean, [1, 0, 2])
                # [B V 1]*[B V 1]
                h_mv_weighted = tf.reduce_sum( h_mv_trans*self.att_vari, 1 )
                
                # [B 1]
                self.py = h_mv_weighted
                # [B V]
                self.py_indi = tf.squeeze(h_mv_trans, [2])
                    
                # --- negative log likelihood
                
                # [B 1] -> [B V]
                y_tile = tf.tile(self.y, [1, self.N_DATA_DIM])
                
                # [B V]
                tmp_mean = tf.transpose(tf.squeeze(h_mean, [2]), [1,0]) 
                tmp_var  = tf.transpose(tf.squeeze(h_var,  [2]), [1,0])
                
                # [B V]
                tmp_llk = tf.exp(-0.5*tf.square(y_tile - tmp_mean)/(tmp_var + 1e-5))/(2.0*np.pi*(tmp_var + 1e-5))**0.5
                
                llk = tf.multiply( tmp_llk, tf.squeeze(self.att_vari, [2]) ) 
                self.neg_logllk = tf.reduce_sum( -1.0*tf.log(tf.reduce_sum(llk, 1)+1e-5) )
                
                # --- regularization
                # ?
                self.regularization = l2_dense*mv_dense_regu + l2_att*(regu_att_temp + regu_att_vari) 
                
        
            elif bool_att == 'both-fusion':
                
                print ' --- MV-RNN using both temporal and variate attention in ', bool_att
                
                with tf.variable_scope('lstm'):
                    
                    # ? layer_norm
                    # ? ? ? no-memory-loss dropout: off
                    mv_cell = MvLSTMCell(n_lstm_dim_layers[0], n_var = n_data_dim ,\
                                         initializer = tf.contrib.layers.xavier_initializer(),\
                                         memory_update_keep_prob = tf.gather(self.keep_prob, 0),\
                                         layer_norm = layer_norm)
                    
                    # ? dropout for input-hidden, hidden-hidden, hidden-output 
                    # ? variational dropout
                    drop_mv_cell = tf.nn.rnn_cell.DropoutWrapper(mv_cell, state_keep_prob = tf.gather(self.keep_prob, 0))
                    
                    h, state = tf.nn.dynamic_rnn(cell = drop_mv_cell, inputs = self.x, dtype = tf.float32)
                
                
                # [V B T D]
                h_list = tf.split(h, [int(n_lstm_dim_layers[0]/self.N_DATA_DIM)]*self.N_DATA_DIM, 2)
                
                
                # --- temporal and variate attention 
                
                # ? dropout
                # h_att_temp_input = tf.nn.dropout(h_list, tf.gather(self.keep_prob,1))
                h_att_temp_input = h_list
                
                # temporal attention
                # h_temp [V B 2D]
                h_temp, regu_att_temp, self.att_temp = mv_attention_temp(h_att_temp_input,\
                                                                         int(n_lstm_dim_layers[0]/self.N_DATA_DIM),\
                                                                         'att_temp', self.N_STEPS, steps, temp_decay,\
                                                                         temp_attention_type, self.N_DATA_DIM)
                
                # ? dropout
                # h_att_vari_input = tf.nn.dropout(h_temp, tf.gather(self.keep_prob,1))
                h_att_vari_input = h_temp
                
                # variable attention 
                # [B 2D]
                h_vari, regu_att_vari, self.att_vari = mv_attention_variate(h_att_vari_input,\
                                                                       2*int(n_lstm_dim_layers[0]/self.N_DATA_DIM),\
                                                                       'att_vari', self.N_DATA_DIM, vari_attention_type)
                self.att = self.att_vari
                
                
                # --- plain dense layers
                # [B 2D]
                
                # dropout
                h, regu_dense, out_dim = multi_dense(h_vari, 2*int(n_lstm_dim_layers[-1]/self.N_DATA_DIM), num_mv_dense, \
                                            'dense', tf.gather(self.keep_prob, 0), max_norm)
                
                # ? dropout
                h = tf.nn.dropout(h, tf.gather(self.keep_prob,1))
                
                with tf.variable_scope("output"):
                    w = tf.get_variable('w', shape=[out_dim, 1],\
                                        initializer=tf.contrib.layers.xavier_initializer())
                    b = tf.Variable(tf.zeros( [ 1 ] ))
            
                    self.py = tf.matmul(h, w) + b
                    regu_dense += tf.nn.l2_loss(w)
                
                # --- regularization
                # ?
                self.regularization = l2_dense*regu_dense + l2_att*(regu_att_temp + regu_att_vari) 
           
        
        # regularization in LSTM 
        self.regul_lstm = sum( tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables() \
                                   if ("lstm" in tf_var.name and "input" in tf_var.name))
        # ?
        #self.regularization += self.regul_lstm

        
    def train_ini(self):  
        
        tmp_sq_diff = tf.square(self.y - self.py) 
        self.error_sqsum = tf.reduce_sum( tmp_sq_diff )
        self.error_mse = tf.reduce_mean( tmp_sq_diff )
        
        # loss function 
        if self.loss_type == 'mse':
            
            # ? 
            self.loss = self.error_mse + self.regularization
            
        elif self.loss_type == 'lk':
            
            # ? 
            self.loss = self.neg_logllk + self.regularization
            
        else:
            print '--- [ERROR] loss type'
            return
        
        
        self.optimizer = \
        tf.train.AdamOptimizer(learning_rate = self.LEARNING_RATE).minimize(self.loss)  
        
        self.init = tf.global_variables_initializer()
        self.sess.run( [self.init] )
        
    def train_update_optimizer(lr):
        self.optimizer = tf.train.AdamOptimizer(learning_rate = lr).minimize(self.loss) 
        
    def train_batch(self, x_batch, y_batch, keep_prob ):
        
        _, tmp_loss, tmp_sqsum = self.sess.run([self.optimizer, self.loss, self.error_sqsum],\
                              feed_dict={self.x:x_batch, self.y:y_batch, self.keep_prob:keep_prob })
        
        return tmp_loss, tmp_sqsum

#   initialize inference         
    def inference_ini(self):
        
        # error metric
        self.rmse = tf.sqrt( tf.reduce_mean(tf.square(self.y - self.py)) )
        self.mae =  tf.reduce_mean( tf.abs(self.y - self.py) )
        self.mape = tf.reduce_mean( tf.abs( (self.y - self.py)*1.0/(self.y+1e-5) ) )
        
        
#   infer givn testing data    
    def inference(self, x_test, y_test, keep_prob):
        
        if self.att_type == 'both-att':
            return self.sess.run([self.att, self.py, self.rmse, self.mae, self.mape, self.py_indi], \
                                 feed_dict = {self.x:x_test, self.y:y_test, self.keep_prob:keep_prob})
        else:
            return self.sess.run([self.att, self.py, self.rmse, self.mae, self.mape], \
                             feed_dict = {self.x:x_test, self.y:y_test, self.keep_prob:keep_prob})
            
            
    
    def predict(self, x_test, y_test, keep_prob):
        
        return self.sess.run( self.py, feed_dict = {self.x:x_test, self.y:y_test, self.keep_prob:keep_prob})
    
    
    def test_regularization(self, x_test, y_test, keep_prob):
        return self.sess.run([self.regularization], feed_dict = {self.x:x_test, self.y:y_test,\
                                                                            self.keep_prob:keep_prob})
    
    def test_attention(self, x_test, y_test, keep_prob):
        return self.sess.run([self.att],  feed_dict = {self.x:x_test, self.y:y_test, self.keep_prob:keep_prob})
    
    def testfunc(self, x_batch, y_batch, keep_prob ):
        
        #tmpname= []
        #for tf_var in tf.trainable_variables():
        #    tmpname.append( tf_var.name )
            
        #self.test, regul_lstm    
        return self.sess.run([self.test],\
                            feed_dict={self.x:x_batch, self.y:y_batch, self.keep_prob:keep_prob })
  

