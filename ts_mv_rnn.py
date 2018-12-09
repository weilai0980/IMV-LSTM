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
    
    def __init__(self, n_lstm_dim_layers, n_steps, n_data_dim, session,\
                 lr, l2_dense, max_norm, bool_residual, att_type, l2_att, num_dense):
        
        self.LEARNING_RATE = lr
        self.L2 =  l2
        
        self.n_lstm_dim_layers = n_lstm_dim_layers
        
        self.N_STEPS    = n_steps
        self.N_DATA_DIM = n_data_dim
        
        self.att_type = att_type
        
        # placeholders
        self.x = tf.placeholder(tf.float32, [None, self.N_STEPS, self.N_DATA_DIM])
        self.y = tf.placeholder(tf.float32, [None, 1])
        self.keep_prob = tf.placeholder(tf.float32, [None])
        
        # begin to build the graph
        self.sess = session
        
        h, _ = plain_lstm( self.x, n_lstm_dim_layers, 'lstm', self.keep_prob )
        
        
        if att_type == 'temp':
            
            print(' --- Plain RNN using temporal attention:  ')
                
            h, self.att, regu_att = attention_temp_logit( h, n_lstm_dim_layers[-1], 'att', self.N_STEPS )
            
            # dropout
            h, regu_dense, out_dim = multi_dense( h, 2*n_lstm_dim_layers[-1], num_dense, \
                                                  'dense', tf.gather(self.keep_prob, 0), max_norm)
            
            #?
            self.regularization = l2_dense*regu_dense + l2_att*regu_att
            
        else:
            
            print(' --- Plain RNN using no attention:  ')
            
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
            
            w = tf.get_variable('w', shape=[out_dim, 1],\
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
        
        # filtering before mape calculation
        mask = tf.greater(tf.abs(self.y), 0.00001)
        
        y_mask = tf.boolean_mask(self.y, mask)
        y_hat_mask = tf.boolean_mask(self.py, mask)
        
        self.mape = tf.reduce_mean( tf.abs((y_mask - y_hat_mask)*1.0/(y_mask+1e-10)) )

#   infer givn testing data
    def inference(self, x_test, y_test, keep_prob):
        
        return self.sess.run([self.py, self.rmse, self.mae, self.mape], \
                             feed_dict = {self.x:x_test, self.y:y_test, self.keep_prob:keep_prob})
        
        
    
    def predict(self, x_test, y_test, keep_prob):
        return self.sess.run([self.py], feed_dict = {self.x:x_test, self.y:y_test, self.keep_prob:keep_prob})
    
    def test_attention(self, x_test, y_test, keep_prob):
        return self.sess.run( self.att,  feed_dict = {self.x:x_test, self.y:y_test, self.keep_prob:keep_prob})
    
    
# ---- separate RNN ----

class tsLSTM_seperate():
    
    def __init__(self, n_lstm_dim_layers, n_steps, n_data_dim, session,\
                 lr, l2_dense, max_norm , bool_residual, att_type, temp_attention_type, vari_attention_type,\
                 dense_regul_type, l2_att, num_dense):
        
        self.LEARNING_RATE = lr
        self.L2 =  l2_dense
        
        self.n_lstm_dim_layers = n_lstm_dim_layers
        
        self.N_STEPS    = n_steps
        self.N_DATA_DIM = n_data_dim
        
        self.x = tf.placeholder(tf.float32, [None, self.N_STEPS, self.N_DATA_DIM])
        self.y = tf.placeholder(tf.float32, [None, 1])
        
        self.keep_prob = tf.placeholder(tf.float32)
        
        self.n_dense_dim_layers = n_dense_dim_layers
        
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
            #h_temp, regu_att_temp, self.att_temp = sep_attention_temp(h_att_temp_input,\
            #                                                          n_lstm_dim_layers[-1],\
            #                                                          'att_temp',\
            #                                                          self.N_STEPS,\
            #                                                          temp_attention_type,\
            #                                                          self.N_DATA_DIM)
            
            h_temp, regu_att_temp, self.att_temp = mv_attention_temp(h_att_temp_input,
                                                                     n_lstm_dim_layers[-1],
                                                                     'att_temp', 
                                                                     self.N_STEPS, 
                                                                     [], 
                                                                     "",
                                                                     temp_attention_type, 
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
            # self.att = self.att_vari
            #[self.att_temp, self.att_vari]
                
            
            # mv-dense layers before the output layer
            h_mv, tmp_regu, dim_vari = multi_mv_dense(num_dense, 
                                                      self.keep_prob, 
                                                      h_temp,\
                                                      2*int(n_lstm_dim_layers[-1]), \
                                                      'mv_dense', 
                                                      self.N_DATA_DIM, \
                                                      False, 
                                                      max_norm, 
                                                      dense_regul_type)
            
            # ? dropout
            #h_mv2 = tf.nn.dropout(h_mv2, tf.gather(self.keep_prob,1))
            # outpout layer without max-norm regularization
            h_mv, regu_mv_dense = mv_dense( h_mv, dim_vari, 'mv_dense3', \
                                            self.N_DATA_DIM, 2, True, 0.0, dense_regul_type )
                
            # --- mixture prediction 
            
            #[V B 1], [V B 1] 
            h_mean, h_var = tf.split(h_mv, [1, 1], 2)
            h_var = tf.square(h_var)
            
            # [V B d] - [B V d]
            h_mv_trans = tf.transpose(h_mean, [1, 0, 2])
            # [B V d]*[B V 1]
            h_mv_weighted = tf.reduce_sum( h_mv_trans*self.att_vari, 1 )
                
            # [V 1]
            self.py = h_mv_weighted
                    
            # --- regularization
            # ?
            self.regularization = l2_dense*regu_mv_dense + l2_dense*tmp_regu 
            #+ l2_att*(regu_att_temp + regu_att_vari) 
            
    
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
        
        _, loss, err_sum = self.sess.run([self.optimizer, self.loss, self.error_sqsum ],\
                                      feed_dict = {self.x:x_batch, self.y:y_batch, self.keep_prob:keep_prob })
        return loss, err_sum

#   initialize inference         
    def inference_ini(self):

        # error metric
        self.rmse = tf.sqrt( tf.reduce_mean(tf.square(self.y - self.py)) )
        self.mae =  tf.reduce_mean( tf.abs(self.y - self.py) )
        
        # filtering before mape calculation
        mask = tf.greater(tf.abs(self.y), 0.00001)
        
        y_mask = tf.boolean_mask(self.y, mask)
        y_hat_mask = tf.boolean_mask(self.py, mask)
        
        self.mape = tf.reduce_mean( tf.abs((y_mask - y_hat_mask)*1.0/(y_mask+1e-10)) )
        
#   infer givn testing data
    def inference(self, x_test, y_test, keep_prob):
        
        return self.sess.run([self.py, self.rmse, self.mae, self.mape], \
                             feed_dict = {self.x:x_test, self.y:y_test, self.keep_prob:keep_prob})

# ---- multi-variate RNN ----
'''
RNN Regularization:

   Batch norm, Layer norm, Batch Normalized Recurrent Neural Networks
   
   DropConnect
    
   Regularization of Deep Neural Networks with Spectral Dropout
   
   dropout on input-hidden, hidden-hidden, hidden-output 

   RNN dropout without memory loss
   
   RNN-Drop: memory dropout

   max norm regularization
'''

class tsLSTM_mv():
    
    def __init__(self, n_lstm_dim_layers, n_steps, n_data_dim, session,\
                 lr, max_norm, bool_residual, \
                 att_type, temp_decay_type, temp_attention_type, \
                 l2_dense, l2_att, vari_attention_type, loss_type, \
                 dense_regul_type, layer_norm_type, num_mv_dense, ke_type, mv_rnn_gate_type,
                 bool_regular_lstm, bool_regular_attention, bool_regular_dropout_output,\
                 vari_attention_after_mv_desne,\
                 learning_vari_impt
                 ):
        
        '''
        Args:
        
        n_lstm_dim_layers: list of integers, list of sizes of each LSTM layer
        
        n_steps: integer, time step of the input sequence 
        
        n_data_dim: integer, dimension of data at each time step
        
        session: tensorflow session
        
        lr: float, learning rate
        
        max_norm: max norm constraint on weights used with dropout
        
        bool_residual: add residual connection between stacked LSTM layers
        
        att_type: string, type of attention, {both-att, both-fusion}  
        
        temp_decay_type: string
        
        temp_attention_type: string,
        
        l2_dense: float, l2 regularization on the dense layers
        
        l2_att: float, l2 regularization on attentions
        
        vari_attention_type: string,
        
        loss_type: string, type of loss functions, {mse, lk, pseudo_lk}
        
        dense_regul_type: string, regularization type, {l2, l1}
        
        layer_norm_type: string, layer normalization type in MV layer
        
        num_mv_dense: int, number of dense layers after the MV layer
        
        ke_type: string, knowledge extration type, {aggre_posterior, base_posterior, base_prior}
        
        mv_rnn_gate_type: string, gate update in MV layer, {full, tensor}
        
        bool_regular_lstm: regularization on LSTM weights
        
        bool_regular_attention: regularization on attention
        
        bool_regular_dropout_output: dropout before the outpout layer
        
        vari_attention_after_temp: if variable attention after the multi-mv-dense
        
        Returns:
        
        '''
        
        self.LEARNING_RATE = lr

        self.n_lstm_dim_layers = n_lstm_dim_layers
        
        self.N_STEPS    = n_steps
        self.N_DATA_DIM = n_data_dim
        
        self.loss_type = loss_type
        self.att_type = att_type
        
        # placeholders
        self.x = tf.placeholder(tf.float32, [None, self.N_STEPS, self.N_DATA_DIM])
        self.y = tf.placeholder(tf.float32, [None, 1])
        self.keep_prob = tf.placeholder(tf.float32, shape =() )
        
        steps = tf.constant( list(range(self.N_STEPS-2, -1, -1)), dtype=tf.float32 )
        alpha = tf.constant( 0.2, dtype=tf.float32 )
        # begin to build the graph
        self.sess = session
        
        self.lr = lr
        #tf.Variable(lr, trainable = False)
        self.new_lr = tf.placeholder(tf.float32, shape = (), name = 'new_lr')
        
        # trainable variables 
        # [1 V 1]
        self.vari_impt_logits = tf.get_variable('vari_impt_logits', [1, self.N_DATA_DIM, 1],\
                                                initializer = tf.contrib.layers.xavier_initializer() )
                                                # tf.initializers.ones()
                                                # tf.contrib.layers.xavier_initializer())
        # residual connections
        if bool_residual == True:
            
            print('--- [ERROR] Wrongly use residual connection')
            
        # no residual connection 
        else:
            
            '''
            
            # no attention
            if bool_att == '':
                
                print(' --- MV-RNN using no attention: ')
                
                with tf.variable_scope('lstm'):
                    lstm_cell = MvLSTMCell( n_lstm_dim_layers[0], n_var = n_data_dim, \
                                            initializer=tf.contrib.layers.xavier_initializer() )
                    h, state = tf.nn.dynamic_rnn(cell = lstm_cell, inputs = self.x, dtype = tf.float32)
            
                # obtain the last hidden state    
                h_last = tf.transpose( h, [1,0,2] )[-1]
                h = h_last
                
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
            
            # only temporal attention
            elif bool_att == 'temp':
                
                print(' --- MV-RNN using only temporal attention: ')

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
                                                       'att', self.N_STEPS, steps, temp_decay_type, temp_attention )
               
                # reshape to [B 2H]
                h_tmp = tf.split(h_temp, num_or_size_splits = n_data_dim, axis = 0) 
                h_att = tf.squeeze(tf.concat(h_tmp, 2), [0])
                
                # ---
                
                # ?
                h, regu_dense = plain_dense_leaky( h_att, 2*n_lstm_dim_layers[-1], n_dense_dim_layers, 'dense', \
                                                   self.keep_prob, alpha )
                
                # ?
                if temp_decay_type == '' or temp_decay_type == 'cutoff' :
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
                total_cnt
                print(' --- MV-RNN using only variate attention with direct output : ')
                
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
                    print('--- [ERROR] variate attention type')
                    
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
                
                print(' --- MV-RNN using only variate attention with mv-dense: ')
                
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
                
                # full attention
                # [V B d] - [B V d]
                h_mv_trans = tf.transpose(h_mv, [1, 0, 2])
                # [B V d]*[B V 1]
                h_mv_weighted = tf.reduce_sum( h_mv_trans*self.att, 1 )
                
                self.py = h_mv_weighted
                
                # --- regularization
                self.regularization = l2_dense*regu_mv_dense + l2_dense*regu_mv_dense1 + l2_att*regu_att_vari
            '''    
            
            if att_type == 'both-att':
                
                print(' --- MV-RNN using both temporal and variate attention in ', att_type)
                
                with tf.variable_scope('lstm'):
                    
                    # ? layer_norm
                    # ? no-memory-loss dropout: off
                    mv_cell = MvLSTMCell(n_lstm_dim_layers[0], \
                                         n_var = n_data_dim ,\
                                         initializer = tf.contrib.layers.xavier_initializer(),\
                                         memory_update_keep_prob = self.keep_prob,\
                                         layer_norm = layer_norm_type,\
                                         gate_type = mv_rnn_gate_type)
                    
                    # ? dropout for input-hidden, hidden-hidden, hidden-output 
                    # ? variational dropout
                    drop_mv_cell = tf.nn.rnn_cell.DropoutWrapper(mv_cell,\
                                                                 state_keep_prob = self.keep_prob )
                    
                    h, state = tf.nn.dynamic_rnn(cell = drop_mv_cell, inputs = self.x, dtype = tf.float32)
                
                
                # stacked mv-lstm
                for i in range(1, len(n_lstm_dim_layers)):
                    
                    with tf.variable_scope('lstm' + str(i)):
                        
                        mv_cell = MvLSTMCell(n_lstm_dim_layers[i], \
                                             n_var = n_data_dim ,\
                                             initializer = tf.contrib.layers.xavier_initializer(),\
                                             memory_update_keep_prob = self.keep_prob,\
                                             layer_norm = layer_norm_type,
                                             gate_type = mv_rnn_gate_type)
                    
                        # ? dropout for input-hidden, hidden-hidden, hidden-output 
                        # ? variational dropout
                        drop_mv_cell = tf.nn.rnn_cell.DropoutWrapper(mv_cell, \
                                                                     state_keep_prob = self.keep_prob )
                    
                        h, state = tf.nn.dynamic_rnn(cell = drop_mv_cell, inputs = h, dtype = tf.float32)
                
                
                # [V B T D]
                h_list = tf.split(h, [int(n_lstm_dim_layers[-1]/self.N_DATA_DIM)]*self.N_DATA_DIM, 2)
                
                
                # ---- temporal attention 
                
                # ? dropout
                # h_att_temp_input = tf.nn.dropout(h_list, tf.gather(self.keep_prob,1))
                h_att_temp_input = h_list
                
                # temporal attention
                # h_temp [V B 2D]
                h_temp, regu_att_temp, self.att_temp = mv_attention_temp(h_att_temp_input,
                                                                         int(n_lstm_dim_layers[-1]/self.N_DATA_DIM),
                                                                         'att_temp', 
                                                                         self.N_STEPS, 
                                                                         steps, 
                                                                         temp_decay_type,
                                                                         temp_attention_type, 
                                                                         self.N_DATA_DIM)
                # ---- variate attention BEFORE
                
                if vari_attention_after_mv_desne == False:
                    
                    # ? dropout
                    # h_att_vari_input = tf.nn.dropout(h_temp, tf.gather(self.keep_prob,1))
                    h_att_vari_input = h_temp
                
                    # variable attention 
                    _, regu_att_vari, self.att_vari, vari_logits = mv_attention_variate(h_att_vari_input,
                                                                           2*int(n_lstm_dim_layers[-1]/self.N_DATA_DIM),
                                                                           'att_vari', 
                                                                           self.N_DATA_DIM, 
                                                                           vari_attention_type)
                # ---- multiple mv-dense layers
                # [V B 2D] - [V B d]
                
                # mv-dense layers before the output layer
                h_mv, tmp_regu, dim_vari = multi_mv_dense(num_mv_dense, 
                                                          self.keep_prob, 
                                                          h_temp,
                                                          2*int(n_lstm_dim_layers[-1]/self.N_DATA_DIM), 
                                                          'mv_dense', 
                                                          self.N_DATA_DIM,
                                                          False, 
                                                          max_norm, 
                                                          dense_regul_type)
                
                # ---- variate attention AFTER
                
                if vari_attention_after_mv_desne == True:
                    
                    # ? dropout
                    # h_att_vari_input = tf.nn.dropout(h_temp, tf.gather(self.keep_prob,1))
                    h_att_vari_input = h_mv
                
                    # variable attention 
                    # [B V 1]
                    _, regu_att_vari, self.att_vari, vari_logits = mv_attention_variate(h_att_vari_input,
                                                                                        dim_vari,
                                                                                        'att_vari', 
                                                                                        self.N_DATA_DIM, 
                                                                                        vari_attention_type)
                
                
                # ---- individual prediction
                
                # ? dropout
                
                if bool_regular_dropout_output == True:
                    h_mv = tf.nn.dropout(h_mv, self.keep_prob)
                
                # outpout layer without max-norm regularization
                h_mv, tmp_regu_output = mv_dense(h_mv, dim_vari, 'output', self.N_DATA_DIM, 2, True, 0.0, dense_regul_type)
                
                mv_dense_regu = tmp_regu + tmp_regu_output
                
                
                # ---- mixture prediction 
                
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
                
                
                # ---- negative log likelihood
                
                # [B 1] -> [B V]
                y_tile = tf.tile(self.y, [1, self.N_DATA_DIM])
                
                # [B V]
                tmp_mean = tf.transpose(tf.squeeze(h_mean, [2]), [1,0]) 
                tmp_var  = tf.transpose(tf.squeeze(h_var,  [2]), [1,0])
                
                # ?
                # [B V]
                tmp_llk = tf.exp(-0.5*tf.square(y_tile - tmp_mean)/(tmp_var + 1e-5))/(2.0*np.pi*(tmp_var + 1e-5))**0.5
                llk = tf.multiply( tmp_llk, tf.squeeze(self.att_vari, [2]) ) 
                self.neg_logllk = tf.reduce_sum(-1.0*tf.log(tf.reduce_sum(llk, 1)+1e-5))
                
                # ?
                tmp_pseudo_llk = tf.exp( -0.5*tf.square(y_tile - tmp_mean) )/(2.0*np.pi)**0.5
                pseudo_llk = tf.multiply( tmp_pseudo_llk, tf.squeeze(self.att_vari, [2]) ) 
                self.pseudo_neg_logllk = tf.reduce_sum( -1.0*tf.log(tf.reduce_sum(pseudo_llk, 1)+1e-5) )
                
                
                # ---- knowledge extraction
                
                # attention summarization 
                # self.att_temp: [V B T-1] self.att_vari: [B V 1]
                
                # variable-wise temporal 
                    
                # [V T-1]
                var_temp_w = tf.reduce_sum(self.att_temp, 1)
                # [V 1]
                var_sum_w = tf.reduce_sum(var_temp_w, 1, keepdims = True)
                # [V T-1]
                self.ke_temp = 1.0*var_temp_w / var_sum_w
                
                
                # -- prior variable 
                
                # [B V]
                self.att_prior = tf.squeeze( self.att_vari )
                
                # [V]
                var_w_prior = tf.squeeze(tf.reduce_sum(self.att_vari, 0))
                # 1
                sum_w_prior = tf.reduce_sum(var_w_prior)
                self.ke_var_prior = 1.0*var_w_prior / sum_w_prior
                
                
                # -- posterior variable 
                
                # TO DO ?
                # [B V] : [B V 1]
                tmp_energy = tf.exp( -0.5 * tf.square(y_tile - tmp_mean) ) + 0.001
                
                # ?
                # [B V]
                joint_llk = tmp_energy * tf.squeeze(self.att_vari, 2)
                # [B]
                normalizer = tf.reduce_sum(joint_llk, 1, keepdims = True)
                # [B V]    
                tmp_posterior = 1.0*joint_llk / (normalizer + 1e-10)
                
                # [B V]
                self.att_posterior = tmp_posterior
                
                # test?
                self.sumw = tf.reduce_sum(tmp_posterior) 
                
                # [V]
                var_w_posterior = tf.squeeze(tf.reduce_sum(tmp_posterior, 0))
                # 1
                sum_w_posterior = tf.reduce_sum(tmp_posterior)
                
                self.ke_var_posterior = 1.0*var_w_posterior / sum_w_posterior
                
                # ---- regularization
                
                # ?
                self.regularization = l2_dense*mv_dense_regu 
                
                if bool_regular_attention == True:
                    self.regularization += 0.1*l2_dense*(regu_att_temp + regu_att_vari)
                
        
            elif att_type == 'both-fusion':
                
                print(' --- MV-RNN using both temporal and variate attention in ', att_type)
                
                with tf.variable_scope('lstm'):
                    
                    # ? layer_norm
                    # ? ? ? no-memory-loss dropout: off
                    mv_cell = MvLSTMCell(n_lstm_dim_layers[0], 
                                         n_var = n_data_dim ,\
                                         initializer = tf.contrib.layers.xavier_initializer(),\
                                         memory_update_keep_prob = self.keep_prob,\
                                         layer_norm = layer_norm_type,
                                         gate_type = mv_rnn_gate_type)
                    
                    # ? dropout for input-hidden, hidden-hidden, hidden-output 
                    # ? variational dropout
                    drop_mv_cell = tf.nn.rnn_cell.DropoutWrapper(mv_cell, state_keep_prob = self.keep_prob)
                    
                    h, state = tf.nn.dynamic_rnn(cell = drop_mv_cell, inputs = self.x, dtype = tf.float32)
                
                # [V B T D]
                h_list = tf.split(h, [int(n_lstm_dim_layers[0]/self.N_DATA_DIM)]*self.N_DATA_DIM, 2)
                
                
                # ---- temporal and variate attention 
                
                # ? dropout
                # h_att_temp_input = tf.nn.dropout(h_list, tf.gather(self.keep_prob,1))
                h_att_temp_input = h_list
                
                # temporal attention
                # h_temp [V B 2D]
                h_temp, regu_att_temp, self.att_temp = mv_attention_temp(h_att_temp_input,\
                                                                         int(n_lstm_dim_layers[0]/self.N_DATA_DIM),\
                                                                         'att_temp', \
                                                                         self.N_STEPS, \
                                                                         steps, \
                                                                         temp_decay_type,\
                                                                         temp_attention_type, \
                                                                         self.N_DATA_DIM)
                
                # ? dropout
                # h_att_vari_input = tf.nn.dropout(h_temp, tf.gather(self.keep_prob,1))
                h_att_vari_input = h_temp
                
                # variable attention 
                # [B 2D]
                h_vari, regu_att_vari, self.att_vari = mv_attention_variate(h_att_vari_input,\
                                                                            2*int(n_lstm_dim_layers[0]/self.N_DATA_DIM),\
                                                                            'att_vari',\
                                                                            self.N_DATA_DIM,\
                                                                            vari_attention_type)
                # ---- plain dense layers
                # [B 2D]
                
                # dropout
                h, regu_dense, out_dim = multi_dense(h_vari, \
                                                     2*int(n_lstm_dim_layers[-1]/self.N_DATA_DIM), \
                                                     num_mv_dense, \
                                                     'dense', \
                                                     self.keep_prob, \
                                                     max_norm)
                
                # ? dropout
                if bool_regular_dropout_output == True:
                    h = tf.nn.dropout(h, self.keep_prob)
                
                with tf.variable_scope("output"):
                    
                    w = tf.get_variable('w', shape=[out_dim, 1],\
                                        initializer=tf.contrib.layers.xavier_initializer())
                    b = tf.Variable(tf.zeros( [ 1 ] ))
            
                    self.py = tf.matmul(h, w) + b
                    regu_dense += tf.nn.l2_loss(w)
                
                # ---- regularization
                # ?
                
                self.regularization = l2_dense*regu_dense
                
                if bool_regular_attention == True:
                    self.regularization += 0.1*l2_dense*(regu_att_temp + regu_att_vari)
                
                # ---- knowledge extraction 
                
                # [B V]
                self.att_prior = tf.squeeze( self.att_vari )
           
        
        # ---- regularization in LSTM
        
        # ?
        self.regul_lstm = sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables() if ("lstm" in tf_var.name))
        
        if bool_regular_lstm == True:
            self.regularization += 0.1*l2_dense*self.regul_lstm

        
    def train_ini(self):  
        
        self.square_error = tf.reduce_sum(tf.square(self.y - self.py))
        
        # loss function 
        if self.loss_type == 'mse':
            
            #?
            self.mse = tf.reduce_mean(tf.square(self.y - self.py))
            
            # ? 
            self.loss = self.mse + self.regularization
            
        elif self.loss_type == 'lk':
            
            # ? 
            self.loss = self.neg_logllk + self.regularization
        
        elif self.loss_type == 'pseudo_lk':
            
            self.loss = self.pseudo_neg_logllk + self.regularization
            
        else:
            
            print('--- [ERROR] loss type')
            return
        
        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.lr).minimize(self.loss)  
        
        self.init = tf.global_variables_initializer()
        self.sess.run( [self.init] )
    
    def train_batch(self, x_batch, y_batch, keep_prob, bool_set_lr, new_lr):
        
        # learning rate decay update
        if bool_set_lr == True:
            
            self.lr_update = tf.assign(self.lr, self.new_lr)
            _ = self.sess.run([self.lr_update], feed_dict={self.new_lr:lr})
        
        _, tmp_loss, tmp_sqsum = self.sess.run([self.optimizer, self.loss, self.square_error],\
                                               feed_dict = {self.x:x_batch, \
                                                            self.y:y_batch, \
                                                            self.keep_prob:keep_prob })
        return tmp_loss, tmp_sqsum

#   initialize inference         
    def inference_ini(self):
        
        # error metric
        self.rmse = tf.sqrt( tf.reduce_mean(tf.square(self.y - self.py)) )
        self.mae =  tf.reduce_mean( tf.abs(self.y - self.py) )
        
        # filtering before mape calculation
        mask = tf.greater(tf.abs(self.y), 0.00001)
        
        y_mask = tf.boolean_mask(self.y, mask)
        y_hat_mask = tf.boolean_mask(self.py, mask)
        
        self.mape = tf.reduce_mean( tf.abs( (y_mask - y_hat_mask)*1.0/(y_mask+1e-10) ) )
        
        
#   inference givn data    
    def inference(self, x, y, keep_prob):
        
        return self.sess.run([self.py, self.rmse, self.mae, self.mape, self.vari_impt], 
                             feed_dict = {self.x:x, self.y:y, self.keep_prob:keep_prob})
    
    def predict(self, x, y, keep_prob):
        
        return self.sess.run( self.py, feed_dict = {self.x:x, self.y:y, self.keep_prob:keep_prob})
    
    def knowledge_extraction(self, x, y, keep_prob):
        
        if self.att_type == 'both-att':
            
            # self.att_temp: [V B T-1] self.att_vari: [B V 1]
            
            return self.sess.run([self.sumw, 
                                  tf.transpose(self.att_temp, [1, 0, 2]), 
                                  self.att_prior,
                                  self.att_posterior, 
                                  self.ke_temp, 
                                  self.ke_var_prior, 
                                  self.ke_var_posterior],
                                  feed_dict = {self.x:x, 
                                              self.y:y, 
                                              self.keep_prob:keep_prob})
        elif self.att_type == 'both-fusion':
            
            # self.att_temp: [V B T-1] self.att_vari: [B V 1]
            
            return self.sess.run([tf.transpose(self.att_temp, [1, 0, 2]), 
                                  self.att_prior],
                                  feed_dict = {self.x:x, 
                                              self.y:y, 
                                              self.keep_prob:keep_prob})
        else:
            return -1