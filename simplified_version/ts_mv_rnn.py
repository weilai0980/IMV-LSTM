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
from ts_mv_rnn_layers import *
     
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
    
    def __init__(self, session):
        
        self.n_lstm_dim_layers = 0.0
        
        self.N_STEPS = 0.0
        self.N_DATA_DIM = 0.0
        
        self.loss_type = 0.0
        self.att_type = 0.0
        
        # placeholders
        self.x = 0
        self.y = 0
        self.keep_prob = 0
        
        self.sess = session
        
        self.lr = 0
        self.new_lr = 0
        
    def network_ini(self, 
                    n_lstm_dim_layers, 
                    n_steps, 
                    n_data_dim,
                    lr, 
                    max_norm, 
                    bool_residual,
                    att_type, 
                    temp_attention_type,
                    l2_dense, 
                    l2_att, 
                    vari_attention_type, 
                    loss_type,
                    dense_regul_type, 
                    layer_norm_type, 
                    num_mv_dense, 
                    ke_type, 
                    mv_rnn_gate_type,
                    bool_regular_lstm, 
                    bool_regular_attention, 
                    bool_regular_dropout_output,
                    vari_attention_after_mv_desne,
                    learning_vari_impt,
                    lr_decay
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
        
        learning_vari_impt: add the global variable importance learning process
        
        lr_decay: bool, if decay on learning rate
        
        Returns:
        
        '''
        
        # ---- fix the random seed to reproduce the results
        np.random.seed(1)
        tf.set_random_seed(1)
        
        
        # ---- ini
        
        self.n_lstm_dim_layers = n_lstm_dim_layers
        
        self.N_STEPS    = n_steps
        self.N_DATA_DIM = n_data_dim
        
        self.loss_type = loss_type
        self.att_type = att_type
        
        self.l2 = l2_dense
        
        # placeholders
        self.x = tf.placeholder(tf.float32, [None, self.N_STEPS, self.N_DATA_DIM], name = 'x')
        self.y = tf.placeholder(tf.float32, [None, 1], name = 'y')
        self.keep_prob = tf.placeholder(tf.float32, shape =(), name = 'keep_prob')
        
        
        if lr_decay == True:
            self.lr = tf.Variable(lr, trainable = False)
        else:
            self.lr = lr
        
        self.new_lr = tf.placeholder(tf.float32, shape = (), name = 'new_lr')
        
        
        # ---- begin to build the graph
        
        # residual connections
        if bool_residual == True:
            
            print('--- [ERROR] Wrongly use residual connection')
            
        # no residual connection 
        else:
              
            # both temporal and varaible attention combined via mixture 
            if att_type == 'both-mixture':
                
                print(' --- MV-RNN using both temporal and variate attention in ', att_type, '\n')
                
                with tf.variable_scope('lstm'):
                    
                    # ? layer_norm
                    # ? no-memory-loss dropout: off
                    mv_cell = MvLSTMCell(num_units = n_lstm_dim_layers[0], \
                                         n_var = n_data_dim ,\
                                         initializer = tf.contrib.layers.xavier_initializer(),\
                                         memory_update_keep_prob = self.keep_prob,\
                                         layer_norm = layer_norm_type,\
                                         gate_type = mv_rnn_gate_type)
                    
                    
                    # ? dropout for input-hidden, hidden-hidden, hidden-output 
                    # ? variational dropout
                    drop_mv_cell = tf.nn.rnn_cell.DropoutWrapper(mv_cell,\
                                                                 state_keep_prob = self.keep_prob )
                    
                    h, state = tf.nn.dynamic_rnn(cell = drop_mv_cell, 
                                                 inputs = self.x, 
                                                 dtype = tf.float32)
                
                
                # stacked mv-lstm
                for i in range(1, len(n_lstm_dim_layers)):
                    
                    with tf.variable_scope('lstm' + str(i)):
                        
                        mv_cell = MvLSTMCell(num_units = n_lstm_dim_layers[i], \
                                             n_var = n_data_dim ,\
                                             initializer = tf.contrib.layers.xavier_initializer(),\
                                             memory_update_keep_prob = self.keep_prob,\
                                             layer_norm = layer_norm_type,
                                             gate_type = mv_rnn_gate_type)
                    
                        # ? dropout for input-hidden, hidden-hidden, hidden-output 
                        # ? variational dropout
                        drop_mv_cell = tf.nn.rnn_cell.DropoutWrapper(mv_cell, \
                                                                     state_keep_prob = self.keep_prob )
                    
                        h, state = tf.nn.dynamic_rnn(cell = drop_mv_cell, 
                                                     inputs = h, 
                                                     dtype = tf.float32)
                
                # [V B T D]
                h_list = tf.split(h, [int(n_lstm_dim_layers[-1]/self.N_DATA_DIM)]*self.N_DATA_DIM, 2)
                
                
                # ---- temporal attention 
                
                # ? dropout
                # h_att_temp_input = tf.nn.dropout(h_list, tf.gather(self.keep_prob,1))
                h_att_temp_input = h_list
                
                
                # temporal attention
                # h_temp [V B 2D]
                h_temp, regu_att_temp, self.att_temp = mv_attention_temp(h_list = h_att_temp_input,
                                                                         v_dim = int(n_lstm_dim_layers[-1]/self.N_DATA_DIM),
                                                                         scope = 'att_temp', 
                                                                         n_step = self.N_STEPS, 
                                                                         att_type = temp_attention_type, 
                                                                         num_vari = self.N_DATA_DIM)
                # ---- variate attention BEFORE
                
                if vari_attention_after_mv_desne == False:
                    
                    # ? dropout
                    # h_att_vari_input = tf.nn.dropout(h_temp, tf.gather(self.keep_prob,1))
                    h_att_vari_input = h_temp
                
                    # variable attention 
                    _, regu_att_vari, self.att_vari, vari_logits = mv_attention_variate(h_temp = h_att_vari_input,
                                                                       var_dim = 2*int(n_lstm_dim_layers[-1]/self.N_DATA_DIM),
                                                                                        scope = 'att_vari', 
                                                                                        num_vari = self.N_DATA_DIM, 
                                                                                        att_type = vari_attention_type)
                
                # ---- multiple mv-dense layers
                # [V B 2D] - [V B d]
                
                # mv-dense layers before the output layer
                h_mv, regu_mv_dense, dim_vari = multi_mv_dense(num_layers = num_mv_dense,
                                                               keep_prob = self.keep_prob,
                                                               h_vari = h_temp,
                                                               dim_vari = 2*int(n_lstm_dim_layers[-1]/self.N_DATA_DIM), 
                                                               scope = "mv_dense",
                                                               num_vari = self.N_DATA_DIM,
                                                               activation_type = "relu",
                                                               max_norm_regul = max_norm,
                                                               regul_type = dense_regul_type)
                
                # ---- variate attention AFTER
                
                if vari_attention_after_mv_desne == True:
                    
                    # ? dropout
                    # h_att_vari_input = tf.nn.dropout(h_temp, tf.gather(self.keep_prob,1))
                    h_att_vari_input = h_mv
                
                    # variable attention 
                    # [B V 1]
                    _, regu_att_vari, self.att_vari, vari_logits = mv_attention_variate(h_temp = h_att_vari_input,
                                                                                        var_dim = dim_vari,
                                                                                        scope = 'att_vari', 
                                                                                        num_vari = self.N_DATA_DIM, 
                                                                                        att_type = vari_attention_type)
                    
                
                # ---- output layer: individual predictions
                # remove the constraints of max_norm regularization
                
                # ? dropout
                if bool_regular_dropout_output == True:
                    h_mv = tf.nn.dropout(h_mv, self.keep_prob)
                
                # outpout layer NO max-norm regularization
                h_mean, self.regu_dense_mean = mv_dense(h_vari = h_mv, 
                                                        dim_vari = dim_vari, 
                                                        scope = 'output_mean', 
                                                        num_vari = self.N_DATA_DIM, 
                                                        dim_to = 1, 
                                                        activation_type = "", 
                                                        max_norm_regul = 0.0, 
                                                        regul_type = dense_regul_type)
                
                # outpout layer NO max-norm regularization
                h_var, self.regu_dense_var = mv_dense(h_vari = h_mv, 
                                                      dim_vari = dim_vari, 
                                                      scope = 'output_var', 
                                                      num_vari = self.N_DATA_DIM, 
                                                      dim_to = 1, 
                                                      activation_type = "", 
                                                      max_norm_regul = 0.0, 
                                                      regul_type = dense_regul_type)
                
                
                # ---- output: mixture prediction 
                
                #[V B 1], [V B 1] 
                # h_mean, h_var = tf.split(h_mv, [1, 1], 2)
                h_var = tf.square(h_var)
                
                # [V B 1] - [B V 1]
                h_mv_trans = tf.transpose(h_mean, [1, 0, 2])
                # [B V 1]*[B V 1]
                h_mv_weighted = tf.reduce_sum(h_mv_trans*self.att_vari, 1)
                
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
                tmp_var_inv  = tf.transpose(tf.squeeze(h_var,  [2]), [1,0])
                
                # [B V]
                tmp_llk = tf.exp(-0.5*tf.square(y_tile - tmp_mean)/(tmp_var + 1e-5))/(2.0*np.pi*(tmp_var + 1e-5))**0.5
                llk = tf.multiply(tmp_llk, tf.squeeze(self.att_vari, [2])) 
                self.neg_llk = tf.reduce_sum(-1.0*tf.log(tf.reduce_sum(llk, 1)+1e-5))
                
                # [B V]
                tmp_llk_inv = tf.exp(-0.5*tf.square(y_tile - tmp_mean)*tmp_var_inv)/(2.0*np.pi)**0.5*(tmp_var_inv**0.5)
                llk_inv = tf.multiply(tmp_llk_inv, tf.squeeze(self.att_vari, [2])) 
                self.neg_llk_inv = tf.reduce_sum(-1.0*tf.log(tf.reduce_sum(llk_inv, 1)+1e-5))
                
                # [B V]
                tmp_pseudo_llk = tf.exp(-0.5*tf.square(y_tile - tmp_mean))/(2.0*np.pi)**0.5
                pseudo_llk = tf.multiply(tmp_pseudo_llk, tf.squeeze(self.att_vari, [2])) 
                self.pseudo_neg_llk = tf.reduce_sum(-1.0*tf.log(tf.reduce_sum(pseudo_llk, 1)+1e-5))
                
                
                # ---- knowledge extraction
                
                # attention summarization
                # self.att_temp: [V B T-1] self.att_vari: [B V 1]
                
                # -- variable-wise temporal
                    
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
                self.regularization = l2_dense*regu_mv_dense 
                
                # on attention
                if bool_regular_attention == True:
                    self.regularization += 0.1*l2_dense*(regu_att_temp + regu_att_vari)
                
                    
            # both temporal and varaible attention combined via fusion
            elif att_type == 'both-fusion':
                
                print(' --- MV-RNN using both temporal and variate attention in ', att_type, '\n')
                
                with tf.variable_scope('lstm'):
                    
                    # ? layer_norm
                    # ? ? ? no-memory-loss dropout: off
                    mv_cell = MvLSTMCell(num_units = n_lstm_dim_layers[0], 
                                         n_var = n_data_dim ,
                                         initializer = tf.contrib.layers.xavier_initializer(),
                                         memory_update_keep_prob = self.keep_prob,
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
                h_temp, regu_att_temp, self.att_temp = mv_attention_temp(h_list = h_att_temp_input,
                                                                         v_dim = int(n_lstm_dim_layers[0]/self.N_DATA_DIM),
                                                                         scope = 'att_temp',
                                                                         n_step = self.N_STEPS,
                                                                         att_type = temp_attention_type,
                                                                         num_vari = self.N_DATA_DIM)
                # ? dropout
                # h_att_vari_input = tf.nn.dropout(h_temp, tf.gather(self.keep_prob,1))
                h_att_vari_input = h_temp
                
                # variable attention 
                # [B 2D]
                h_vari, regu_att_vari, self.att_vari, _ = mv_attention_variate(h_temp = h_att_vari_input,
                                                                       var_dim = 2*int(n_lstm_dim_layers[0]/self.N_DATA_DIM),
                                                                               scope = 'att_vari',
                                                                               num_vari = self.N_DATA_DIM,
                                                                               att_type = vari_attention_type)
                
                # ---- plain multi dense layers
                # [B 2D]
                
                # dropout
                h, regu_dense, out_dim = multi_dense(x = h_vari,
                                                     x_dim = 2*int(n_lstm_dim_layers[-1]/self.N_DATA_DIM),
                                                     num_layers = num_mv_dense,
                                                     scope = 'multi_dense',
                                                     dropout_keep_prob = self.keep_prob,
                                                     max_norm_regul = max_norm)
                
                # ---- output layer
                # outpout layer NO max-norm regularization
                
                # ? dropout
                if bool_regular_dropout_output == True:
                    h = tf.nn.dropout(h, self.keep_prob)
                
                # [B 1]
                h_mean, self.regu_dense_mean = dense(x = h,
                                                     x_dim = out_dim, 
                                                     out_dim = 1,
                                                     scope = "output_mean",
                                                     dropout_keep_prob = 1.0, 
                                                     max_norm_regul = 0.0,
                                                     activation_type = "")
                
                # [B 1]
                h_var, self.regu_dense_var = dense(x = h,
                                                   x_dim = out_dim, 
                                                   out_dim = 1,
                                                   scope = "output_var",
                                                   dropout_keep_prob = 1.0, 
                                                   max_norm_regul = 0.0,
                                                   activation_type = "")
                
                # [B 1] [B 1]
                self.py = h_mean
                var = tf.square(h_var)
                inv_var = tf.square(h_var)
                
                # ---- negative log likelihood
                
                # llk: log likelihood 
                # [B 1]
                tmp_lk = tf.exp(-0.5*tf.square(self.y - self.py)/(var + 1e-5))/(2.0*np.pi*(var + 1e-5))**0.5
                #self.neg_llk = tf.reduce_sum(-1.0*tf.log(tmp_lk + 1e-5))
                
                self.neg_llk = tf.reduce_sum(0.5*tf.square(self.y - self.py)/(var + 1e-5) + 0.5*tf.log(var + 1e-5))
                
                # [B 1]
                tmp_lk_inv = tf.exp(-0.5*tf.square(self.y - self.py)*inv_var)/(2.0*np.pi)**0.5*(inv_var**0.5)
                #self.neg_llk_inv = tf.reduce_sum(-1.0*tf.log(tmp_lk_inv + 1e-5))
                
                self.neg_llk_inv = tf.reduce_sum(0.5*tf.square(self.y - self.py)*inv_var - 0.5*tf.log(inv_var + 1e-5))
                
                # [B 1]
                tmp_pseudo_lk = tf.exp(-0.5*tf.square(self.y - self.py))/(2.0*np.pi)**0.5
                #self.pseudo_neg_llk = tf.reduce_sum(-1.0*tf.log(tf.reduce_sum(pseudo_lk, 1)+1e-5))
                
                self.pseudo_neg_llk = tf.reduce_sum(0.5*tf.square(self.y - self.py))
                
                
                # ---- regularization
                # ?
                
                self.regularization = l2_dense*regu_dense
                
                if bool_regular_attention == True:
                    self.regularization += 0.1*l2_dense*(regu_att_temp + regu_att_vari)
                
                # ---- knowledge extraction 
                
                # [B V]
                self.att_prior = tf.squeeze(self.att_vari)
                
        
        # ---- regularization on LSTM
        
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
            self.loss = self.mse + self.regularization + self.l2*self.regu_dense_mean
        
        elif self.loss_type == 'lk_inv':
            
            # ? 
            self.loss = self.neg_llk_inv + self.regularization + self.l2*(self.regu_dense_mean + self.regu_dense_var)

        elif self.loss_type == 'lk':
            
            # ? 
            self.loss = self.neg_llk + self.regularization + self.l2*(self.regu_dense_mean + self.regu_dense_var)
        
        #elif self.loss_type == 'pseudo_lk':
        #    self.loss = self.pseudo_neg_llk + self.regularization + self.l2*self.regu_dense_mean
            
        else:
            
            print('--- [ERROR] loss type')
            return
        
        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.lr).minimize(self.loss)  
        
        self.init = tf.global_variables_initializer()
        self.sess.run([self.init])
    
    def train_batch(self, 
                    x_batch, 
                    y_batch, 
                    keep_prob, 
                    bool_lr_update, 
                    lr):
        
        # learning rate decay update
        if bool_lr_update == True:
            
            lr_update = tf.assign(self.lr, self.new_lr)
            _ = self.sess.run([lr_update], feed_dict={self.new_lr:lr})
        
        
        _, tmp_loss, tmp_sqsum = self.sess.run([self.optimizer, self.loss, self.square_error],
                                               feed_dict = {self.x:x_batch,
                                                            self.y:y_batch,
                                                            self.keep_prob:keep_prob })
        return tmp_loss, tmp_sqsum

#   initialize inference         
    def inference_ini(self):
        
        # error metric
        self.rmse = tf.sqrt(tf.reduce_mean(tf.square(self.y - self.py)))
        self.mae =  tf.reduce_mean(tf.abs(self.y - self.py))
        
        # filtering before mape calculation
        mask = tf.greater(tf.abs(self.y), 0.00001)
        
        y_mask = tf.boolean_mask(self.y, mask)
        y_hat_mask = tf.boolean_mask(self.py, mask)
        
        self.mape = tf.reduce_mean(tf.abs((y_mask - y_hat_mask)*1.0/(y_mask+1e-10)))
        
        # for restoring models
        tf.add_to_collection("pred", self.py)
        tf.add_to_collection("rmse", self.rmse)
        tf.add_to_collection("mae", self.mae)
        tf.add_to_collection("mape", self.mape)
        
        # for restoring models
        tf.add_to_collection("att_temp", tf.transpose(self.att_temp, [1, 0, 2]))
        tf.add_to_collection("att_prior", self.att_prior)
        tf.add_to_collection("att_posterior", self.att_posterior)
        tf.add_to_collection("ke_temp", self.ke_temp)
        tf.add_to_collection("ke_var_prior", self.ke_var_prior)
        tf.add_to_collection("ke_var_posterior", self.ke_var_posterior)
        
        
#   inference given data    
    def inference(self, 
                  x, 
                  y, 
                  keep_prob):
        
        return self.sess.run([self.py, self.rmse, self.mae, self.mape], 
                             feed_dict = {self.x:x, self.y:y, self.keep_prob:keep_prob})
    
    def predict(self, 
                x, 
                y, 
                keep_prob):
        
        return self.sess.run(self.py, feed_dict = {self.x:x, self.y:y, self.keep_prob:keep_prob})
    
    def knowledge_extraction(self, 
                             x, 
                             y, 
                             keep_prob):
        
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
            return self.sess.run([tf.transpose(self.att_temp, [1, 0, 2]), self.att_prior],
                                  feed_dict = {self.x:x, 
                                               self.y:y, 
                                               self.keep_prob:keep_prob})
        else:
            return -1
        
        
    # restore the model from the files
    def pre_train_restore_model(self, 
                                path_meta, 
                                path_data):
        
        saver = tf.train.import_meta_graph(path_meta, clear_devices=True)
        saver.restore(self.sess, path_data)
        
        return 0
        
    # inference using pre-trained model 
    def pre_train_inference(self, 
                            x_test, 
                            y_test, 
                            keep_prob):
        
        return self.sess.run([tf.get_collection('pred')[0],
                              tf.get_collection('rmse')[0],
                              tf.get_collection('mae')[0],
                              tf.get_collection('mape')[0]],
                              {'x:0': x_test, 'y:0': y_test, 'keep_prob:0': keep_prob})