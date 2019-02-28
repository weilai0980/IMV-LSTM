#!/usr/bin/python

from utils_libs import *
import sys

import collections
import hashlib
import numbers

from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops.rnn_cell_impl import * 
from tensorflow.python.ops import nn_ops

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"
_WEIGHTS_MULTI_VARIABLE_NAME = "mv_kernel"


def _mv_linear_full(args,
                    output_size,
                    bias,
                    kernel_initializer, 
                    n_var,
                    layer_norm,
                    bias_initializer=None):
    
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_initializer: starting value to initialize the bias
      (default is all zeros).
    kernel_initializer: starting value to initialize the weight.
    
  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """

  if args is None or (nest.is_sequence(args) and not args):
    raise ValueError("`args` must be specified")
  if not nest.is_sequence(args):
    args = [args]

  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.get_shape() for a in args]
  for shape in shapes:
    if shape.ndims != 2:
      raise ValueError("linear is expecting 2D arguments: %s" % shapes)
    if shape[1].value is None:
      raise ValueError("linear expects shape[1] to be provided for shape %s, "
                       "but saw %s" % (shape, shape[1]))
    else:
      total_arg_size += shape[1].value

  dtype = [a.dtype for a in args][0]


  # --- begin multi-variate cell update ---
    
  scope = vs.get_variable_scope()
  with vs.variable_scope(scope) as outer_scope:
    
    # mv cell
    output_size_mv   = output_size/4
    output_size_gate = output_size/4*3

    # define relevant dimensions
    input_dim  = args[0].get_shape()[1].value
    hidden_dim = args[1].get_shape()[1].value
    
    input_dim_per_var  = int(input_dim / n_var)
    hidden_dim_per_var = int(hidden_dim / n_var)
    
    # --- hidden update
    
    # [OPTIMIZED]
    #[ V, 1, d D]
    weights_IH = vs.get_variable('input_hidden', 
                                 [n_var, 1, hidden_dim_per_var, input_dim_per_var],
                                 dtype = dtype,
                                 initializer = kernel_initializer)
    
    # [OPTIMIZED]
    weights_HH = vs.get_variable('hidden_hidden', 
                                 [n_var, 1, hidden_dim_per_var, hidden_dim_per_var],
                                 dtype = dtype,
                                 initializer = kernel_initializer)
    
    # reshape input
    # [B H]
    tmp_input = args[0] 
    blk_input = array_ops.split( tmp_input, num_or_size_splits = n_var, axis = 1 )
    #blk_input = array_ops.stack( blk_input, 2 )
    # [V B 1 D]
    mv_input = array_ops.expand_dims(blk_input, 2)
    #array_ops.transpose( blk_input, [2, 0, 1] )
    
    # reshape hidden 
    tmp_h = args[1]
    blk_h = array_ops.split( tmp_h, num_or_size_splits = n_var, axis = 1 )
    #blk_h = array_ops.stack( blk_h, 2 )
    mv_h = array_ops.expand_dims(blk_h, 2)
    #= array_ops.transpose( blk_h, [2, 0, 1] )
    
    # [OPTIMIZED] perform multi-variate input and hidden transition
    
    tmp_IH = math_ops.reduce_sum( mv_input*weights_IH, 3 )
    tmp_HH = math_ops.reduce_sum( mv_h*weights_HH, 3 )
    # [V B D]
    
    # --- layer normaization specific for each variable 
    if layer_norm == 'mv-ln':
        
        #[V B D]
        tmp_new_h = tmp_IH + tmp_HH
        
        #[V B 1]
        tmp_mean = tf.reduce_mean(tmp_new_h, axis = 2, keep_dims = True)
        #[V B 1]
        tmp_var = tf.reduce_mean(tf.square(tmp_new_h - tmp_mean), axis = 2, keep_dims = True)
        #[V B 1]
        tmp_std = tf.sqrt( 1.0*tmp_var )
        
        #[V 1 D]
        ln_g = vs.get_variable( 'ln_g', [n_var, 1, hidden_dim_per_var], dtype=dtype, initializer=kernel_initializer)
        ln_b = vs.get_variable( 'ln_b', [n_var, 1, hidden_dim_per_var], dtype=dtype, initializer=kernel_initializer)
    
        #[V B D]
        normalized_h = ln_g*(tmp_new_h - tmp_mean)/(tmp_std + 1e-5) + ln_b
        
        tmp_h  = array_ops.concat( array_ops.split(normalized_h, num_or_size_splits = n_var, axis = 0), 2 )
        res_new_h = array_ops.squeeze(tmp_h, axis = 0)
        
    # mv_ln
    # mv_bn
    
    # shared_ln
    # shared_bn
    
    # TO DO 
    elif layer_norm == 'mv-bn':
        
        # [V 1 B D]
        tmp_mv = array_ops.split(tmp_IH, num_or_size_splits = n_var, axis = 0) 
        tmp_h  = array_ops.split(tmp_HH, num_or_size_splits = n_var, axis = 0)
    
        #[1 B H]
        res_mv = array_ops.concat(tmp_mv, 2)
        res_h = array_ops.concat(tmp_h, 2)
    
        res_mv = array_ops.squeeze(res_mv, axis = 0)
        res_h = array_ops.squeeze(res_h, axis = 0)
        #[B H]
        res_new_h = res_mv + res_h
    
    else:
        
        # [V 1 B D]
        tmp_mv = array_ops.split(tmp_IH, num_or_size_splits = n_var, axis = 0) 
        tmp_h  = array_ops.split(tmp_HH, num_or_size_splits = n_var, axis = 0)
    
        #[1 B H]
        res_mv = array_ops.concat(tmp_mv, 2)
        res_h = array_ops.concat(tmp_h, 2)
    
        res_mv = array_ops.squeeze(res_mv, axis = 0)
        res_h = array_ops.squeeze(res_h, axis = 0)
        #[B H]
        res_new_h = res_mv + res_h
    
    #for k in range(n_var):
    #    res_mv.append( math_ops.matmul( mv_input[k], weights_IH[k] ))
    #    res_h.append(  math_ops.matmul( mv_h[k],     weights_HH[k] ))                    
    
    
    # --- gates of input, output, forget
    
    # define transition variables
    weights_gate = vs.get_variable('gate', 
                                   [total_arg_size, output_size_gate],
                                   dtype=dtype,
                                   initializer = kernel_initializer)
    
    tmp_IH = math_ops.reduce_sum( mv_input*weights_IH, 3 )
    tmp_HH = math_ops.reduce_sum( mv_h*weights_HH, 3 )
    
    #[B 3H]
    if len(args) == 1:
        res_gate = math_ops.matmul(args[0], weights_gate)
        
    else:
        res_gate = math_ops.matmul(array_ops.concat(args, 1), weights_gate)
    
    # concate gates and new input
    
    #res_mv = array_ops.concat(tmp_mv, 1)
    #res_h = array_ops.concat(tmp_h, 1)
    
    #[B 4H]
    res = array_ops.concat([res_gate, res_new_h], 1)
    
    # --- finish multi-variate cell update ---
    
    if not bias:
      return res
    with vs.variable_scope(outer_scope) as inner_scope:
      inner_scope.set_partitioner(None)
      if bias_initializer is None:
        bias_initializer = init_ops.constant_initializer(0.0, dtype=dtype)
      biases = vs.get_variable(
          _BIAS_VARIABLE_NAME, [output_size],
          dtype=dtype,
          initializer=bias_initializer)
    
    return nn_ops.bias_add(res, biases)

def _mv_linear_tensor(args,
                      output_size,
                      bias,
                      kernel_initializer, 
                      n_var,
                      layer_norm,
                      bias_initializer=None):
    
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_initializer: starting value to initialize the bias
      (default is all zeros).
    kernel_initializer: starting value to initialize the weight.
  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """

  if args is None or (nest.is_sequence(args) and not args):
    raise ValueError("`args` must be specified")
  if not nest.is_sequence(args):
    args = [args]

  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.get_shape() for a in args]
  for shape in shapes:
    if shape.ndims != 2:
      raise ValueError("linear is expecting 2D arguments: %s" % shapes)
    if shape[1].value is None:
      raise ValueError("linear expects shape[1] to be provided for shape %s, "
                       "but saw %s" % (shape, shape[1]))
    else:
      total_arg_size += shape[1].value

  dtype = [a.dtype for a in args][0]


  # --- begin multi-variate cell update ---
    
  scope = vs.get_variable_scope()
  with vs.variable_scope(scope) as outer_scope:
    
    # mv cell
    output_size_mv   = output_size/4
    output_size_gate = output_size/4*3

    # define relevant dimensions
    input_dim  = args[0].get_shape()[1].value
    hidden_dim = args[1].get_shape()[1].value
    
    input_dim_per_var  = int(input_dim / n_var)
    hidden_dim_per_var = int(hidden_dim / n_var)
    
    # --- hidden update
    
    #[ V, 1, d D]
    weights_IH = vs.get_variable('input_hidden', 
                                 [n_var, 1, hidden_dim_per_var, input_dim_per_var],
                                 dtype = dtype,
                                 initializer = kernel_initializer)
    
    weights_HH = vs.get_variable('hidden_hidden', 
                                 [n_var, 1, hidden_dim_per_var, hidden_dim_per_var],
                                 dtype = dtype,
                                 initializer = kernel_initializer)
    # reshape input
    # [B H]
    tmp_input = args[0] 
    blk_input = array_ops.split(tmp_input, num_or_size_splits = n_var, axis = 1)
    # [V B 1 D]
    mv_input = array_ops.expand_dims(blk_input, 2)
    
    # reshape hidden 
    tmp_h = args[1]
    blk_h = array_ops.split(tmp_h, num_or_size_splits = n_var, axis = 1)
    mv_h = array_ops.expand_dims(blk_h, 2)
    
    tmp_IH = math_ops.reduce_sum( mv_input*weights_IH, 3 )
    tmp_HH = math_ops.reduce_sum( mv_h*weights_HH, 3 )
    # [V B D]
    
    # [V 1 B D]
    tmp_mv = array_ops.split(tmp_IH, num_or_size_splits = n_var, axis = 0) 
    tmp_h  = array_ops.split(tmp_HH, num_or_size_splits = n_var, axis = 0)
    
    #[1 B H]
    res_mv = array_ops.concat(tmp_mv, 2)
    res_h = array_ops.concat(tmp_h, 2)
    
    res_mv = array_ops.squeeze(res_mv, axis = 0)
    res_h = array_ops.squeeze(res_h, axis = 0)
    #[B H]
    res_new_h = res_mv + res_h
    
    # --- gates of input, output, forget
    
    #[ V, 1, d D]
    gate_w_IH = vs.get_variable('gate_input_hidden',
                                 [n_var, 1, 3*hidden_dim_per_var, input_dim_per_var],
                                 dtype = dtype,
                                 initializer = kernel_initializer)
    # [OPTIMIZED]
    gate_w_HH = vs.get_variable('gate_hidden_hidden',
                                 [n_var, 1, 3*hidden_dim_per_var, hidden_dim_per_var],
                                 dtype = dtype,
                                 initializer = kernel_initializer)
    # [V 1 B 3D]
    gate_tmp_IH = math_ops.reduce_sum(mv_input*gate_w_IH, 3)
    gate_tmp_HH = math_ops.reduce_sum(mv_h*gate_w_HH, 3)
    
    # [V 1 B 3D]
    gate_tmp_mv = array_ops.split(gate_tmp_IH, num_or_size_splits = n_var, axis = 0) 
    gate_tmp_h  = array_ops.split(gate_tmp_HH, num_or_size_splits = n_var, axis = 0)
    
    #[1 B 3H]
    gate_res_mv = array_ops.concat(gate_tmp_mv, 2)
    gate_res_h = array_ops.concat(gate_tmp_h, 2)
    
    gate_res_mv = array_ops.squeeze(gate_res_mv, axis = 0)
    gate_res_h = array_ops.squeeze(gate_res_h, axis = 0)
    #[B 3H]
    res_gate = gate_res_mv + gate_res_h
    
    #[B 4H]
    res = array_ops.concat([res_gate, res_new_h], 1)
    
    # --- finish multi-variate cell update ---
    
    if not bias:
      return res
    with vs.variable_scope(outer_scope) as inner_scope:
      inner_scope.set_partitioner(None)
      if bias_initializer is None:
        bias_initializer = init_ops.constant_initializer(0.0, dtype=dtype)
      biases = vs.get_variable(
          _BIAS_VARIABLE_NAME, [output_size],
          dtype=dtype,
          initializer=bias_initializer)
    
    return nn_ops.bias_add(res, biases)


class MvLSTMCell(RNNCell):
  """Basic LSTM recurrent network cell.
  The implementation is based on: http://arxiv.org/abs/1409.2329.
  We add forget_bias (default: 1) to the biases of the forget gate in order to
  reduce the scale of forgetting in the beginning of the training.
  It does not allow cell clipping, a projection layer, and does not
  use peep-hole connections: it is the basic baseline.
  For advanced models, please use the full @{tf.nn.rnn_cell.LSTMCell}
  that follows.
  """

  def __init__(self, num_units, n_var, initializer, memory_update_keep_prob, layer_norm, gate_type,
               forget_bias=1.0, state_is_tuple=True, activation=None, reuse=None ):
    """Initialize the basic LSTM cell.
    Args:
      num_units: int, The number of units in the LSTM cell.
      forget_bias: float, The bias added to forget gates (see above).
        Must set to `0.0` manually when restoring from CudnnLSTM-trained
        checkpoints.
      state_is_tuple: If True, accepted and returned states are 2-tuples of
        the `c_state` and `m_state`.  If False, they are concatenated
        along the column axis.  The latter behavior will soon be deprecated.
      activation: Activation function of the inner states.  Default: `tanh`.
      reuse: (optional) Python boolean describing whether to reuse variables
        in an existing scope.  If not `True`, and the existing scope already has
        the given variables, an error is raised.
      When restoring from CudnnLSTM-trained checkpoints, must use
      CudnnCompatibleLSTMCell instead.
    """
    super(MvLSTMCell, self).__init__(_reuse=reuse)
    if not state_is_tuple:
      logging.warn("%s: Using a concatenated state is slower and will soon be "
                   "deprecated.  Use state_is_tuple=True.", self)
    self._num_units = num_units
    self._forget_bias = forget_bias
    self._state_is_tuple = state_is_tuple
    self._activation = activation or math_ops.tanh
    self._linear = None
    
    self._input_dim = None
    
    # added by mv_rnn
    self._n_var = n_var
    self._kernel_ini = initializer
    self._memory_update_keep_prob = memory_update_keep_prob
    self._layer_norm = layer_norm 
    
    self.gate_type = gate_type

  @property
  def state_size(self):
    return (LSTMStateTuple(self._num_units, self._num_units)
            if self._state_is_tuple else 2 * self._num_units)

  @property
  def output_size(self):
    return self._num_units

  def call(self, inputs, state):
    """Long short-term memory cell (LSTM).
    Args:
      inputs: `2-D` tensor with shape `[batch_size x input_size]`.
      state: An `LSTMStateTuple` of state tensors, each shaped
        `[batch_size x self.state_size]`, if `state_is_tuple` has been set to
        `True`.  Otherwise, a `Tensor` shaped
        `[batch_size x 2 * self.state_size]`.
    Returns:
      A pair containing the new hidden state, and the new state (either a
        `LSTMStateTuple` or a concatenated state, depending on
        `state_is_tuple`).
    """
    sigmoid = math_ops.sigmoid
    # Parameters of gates are concatenated into one multiply for efficiency.
    if self._state_is_tuple:
      c, h = state
    else:
      c, h = array_ops.split(value=state, num_or_size_splits=2, axis=1)

    if self._linear is None:
        
        if self.gate_type == 'full':
            self._linear = _mv_linear_full([inputs, h], 
                                           4 * self._num_units, 
                                           True, 
                                           kernel_initializer = self._kernel_ini,\
                                           n_var = self._n_var, 
                                           layer_norm = self._layer_norm)
            
        elif self.gate_type == 'tensor':
            self._linear = _mv_linear_tensor([inputs, h], 
                                             4 * self._num_units, 
                                             True, 
                                             kernel_initializer = self._kernel_ini,\
                                             n_var = self._n_var, 
                                             layer_norm = self._layer_norm)
        else:
            print('[ERROR]    mv-lstm cell type')
    
    i, f, o, j = array_ops.split(value = self._linear, num_or_size_splits=4, axis=1)
    
    
    # --- !!! Recurrent Dropout without Memory Loss
    # ?
    # j = nn_ops.dropout( j, keep_prob = self._memory_update_keep_prob )
    #, seed=self._gen_seed(salt_prefix, i)
    
    
    # --- layer normaization, use informatin from all variables 
    if self._layer_norm == 'shared':
        
        scope = vs.get_variable_scope()
        with vs.variable_scope(scope) as outer_scope:
            
            #[B 1]
            tmp_mean = tf.reduce_mean(j, axis = 1, keep_dims = True)
            #[B H]
            tmp_var  = tf.reduce_mean(tf.square(j - tmp_mean), axis = 1, keep_dims = True) 
            tmp_std = tf.sqrt( 1.0*tmp_var )
        
            #[1 H]
            ln_g = vs.get_variable('ln_g', [1, self._num_units ], dtype=dtype, initializer=kernel_initializer)
            ln_b = vs.get_variable('ln_b', [1, self._num_units ], dtype=dtype, initializer=kernel_initializer)
        
            #[B H]
            j = ln_g*(j - tmp_mean)/(tmp_std + 1e-5) + ln_b
    
    # ---
    
    #?
    #i, j, f, o = array_ops.split(
    #    value= self._linear , num_or_size_splits=4, axis=1)
    
    # --- multivariate new_input ---
    '''
    if self._input_dim == None:
        self._input_dim = inputs.get_shape()[1].value
    
    per_dim = self._num_units*1.0 / self._input_dim
    trans_input = array_ops.transpose(inputs, [1,0])
    
    tmp_j = []
    for tmpdim in range(self._input_dim):
        tmp_j.append( _linear( array_ops.expand_dims(trans_input[tmpdim], 1), per_dim, True) )

    j = array_ops.concat(tmp_j, 1 )
    '''
    # ---  ---
    
    #?
    #i, j, f, o = array_ops.split(
    #    value=self._linear([inputs, h]), num_or_size_splits=4, axis=1)

    new_c = (c * sigmoid(f + self._forget_bias) + sigmoid(i) * self._activation(j))
    new_h = self._activation(new_c) * sigmoid(o)

    if self._state_is_tuple:
      new_state = LSTMStateTuple(new_c, new_h)
    else:
      new_state = array_ops.concat([new_c, new_h], 1)
    return new_h, new_state