#!/usr/bin/python

from utils_libs import *
import sys


import collections
import hashlib
import numbers


from tensorflow.python.ops import rnn_cell_impl

from tensorflow.python.ops.rnn_cell_impl import * 

from tensorflow.python.ops import nn_ops


# discriminative


_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"
_WEIGHTS_MULTI_VARIABLE_NAME = "mv_kernel"


#https://r2rt.com/recurrent-neural-networks-in-tensorflow-ii.html

#lstm_cell = [MyGRUCell(size, kernel_initializer = tf.contrib.layers.xavier_initializer() ) \
             #            for size in [n_lstm_dim, n_lstm_dim]]
#tf.contrib.keras.initializers.glorot_normal()



'''
class RNNCell(base_layer.Layer):
  """Abstract object representing an RNN cell.
  Every `RNNCell` must have the properties below and implement `call` with
  the signature `(output, next_state) = call(input, state)`.  The optional
  third input argument, `scope`, is allowed for backwards compatibility
  purposes; but should be left off for new subclasses.
  This definition of cell differs from the definition used in the literature.
  In the literature, 'cell' refers to an object with a single scalar output.
  This definition refers to a horizontal array of such units.
  An RNN cell, in the most abstract setting, is anything that has
  a state and performs some operation that takes a matrix of inputs.
  This operation results in an output matrix with `self.output_size` columns.
  If `self.state_size` is an integer, this operation also results in a new
  state matrix with `self.state_size` columns.  If `self.state_size` is a
  (possibly nested tuple of) TensorShape object(s), then it should return a
  matching structure of Tensors having shape `[batch_size].concatenate(s)`
  for each `s` in `self.batch_size`.
  """

  def __call__(self, inputs, state, scope=None):
    """Run this RNN cell on inputs, starting from the given state.
    Args:
      inputs: `2-D` tensor with shape `[batch_size x input_size]`.
      state: if `self.state_size` is an integer, this should be a `2-D Tensor`
        with shape `[batch_size x self.state_size]`.  Otherwise, if
        `self.state_size` is a tuple of integers, this should be a tuple
        with shapes `[batch_size x s] for s in self.state_size`.
      scope: VariableScope for the created subgraph; defaults to class name.
    Returns:
      A pair containing:
      - Output: A `2-D` tensor with shape `[batch_size x self.output_size]`.
      - New state: Either a single `2-D` tensor, or a tuple of tensors matching
        the arity and shapes of `state`.
    """
    if scope is not None:
      with vs.variable_scope(scope,
                             custom_getter=self._rnn_get_variable) as scope:
        return super(RNNCell, self).__call__(inputs, state, scope=scope)
    else:
      with vs.variable_scope(vs.get_variable_scope(),
                             custom_getter=self._rnn_get_variable):
        return super(RNNCell, self).__call__(inputs, state)

  def _rnn_get_variable(self, getter, *args, **kwargs):
    variable = getter(*args, **kwargs)
    if context.in_graph_mode():
      trainable = (variable in tf_variables.trainable_variables() or
                   (isinstance(variable, tf_variables.PartitionedVariable) and
                    list(variable)[0] in tf_variables.trainable_variables()))
    else:
      trainable = variable._trainable  # pylint: disable=protected-access
    if trainable and variable not in self._trainable_weights:
      self._trainable_weights.append(variable)
    elif not trainable and variable not in self._non_trainable_weights:
      self._non_trainable_weights.append(variable)
    return variable

  @property
  def state_size(self):
    """size(s) of state(s) used by this cell.
    It can be represented by an Integer, a TensorShape or a tuple of Integers
    or TensorShapes.
    """
    raise NotImplementedError("Abstract method")

  @property
  def output_size(self):
    """Integer or TensorShape: size of outputs produced by this cell."""
    raise NotImplementedError("Abstract method")

  def build(self, _):
    # This tells the parent Layer object that it's OK to call
    # self.add_variable() inside the call() method.
    pass

  def zero_state(self, batch_size, dtype):
    """Return zero-filled state tensor(s).
    Args:
      batch_size: int, float, or unit Tensor representing the batch size.
      dtype: the data type to use for the state.
    Returns:
      If `state_size` is an int or TensorShape, then the return value is a
      `N-D` tensor of shape `[batch_size x state_size]` filled with zeros.
      If `state_size` is a nested list or tuple, then the return value is
      a nested list or tuple (of the same structure) of `2-D` tensors with
      the shapes `[batch_size x s]` for each s in `state_size`.
    """
    with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
      state_size = self.state_size
      return _zero_state_tensors(state_size, batch_size, dtype)
'''

def _linear(args,
            output_size,
            bias,
            bias_initializer=None,
            kernel_initializer=None):
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

  # Now the computation.
  scope = vs.get_variable_scope()
  with vs.variable_scope(scope) as outer_scope:
    weights = vs.get_variable(
        _WEIGHTS_VARIABLE_NAME, [total_arg_size, output_size],
        dtype=dtype,
        initializer=kernel_initializer)
    if len(args) == 1:
      res = math_ops.matmul(args[0], weights)
    else:
      res = math_ops.matmul(array_ops.concat(args, 1), weights)
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


class MvGRUCell(RNNCell):
  """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078).
  Args:
    num_units: int, The number of units in the GRU cell.
    activation: Nonlinearity to use.  Default: `tanh`.
    reuse: (optional) Python boolean describing whether to reuse variables
     in an existing scope.  If not `True`, and the existing scope already has
     the given variables, an error is raised.
    kernel_initializer: (optional) The initializer to use for the weight and
    projection matrices.
    bias_initializer: (optional) The initializer to use for the bias.
  """

  def __init__(self,
               num_units,
               activation=None,
               reuse=None,
               kernel_initializer=None,
               bias_initializer=None):
    super(MyGRUCell, self).__init__(_reuse=reuse)
    self._num_units = num_units
    self._activation = activation or math_ops.tanh
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer
    self._gate_linear = None
    self._candidate_linear = None

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def call(self, inputs, state):
    """Gated recurrent unit (GRU) with nunits cells."""
    if self._gate_linear is None:
      bias_ones = self._bias_initializer
      if self._bias_initializer is None:
        bias_ones = init_ops.constant_initializer(1.0, dtype=inputs.dtype)
      with vs.variable_scope("gates"):  # Reset gate and update gate.
        
        #?
        self._gate_linear = _linear([inputs, state],
                                     2 * self._num_units, True, bias_initializer=bias_ones,\
                                    kernel_initializer=self._kernel_initializer)
        
        '''
        self._gate_linear = _Linear(
            [inputs, state],
            2 * self._num_units,
            True,
            bias_initializer=bias_ones,
            kernel_initializer=self._kernel_initializer)
        '''

    #value = math_ops.sigmoid(self._gate_linear([inputs, state]))
    
    #?
    value = math_ops.sigmoid(self._gate_linear)
    
    r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

    r_state = r * state
    if self._candidate_linear is None:
      with vs.variable_scope("candidate"):
        
        
        self._candidate_linear = _linear([inputs, r_state],
                                         self._num_units, True, bias_initializer=self._bias_initializer,\
                                         kernel_initializer=self._kernel_initializer)
        
        '''
        self._candidate_linear = _Linear(
            [inputs, r_state],
            self._num_units,
            True,
            bias_initializer=self._bias_initializer,
            kernel_initializer=self._kernel_initializer)
        '''
    
    c = self._activation(self._candidate_linear)
    #?    
    #c = self._activation(self._candidate_linear([inputs, r_state]))
    new_h = u * state + (1 - u) * c
    return new_h, new_h



class _LSTMCell(RNNCell):
  """Long short-term memory unit (LSTM) recurrent network cell.
  The default non-peephole implementation is based on:
    http://www.bioinf.jku.at/publications/older/2604.pdf
  S. Hochreiter and J. Schmidhuber.
  "Long Short-Term Memory". Neural Computation, 9(8):1735-1780, 1997.
  The peephole implementation is based on:
    https://research.google.com/pubs/archive/43905.pdf
  Hasim Sak, Andrew Senior, and Francoise Beaufays.
  "Long short-term memory recurrent neural network architectures for
   large scale acoustic modeling." INTERSPEECH, 2014.
  The class uses optional peep-hole connections, optional cell clipping, and
  an optional projection layer.
  """

  def __init__(self, num_units,
               use_peepholes=False, cell_clip=None,
               initializer=None, num_proj=None, proj_clip=None,
               num_unit_shards=None, num_proj_shards=None,
               forget_bias=1.0, state_is_tuple=True,
               activation=None, reuse=None):
    """Initialize the parameters for an LSTM cell.
    Args:
      num_units: int, The number of units in the LSTM cell.
      use_peepholes: bool, set True to enable diagonal/peephole connections.
      cell_clip: (optional) A float value, if provided the cell state is clipped
        by this value prior to the cell output activation.
      initializer: (optional) The initializer to use for the weight and
        projection matrices.
      num_proj: (optional) int, The output dimensionality for the projection
        matrices.  If None, no projection is performed.
      proj_clip: (optional) A float value.  If `num_proj > 0` and `proj_clip` is
        provided, then the projected values are clipped elementwise to within
        `[-proj_clip, proj_clip]`.
      num_unit_shards: Deprecated, will be removed by Jan. 2017.
        Use a variable_scope partitioner instead.
      num_proj_shards: Deprecated, will be removed by Jan. 2017.
        Use a variable_scope partitioner instead.
      forget_bias: Biases of the forget gate are initialized by default to 1
        in order to reduce the scale of forgetting at the beginning of
        the training. Must set it manually to `0.0` when restoring from
        CudnnLSTM trained checkpoints.
      state_is_tuple: If True, accepted and returned states are 2-tuples of
        the `c_state` and `m_state`.  If False, they are concatenated
        along the column axis.  This latter behavior will soon be deprecated.
      activation: Activation function of the inner states.  Default: `tanh`.
      reuse: (optional) Python boolean describing whether to reuse variables
        in an existing scope.  If not `True`, and the existing scope already has
        the given variables, an error is raised.
      When restoring from CudnnLSTM-trained checkpoints, must use
      CudnnCompatibleLSTMCell instead.
    """
    super(MyLSTMCell, self).__init__(_reuse=reuse)
    if not state_is_tuple:
      logging.warn("%s: Using a concatenated state is slower and will soon be "
                   "deprecated.  Use state_is_tuple=True.", self)
    if num_unit_shards is not None or num_proj_shards is not None:
      logging.warn(
          "%s: The num_unit_shards and proj_unit_shards parameters are "
          "deprecated and will be removed in Jan 2017.  "
          "Use a variable scope with a partitioner instead.", self)

    self._num_units = num_units
    self._use_peepholes = use_peepholes
    self._cell_clip = cell_clip
    self._initializer = initializer
    self._num_proj = num_proj
    self._proj_clip = proj_clip
    self._num_unit_shards = num_unit_shards
    self._num_proj_shards = num_proj_shards
    self._forget_bias = forget_bias
    self._state_is_tuple = state_is_tuple
    self._activation = activation or math_ops.tanh

    if num_proj:
      self._state_size = (
          LSTMStateTuple(num_units, num_proj)
          if state_is_tuple else num_units + num_proj)
      self._output_size = num_proj
    else:
      self._state_size = (
          LSTMStateTuple(num_units, num_units)
          if state_is_tuple else 2 * num_units)
      self._output_size = num_units
    self._linear1 = None
    self._linear2 = None
    if self._use_peepholes:
      self._w_f_diag = None
      self._w_i_diag = None
      self._w_o_diag = None

  @property
  def state_size(self):
    return self._state_size

  @property
  def output_size(self):
    return self._output_size

  def call(self, inputs, state):
    """Run one step of LSTM.
    Args:
      inputs: input Tensor, 2D, batch x num_units.
      state: if `state_is_tuple` is False, this must be a state Tensor,
        `2-D, batch x state_size`.  If `state_is_tuple` is True, this must be a
        tuple of state Tensors, both `2-D`, with column sizes `c_state` and
        `m_state`.
    Returns:
      A tuple containing:
      - A `2-D, [batch x output_dim]`, Tensor representing the output of the
        LSTM after reading `inputs` when previous state was `state`.
        Here output_dim is:
           num_proj if num_proj was set,
           num_units otherwise.
      - Tensor(s) representing the new state of LSTM after reading `inputs` when
        the previous state was `state`.  Same type and shape(s) as `state`.
    Raises:
      ValueError: If input size cannot be inferred from inputs via
        static shape inference.
    """
    num_proj = self._num_units if self._num_proj is None else self._num_proj
    sigmoid = math_ops.sigmoid

    if self._state_is_tuple:
      (c_prev, m_prev) = state
    else:
      c_prev = array_ops.slice(state, [0, 0], [-1, self._num_units])
      m_prev = array_ops.slice(state, [0, self._num_units], [-1, num_proj])

    dtype = inputs.dtype
    input_size = inputs.get_shape().with_rank(2)[1]
    if input_size.value is None:
      raise ValueError("Could not infer input size from inputs.get_shape()[-1]")
    if self._linear1 is None:
      scope = vs.get_variable_scope()
      with vs.variable_scope(
          scope, initializer=self._initializer) as unit_scope:
        if self._num_unit_shards is not None:
          unit_scope.set_partitioner(
              partitioned_variables.fixed_size_partitioner(
                  self._num_unit_shards))
        
        #?
        self._linear1 = _linear([inputs, m_prev], 4 * self._num_units, True)
        #self._linear1 = _Linear([inputs, m_prev], 4 * self._num_units, True)

    # i = input_gate, j = new_input, f = forget_gate, o = output_gate
    lstm_matrix = self._linear1([inputs, m_prev])
    i, j, f, o = array_ops.split(
        value=lstm_matrix, num_or_size_splits=4, axis=1)
    # Diagonal connections
    if self._use_peepholes and not self._w_f_diag:
      scope = vs.get_variable_scope()
      with vs.variable_scope(
          scope, initializer=self._initializer) as unit_scope:
        with vs.variable_scope(unit_scope):
          self._w_f_diag = vs.get_variable(
              "w_f_diag", shape=[self._num_units], dtype=dtype)
          self._w_i_diag = vs.get_variable(
              "w_i_diag", shape=[self._num_units], dtype=dtype)
          self._w_o_diag = vs.get_variable(
              "w_o_diag", shape=[self._num_units], dtype=dtype)

    if self._use_peepholes:
      c = (sigmoid(f + self._forget_bias + self._w_f_diag * c_prev) * c_prev +
           sigmoid(i + self._w_i_diag * c_prev) * self._activation(j))
    else:
      c = (sigmoid(f + self._forget_bias) * c_prev + sigmoid(i) *
           self._activation(j))

    if self._cell_clip is not None:
      # pylint: disable=invalid-unary-operand-type
      c = clip_ops.clip_by_value(c, -self._cell_clip, self._cell_clip)
      # pylint: enable=invalid-unary-operand-type
    if self._use_peepholes:
      m = sigmoid(o + self._w_o_diag * c) * self._activation(c)
    else:
      m = sigmoid(o) * self._activation(c)

    if self._num_proj is not None:
      if self._linear2 is None:
        scope = vs.get_variable_scope()
        with vs.variable_scope(scope, initializer=self._initializer):
          with vs.variable_scope("projection") as proj_scope:
            if self._num_proj_shards is not None:
              proj_scope.set_partitioner(
                  partitioned_variables.fixed_size_partitioner(
                      self._num_proj_shards))
            
            #?
            self._linear2 = _linear(m, self._num_proj, False)
            #self._linear2 = _Linear(m, self._num_proj, False)
      m = self._linear2(m)

      if self._proj_clip is not None:
        # pylint: disable=invalid-unary-operand-type
        m = clip_ops.clip_by_value(m, -self._proj_clip, self._proj_clip)
        # pylint: enable=invalid-unary-operand-type

    new_state = (LSTMStateTuple(c, m) if self._state_is_tuple else
                 array_ops.concat([c, m], 1))
    return m, new_state



def _mv_linear(args,
            output_size,
            bias,
            bias_initializer=None,
            kernel_initializer=None):
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

  # Now the computation.
  scope = vs.get_variable_scope()
  with vs.variable_scope(scope) as outer_scope:
        
    output_size_mv   = output_size/4
    output_size_gate = output_size/4*3
    
    weights = vs.get_variable(
        _WEIGHTS_VARIABLE_NAME, [total_arg_size, output_size],
        dtype=dtype,
        initializer=kernel_initializer)
    
    # --- mv cell update ---
    #split weight matrix
    weights_gate, weights_gate_mv = array_ops.split( weights, [output_size_gate, output_size_mv], 1)
    
    # gates of input, output, forget
    if len(args) == 1:
      res = math_ops.matmul(args[0], weights_gate)
    else:
      res = math_ops.matmul(array_ops.concat(args, 1), weights_gate)
    
    # new input
    tmp_input = args[0] 
    input_dim = args[0].get_shape()[1].value   
    per_dim   = output_size_mv/input_dim
    
    weights_mv_input, weights_mv_trans, _ = array_ops.split(weights_gate_mv, [1, per_dim, total_arg_size-1-per_dim], 0)
    
    weights_mv_input = array_ops.reshape( weights_mv_input, [input_dim, per_dim] )
    weights_mv_trans = array_ops.reshape( weights_mv_trans, [input_dim, per_dim, per_dim] )
    
    trans_input = array_ops.transpose(tmp_input, [1,0])
    
    tmp_h = args[1]
    blk_h = array_ops.split( tmp_h, num_or_size_splits = input_dim, axis=1)
    blk_h = array_ops.stack( blk_h, 2)
    blk_h = array_ops.transpose( blk_h, [2, 0, 1])
    #blk_h = array_ops.reshape( blk_h, [-1, per_dim])
    
    #res_h = math_ops.matmul(blk_h, weights_mv_trans[k]
    res_h = []
    res_mv = []
    for k in range(input_dim):
        res_mv.append( math_ops.matmul(array_ops.expand_dims(trans_input[k],1), array_ops.expand_dims(weights_mv_input[k],0)))
        res_h.append( math_ops.matmul(blk_h[k], weights_mv_trans[k]) )                    
                            
    res_mv = array_ops.concat(res_mv, 1)
    #for k in range(input_dim):
    #    res_h.append( math_ops.matmul(blk_h[k], weights_mv_trans[k]) )
    #blk_h = array_ops.transpose( blk_h, [2, 0,1])
    res_h = array_ops.concat(res_h, 1)
    
    res = array_ops.concat([res, res_mv+res_h], 1)
    # --- finish ---
    
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
# i, j, f, o

def _mv_linear_speed_up(args,
            output_size,
            bias,
            kernel_initializer, 
            n_var,
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
    
    # define transition variables
    weights_gate = vs.get_variable(
        'gate', [total_arg_size, output_size_gate],
        dtype=dtype,
        initializer=kernel_initializer)
    
    weights_IH = vs.get_variable(
        'input_transition',  [n_var, input_dim_per_var, hidden_dim_per_var],
        dtype = dtype,
        initializer = kernel_initializer)
    
    weights_HH = vs.get_variable(
        'hidden_transition', [n_var, hidden_dim_per_var, hidden_dim_per_var],
        dtype = dtype,
        initializer = kernel_initializer)
    
    # reshape input
    tmp_input = args[0] 
    blk_input = array_ops.split( tmp_input, num_or_size_splits = n_var, axis = 1 )
    #blk_input = array_ops.stack( blk_input, 2 )
    mv_input = blk_input
    #array_ops.transpose( blk_input, [2, 0, 1] )
    
    # reshape hidden 
    tmp_h = args[1]
    blk_h = array_ops.split( tmp_h, num_or_size_splits = n_var, axis = 1 )
    #blk_h = array_ops.stack( blk_h, 2 )
    mv_h = blk_h
    #= array_ops.transpose( blk_h, [2, 0, 1] )
    
    # perform multi-variate input and hidden transition
    res_h  = []
    res_mv = []
    for k in range(n_var):
        res_mv.append( math_ops.matmul( mv_input[k], weights_IH[k] ))
        res_h.append(  math_ops.matmul( mv_h[k],     weights_HH[k] ))                    
    
    # derive gates of input, output, forget
    if len(args) == 1:
      res = math_ops.matmul(args[0], weights_gate)
    else:
      res = math_ops.matmul(array_ops.concat(args, 1), weights_gate)
    
    # concate gates and new input
    res_mv = array_ops.concat(res_mv, 1)
    res_h = array_ops.concat(res_h, 1)
    res = array_ops.concat([res, res_mv + res_h], 1)
    
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
# i, j, f, o

def _mv_linear_final_speed_up(args,
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
    
    # define transition variables
    weights_gate = vs.get_variable(
        'gate', [total_arg_size, output_size_gate],
        dtype=dtype,
        initializer=kernel_initializer)
    
    # [OPTIMIZED]
    #[ V, 1, d D]
    weights_IH = vs.get_variable(
        'input_hidden',  [n_var, 1, hidden_dim_per_var, input_dim_per_var],
        dtype = dtype,
        initializer = kernel_initializer)
    
    # [OPTIMIZED]
    weights_HH = vs.get_variable(
        'hidden_hidden', [n_var, 1, hidden_dim_per_var, hidden_dim_per_var],
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
        tmp_var = tf.reduce_mean( tf.square(tmp_new_h - tmp_mean), axis = 2, keep_dims = True)
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
    
    # derive gates of input, output, forget
    #[B 3H]
    if len(args) == 1:
        res_gate = math_ops.matmul(args[0], weights_gate)
    else:
        res_gate = math_ops.matmul(array_ops.concat(args, 1), weights_gate)
    
    # concate gates and new input
    
    #res_mv = array_ops.concat(tmp_mv, 1)
    #res_h = array_ops.concat(tmp_h, 1)
    
    #merge_h = 
    
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

  def __init__(self, num_units, n_var, initializer, memory_update_keep_prob, layer_norm , forget_bias=1.0,
               state_is_tuple=True, activation=None, reuse=None ):
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
        
        #?
        #self._linear = _mv_linear([inputs, h], 4 * self._num_units, True, kernel_initializer = self._kernel_ini, self._n_var)
        #?
        self._linear = _mv_linear_final_speed_up([inputs, h], 4 * self._num_units, True, \
                                                 kernel_initializer = self._kernel_ini,\
                                                 n_var = self._n_var, \
                                                 layer_norm = self._layer_norm)
        
        #self._linear = _mv_linear([inputs, h], 4 * self._num_units, True)  
        #?  
        #self._linear = _Linear([inputs, h], 4 * self._num_units, True)
    
    # i = input_gate, j = new_input, f = forget_gate, o = output_gate
    
    i, f, o, j = array_ops.split(
        value= self._linear , num_or_size_splits=4, axis=1)
    
    
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

    new_c = (
        c * sigmoid(f + self._forget_bias) + sigmoid(i) * self._activation(j))
    new_h = self._activation(new_c) * sigmoid(o)

    if self._state_is_tuple:
      new_state = LSTMStateTuple(new_c, new_h)
    else:
      new_state = array_ops.concat([new_c, new_h], 1)
    return new_h, new_state