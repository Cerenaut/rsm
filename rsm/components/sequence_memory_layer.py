# Copyright (C) 2019 Project AGI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""SequenceMemoryLayer class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np
import tensorflow as tf

from pagi.components.summary_component import SummaryComponent
from pagi.components.conv_autoencoder_component import ConvAutoencoderComponent

from pagi.utils import image_utils
from pagi.utils.layer_utils import activation_fn
from pagi.utils.tf_utils import tf_build_stats_summaries
from pagi.utils.tf_utils import tf_build_top_k_mask_4d_op
from pagi.utils.tf_utils import tf_create_optimizer
from pagi.utils.tf_utils import tf_do_training
from pagi.utils.tf_utils import tf_random_mask
from pagi.utils.tf_utils import tf_get_kernel_initializer
from pagi.utils.tf_utils import tf_init_type_none


class SequenceMemoryLayer(SummaryComponent):
  """
  Sequence Memory layer component based on K-Sparse Convolutional Autoencoder with variable-order feedback dendrites.
  """

  @staticmethod
  def default_hparams():
    """Builds an HParam object with default hyperparameters."""
    return tf.contrib.training.HParams(

        # Optimizer options
        optimizer='adam',
        loss_type='mse',
        learning_rate=0.0005,
        batch_size=80,

        # General options
        training_interval=[0, -1],  # [0,-1] means forever
        mode='predict-input',
        input_norm_first=False,
        hidden_nonlinearity='tanh', # used for hidden layer only
        decode_nonlinearity='none', # Used for decoding
        decode_mode='fc',
        inhibition_decay=0.1,  # controls refractory period
        inhibition_with_mask=True,

        # Deprecated features
        #predictor_norm_input=True,
        #predictor_integrate_input=False,  # default: only current cells
        # Deprecated features

        # Dendrites
        f_keep_rate=1.0,  # Optional dropout
        f_decay_rate=0.0,  # Optional integrated/exp decay
        f_decay_floor=0.0,  # if > 0, then clip to zero at this level
        f_norm_type=None,  # Option to normalize
        f_norm_eps=1.0e-11,  # Prevents norm /0
        f_decay_trainable=False,
        f_decay_rate_max = 0.95,  # If trainable, then this is the max decay rate

        rb_keep_rate=1.0,  # Optional dropout on feedback
        rb_decay_rate=0.0,  # Optional integrated/exp decay feedback
        rb_decay_floor=0.0,  # if > 0, then clip to zero at this level
        rb_norm_type='sum',  # Option to normalize feedback
        rb_norm_eps=1.0e-11,  # Prevents feedback norm /0
        rb_decay_trainable=False,
        rb_decay_rate_max = 0.95,  # If trainable, then this is the max decay rate

        # Geometry
        filters=-1,  # Ignored, but inherited
        filters_field_width=28,
        filters_field_height=28,
        filters_field_stride=28,

        cols=160,
        cells_per_col=3,  # 480 = 160 columns * 3 cells

        # Control statistics
        freq_update_interval=10,  # the interval in batches between frequency measurement updates
        freq_learning_rate=0.1,   # the learning rate of frequency measurement update as coefficient of the exponential rule
        freq_min=0.05,  # used by lifetime sparsity mask

        boost_factor=0.0,
        boost_factor_decay=0.0,
        boost_factor_update_interval=0,  # num training batches between boost factor updates

        # Sparse parameters:
        sparsity=25,
        lifetime_sparsity_dends=False,
        lifetime_sparsity_cols=False,

        # Bias
        # i_scale=1.0,
        # i_bias=0.0,
        f_bias=False,
        r_bias=False,
        b_bias=False,
        d_bias=True,  # Currently ignored (always on)

        # Regularization, 0=Off
        hidden_keep_rate=1.0,
        f_l2=0.0,  # feed Forward
        r_l2=0.0,  # Recurrent
        b_l2=0.0,  # feed-Back
        d_l2=0.0,  # Decode

        # Initialization
        f_init_type=tf_init_type_none,
        r_init_type=tf_init_type_none,
        b_init_type=tf_init_type_none,
        d_init_type=tf_init_type_none,

        f_bias_init_type=tf_init_type_none,
        r_bias_init_type=tf_init_type_none,
        b_bias_init_type=tf_init_type_none,
        d_bias_init_type=tf_init_type_none,

        f_init_sd=0.03,
        r_init_sd=0.03,
        b_init_sd=0.03,
        d_init_sd=0.03,

        # Summaries
        summarize_input=False,
        summarize_encoding=False,
        summarize_decoding=False,
        summarize_weights=False,
        summarize_freq=False
    )

  # Static names
  loss = 'loss'
  training = 'training'
  encoding = 'encoding'

  mode_encode_input = 'encode-input'  # i.e. be an autoencoder
  mode_predict_input = 'predict-input'  # Predict next input F
  mode_predict_target = 'predict-target'  # Predict some target input

  usage = 'usage'
  usage_col = 'usage-col'
  usage_cell = 'usage-cell'
  freq = 'freq'
  freq_col = 'freq-col'
  freq_cell = 'freq-cell'

  boost = 'boost'
  boost_assign = 'boost-assign'
  boost_factor = 'boost-factor'
  #prediction_input = 'prediction-input'
  f_keep = 'f-keep'
  rb_keep = 'rb-keep'
  hidden_keep = 'hidden-keep'
  lifetime_mask = 'lifetime-mask'
  encoding_mask = 'encoding-mask'
  encoding = 'encoding'
  decoding = 'decoding'
  decoding_logits = 'decoding_logits'
  encoding_f = 'forward_encoding'
  encoding_r = 'recurrent_encoding'
  encoding_b = 'feedback_encoding'
  sum_abs_error = 'sum-abs-error'

  # History
  history = 'history'  # Mask per batch sample whether to keep or discard

  # History tensors - value is history dependent
  feedback = 'feedback'
  recurrent = 'recurrent'
  inhibition = 'inhibition'
  previous = 'previous'

  # History masked values
  recurrent_mask = 'recurrent-mask'  # Op/values that are selectively masked due to history mask
  feedback_mask = 'feedback-mask'
  inhibition_mask = 'inhibition-mask'
  previous_mask = 'previous-mask'

  def update_statistics(self, batch_type, session):  # pylint: disable=W0613
    if self.use_freq():
      self._update_freq()
      self._update_lifetime_mask()   # sensitive to minimum frequency

      if self.use_boosting():
        self._update_boost(batch_type, session)
        self._update_boost_factor(batch_type)

  def _update_boost_factor(self, batch_type):
    # Only update boost batch count for training batches
    if batch_type == self.training:
      #print('training batch ', self._boost_batch_count, 'int ', self._hparams.boost_factor_update_interval)
      if ((self._boost_batch_count % self._hparams.boost_factor_update_interval) == 0) and (self._boost_batch_count > 0):
        self._boost_batch_count = 0
        old_boost_factor = self._dual.get_values(self.boost_factor)
        new_boost_factor = np.copy(old_boost_factor)
        new_boost_factor = old_boost_factor * self._hparams.boost_factor_decay
        if old_boost_factor > 0.00001:  # Who cares after this
          logging.info('Updating BOOST=======================> %f -> %f', old_boost_factor, new_boost_factor)
        self._dual.set_values(self.boost_factor, new_boost_factor)
      self._boost_batch_count += 1

  def _update_freq(self):
    """Updates the cell utilisation frequency from usage."""
    if ((self._freq_update_count % self._hparams.freq_update_interval) == 0) and (self._freq_update_count > 0):
      self._freq_update_count = 0
      logging.debug('Updating freq...')
      self._update_freq_with_usage(self.usage_cell, self.freq_cell)
      self._update_freq_with_usage(self.usage_col, self.freq_col)
    self._freq_update_count += 1

  def _update_freq_with_usage(self, usage_key, freq_key):
    """Update frequency from usage count."""

    # Calculate frequency from usage count
    # ---------------------------------------------------------------------
    conv_h = self._encoding_shape[1]
    conv_w = self._encoding_shape[2]
    conv_area = conv_w * conv_h
    freq_norm = 1.0 / float(self._hparams.freq_update_interval * self._hparams.batch_size * conv_area)

    usage = self._dual.get_values(usage_key)
    freq_old = self._dual.get_values(freq_key)
    freq = np.multiply(usage, freq_norm) # Convert to observed frequency
    self._dual.set_values_to(usage_key, 0.0) # Reset usage

    # Linear interpolation
    # ---------------------------------------------------------------------
    a = self._hparams.freq_learning_rate
    b = 1.0 - a

    fa = np.multiply(a, freq)
    fb = np.multiply(b, freq_old)

    freq_new = np.add(fa, fb) # now the frequency has been updated
    self._dual.set_values(freq_key, freq_new)

  def _update_lifetime_mask(self):
    """Update the lifetime mask."""
    if self._freq_update_count != 1:
      return

    lifetime_mask = self._dual.get(self.lifetime_mask)
    freq_values = self._dual.get_values(self.freq_cell)
    lifetime_mask_values = lifetime_mask.get_values()

    num_dendrites = self.get_num_dendrites()
    for d in range(num_dendrites):
      freq = freq_values[d]
      mask_value = False  # no lifetime
      if freq < self._hparams.freq_min:
        mask_value = True
      lifetime_mask_values[d] = mask_value
    lifetime_mask.set_values(lifetime_mask_values)

  def update_recurrent(self):
    """If feedback is only the recurrent state, then simply copy it into place."""
    #if self._hparams.autoencode is True:
    if self._hparams.mode == self.mode_encode_input:
      return  # No feedback

    output = self.get_values(self.encoding)
    self.set_recurrent(output)

  def set_recurrent(self, recurrent_values):
    """Since feedback might include external input, we explicitly need to do this."""
    recurrent = self._dual.get(self.recurrent)
    recurrent.set_values(recurrent_values)

  def use_freq(self):
    """Optional frequency tracking"""
    if self._hparams.freq_update_interval < 0:
      return False
    return True

  def use_feedback(self):
    if self._feedback_shape is None:
      return False
    return True

  def set_feedback(self, feedback_values):
    """Since feedback might include external input, we explicitly need to do this."""
    feedback = self._dual.get(self.feedback)
    feedback.set_values(feedback_values)

  def forget_history(self, session, history_forgetting_probability, clear_previous=False):
    """Clears the history with fixed probability, but optionally keep the previous (next input)."""
    if history_forgetting_probability <= 0.0:
      #print('Forget P=0 so ignore')
      return

    #print('Forget P >0: ', history_forgetting_probability)
    if history_forgetting_probability < 1.0:
      p0 = history_forgetting_probability
      history_mask = tf_random_mask(p0, shape=(self._hparams.batch_size))
      #print('Forget mask: ', history_mask)
    else:  # probability >= 1
      history_mask = np.zeros((self._hparams.batch_size))  # Clear all

    self.update_history(session, history_mask, clear_previous=clear_previous)

  def update_history(self, session, history_mask, clear_previous=True):
    """Clear recurrent history given a mask of all samples in the batch.
    It's as if a new episode or sequence has started so we don't want to learn the transition
    from the previous one. Inputs recurrent and feedback are conditionally cleared, it's assumed
    their values are already up to date and otherwise ready for next batch."""

    # Keep/discard values that change over time. This list includes:
    # * Inhibition
    # * Previous
    # * Recurrent
    # * Feedback
    #print('mask:', history_mask)
    #print('clear previous: ', clear_previous)

    inhibition = self._dual.get(self.inhibition)   # 6d_dend: b, h, w, col, cell, dend = 6d
    previous = self._dual.get(self.previous)  # 4d_input: b, xh, xw, xd = 4d
    recurrent = self._dual.get(self.recurrent)  # 4d_cells: b, h, w, col * cell = 5d
    history = self._dual.get(self.history)

    inhibition_values = inhibition.get_values()  # 6d_dend: b, h, w, col, cell, dend = 6d
    previous_values = previous.get_values()  # 4d_input: b, xh, xw, xd = 4d
    recurrent_values = recurrent.get_values()  # 4d_cells: b, h, w, col * cell = 5d

    inhibition_pl = inhibition.get_pl()  # 6d_dend: b, h, w, col, cell, dend = 6d
    previous_pl = previous.get_pl()  # 4d_input: b, xh, xw, xd = 4d
    recurrent_pl = recurrent.get_pl()  # 4d_cells: b, h, w, col * cell = 5d
    history_pl = history.get_pl()

    feed_dict = {
        inhibition_pl: inhibition_values,
        previous_pl: previous_values,
        recurrent_pl: recurrent_values,
        history_pl: history_mask
    }

    fetches = {
        self.inhibition_mask: self._dual.get_op(self.inhibition_mask),
        self.recurrent_mask: self._dual.get_op(self.recurrent_mask)
    }

    # Dendrite traces
    def trace_update_feed_dict(self, prefix, feed_dict):
      trace_key = self.get_trace_key(prefix)
      self._dual.update_feed_dict(feed_dict, [trace_key])

    def trace_add_fetches(self, prefix, fetches):
      trace_mask_key = self.get_trace_mask_key(prefix)
      self._dual.add_fetches(fetches, [trace_mask_key])

    if self._hparams.f_decay_rate > 0.0:
      trace_update_feed_dict(self, 'f', feed_dict)
      trace_add_fetches(self, 'f', fetches)
    if self._hparams.rb_decay_rate > 0.0:
      trace_update_feed_dict(self, 'r', feed_dict)
      trace_add_fetches(self, 'r', fetches)
      if self.use_feedback():
        trace_update_feed_dict(self, 'b', feed_dict)
        trace_add_fetches(self, 'b', fetches)

    if self.use_feedback():
      feedback = self._dual.get(self.feedback)  # 4d
      feedback_values = feedback.get_values()  # 4d: b, h?, w?, d?
      feedback_pl = feedback.get_pl()  # 4d
      feed_dict.update({
          feedback_pl: feedback_values
      })
      fetches[self.feedback_mask] = self._dual.get_op(self.feedback_mask)

    if clear_previous is True:
      fetches[self.previous_mask] = self._dual.get_op(self.previous_mask)

    # Execute the history update on all "history" tensors
    fetched = session.run(fetches, feed_dict=feed_dict)

    # Store the updated values
    inhibition_values_masked = fetched[self.inhibition_mask]
    recurrent_values_masked = fetched[self.recurrent_mask]

    self._dual.set_values(self.inhibition, inhibition_values_masked)
    self._dual.set_values(self.recurrent, recurrent_values_masked)

    if self.use_feedback():
      feedback_values_masked = fetched[self.feedback_mask]
      self._dual.set_values(self.feedback, feedback_values_masked)

    if clear_previous is True:
      previous_values_masked = fetched[self.previous_mask]
      self._dual.set_values(self.previous, previous_values_masked)

    # Dendrite traces
    def trace_set_fetches(self, prefix, fetched):
      trace_key = self.get_trace_key(prefix)
      trace_mask_key = self.get_trace_mask_key(prefix)
      trace_values_masked = fetched[self.name][trace_mask_key]
      self._dual.set_values(trace_key, trace_values_masked)

    if self._hparams.f_decay_rate > 0.0:
      trace_set_fetches(self, 'f', fetched)
    if self._hparams.rb_decay_rate > 0.0:
      trace_set_fetches(self, 'r', fetched)
      if self.use_feedback():
        trace_set_fetches(self, 'b', fetched)

  def get_num_dendrites(self):
    #num_dendrites = self._hparams.cols * self._hparams.cells_per_col * self._hparams.dends_per_cell
    return self.get_num_cells()

  def get_num_cells(self):
    num_cells = self._hparams.cols * self._hparams.cells_per_col
    return num_cells

  @staticmethod
  def get_encoding_shape_4d(input_shape, hparams):
    return ConvAutoencoderComponent.get_convolved_shape(input_shape,
                                                        hparams.filters_field_height,
                                                        hparams.filters_field_width,
                                                        hparams.filters_field_stride,
                                                        hparams.cols * hparams.cells_per_col,
                                                        padding='SAME')

  def build(self, input_values, input_shape, hparams, name='rsm', encoding_shape=None, feedback_shape=None,
            target_shape=None, target_values=None):  # pylint: disable=W0221
    """Initializes the model parameters.

    Args:
        input_values: Tensor containing input
        input_shape: The shape of the input, for display (internal is vectorized)
        label_values: Tensor containing label
        label_shape: The shape of the label, for display (internal is vectorized)
        hparams: The hyperparameters for the model as tf.contrib.training.HParams.
        name: A globally unique graph name used as a prefix for all tensors and ops.
        encoding_shape: The shape to be used to display encoded (hidden layer) structures
    """
    self._loss_mean = None
    self.name = name
    self._hparams = hparams
    self._freq_update_count = 0
    self._training_batch_count = 0
    self._boost_batch_count = 0

    self._weights_f = None
    self._weights_r = None
    self._weights_b = None
    self._weights_d = None
    self._bias_f = None
    self._bias_r = None
    self._bias_b = None
    self._bias_d = None

    self._input_shape = input_shape
    self._input_values = input_values

    self._target_shape = target_shape
    self._target_values = target_values

    self._encoding_shape = encoding_shape
    if self._encoding_shape is None:
      self._encoding_shape = SequenceMemoryLayer.get_encoding_shape_4d(input_shape, self._hparams)

    self._feedback_shape = feedback_shape

    logging.info('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    logging.info('Layer: %s', self.name)
    logging.info('# Cols: %s', str(self._hparams.cols))
    logging.info('# Cells/Col: %s', str(self._hparams.cells_per_col))
    logging.info('# Total cells: %s', str(self._hparams.cols * self._hparams.cells_per_col))
    logging.info('Input shape: %s', str(self._input_shape))
    logging.info('Encoding shape: %s', str(self._encoding_shape))
    if self.use_feedback():
      logging.info('Feedback shape: %s', str(self._feedback_shape))
    else:
      logging.info('Feedback shape: N/A')
    _, target_shape = self.get_target()
    logging.info('Target shape: %s', str(target_shape))
    logging.info('Decoding mode: %s', str(self._hparams.decode_mode))
    logging.info('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
      self._build()
      self._build_optimizer()

    self.reset()

  def _build_history_update_trace(self, prefix, history_4d):
    trace_key = self.get_trace_key(prefix)
    trace_mask_key = self.get_trace_mask_key(prefix)
    trace_pl = self._dual.get_pl(trace_key)
    trace_masked = tf.multiply(trace_pl, history_4d)
    self._dual.set_op(trace_mask_key, trace_masked)

  def _build_history_update(self):
    """Builds graph ops to update the history tensors given a history mask. The mask is specified in terms of batch samples."""

    history_pl = self._dual.add(self.history, shape=[self._hparams.batch_size], default_value=1.0).add_pl()
    history_4d = tf.reshape(history_pl, [self._hparams.batch_size, 1, 1, 1])
    history_5d = tf.reshape(history_pl, [self._hparams.batch_size, 1, 1, 1, 1])

    previous_pl = self._dual.get_pl(self.previous)
    inhibition_pl = self._dual.get_pl(self.inhibition)
    recurrent_pl = self._dual.get_pl(self.recurrent)

    previous_masked = tf.multiply(previous_pl, history_4d)
    inhibition_masked = tf.multiply(inhibition_pl, history_5d)
    recurrent_masked = tf.multiply(recurrent_pl, history_4d)

    self._dual.set_op(self.previous_mask, previous_masked)
    self._dual.set_op(self.inhibition_mask, inhibition_masked)
    self._dual.set_op(self.recurrent_mask, recurrent_masked)

    # Dendrite traces
    if self._hparams.f_decay_rate > 0.0:
      self._build_history_update_trace('f', history_4d)

    if self._hparams.rb_decay_rate > 0.0:
      self._build_history_update_trace('r', history_4d)

      if self.use_feedback():
        self._build_history_update_trace('b', history_4d)

    if self.use_feedback():
      feedback_pl = self._dual.get_pl(self.feedback)
      feedback_masked = tf.multiply(feedback_pl, history_4d)
      self._dual.set_op(self.feedback_mask, feedback_masked)

  def get_trace_key(self, prefix):
    return prefix + '-trace'

  def get_trace_mask_key(self, prefix):
    return prefix + '-trace-mask'

  def get_keep_rate_key(self, prefix):
    return prefix + '-keep'

  def _build_trainable_decay(self, prefix, input_4d, input_shape_4d, max_decay_rate):
    """Build fully connected trainable decay rates. TODO: enable convolutional decay."""
    name = prefix + '-decay'
    input_area = np.prod(input_shape_4d[1:])
    input_1d = tf.reshape(input_4d, shape=[self._hparams.batch_size, input_area])

    kernel_initializer, bias_initializer = None, None
    use_bias = True
    decay_rates = tf.layers.dense(
        inputs=input_1d, units=input_area,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        activation=tf.nn.sigmoid, use_bias=use_bias, name=name)

    # Obtain reference to Variable:
    # with tf.variable_scope(name, reuse=True):
    #   self._weights_d = tf.get_variable('kernel')
    #   if use_bias:
    #     self._bias_d = tf.get_variable('bias')

    # Limit decay rate
    logging.info('%s decay max at %f', prefix, max_decay_rate)
    decay_rates = decay_rates * max_decay_rate

    # Reshape to input shape
    decay_rates = tf.reshape(decay_rates, input_shape_4d)
    return decay_rates

  def _build_trace(self, prefix, input_4d, input_shape_4d, decay_rate, decay_floor, decay_trainable, max_decay_rate):
    """Decay and integrate an input to produce a trace over time"""
    if decay_rate > 0.0:
      trace_key = self.get_trace_key(prefix)
      trace_pl = self._dual.add(trace_key, shape=input_shape_4d, default_value=0.0).add_pl(default=True)

      # Trainable or constant decay rate
      if decay_trainable is True:
        logging.info('%s decay trainable', prefix)
        # 4d tensor same shape as input
        decay_rates = self._build_trainable_decay(prefix, trace_pl, input_shape_4d, max_decay_rate)
      else:
        # A scalar
        logging.info('%s decay @ rate %f', prefix, decay_rate)
        decay_rates = decay_rate

      # Decay with optional floor
      if decay_floor > 0.0:
        logging.info('%s decay floor at %f', prefix, decay_floor)
        decayed_temp = trace_pl * decay_rates
        threshold = tf.to_float(decayed_temp > decay_floor)
        decayed = decayed_temp * threshold
      else:
        decayed = trace_pl * decay_rates

      # Include new input
      trace_op = tf.maximum(decayed, input_4d)
    else:
      logging.info('%s NO trace therefore no decay', prefix)
      trace_op = input_4d  # no integration... therefore no decay

    return trace_op  # Current value

  def _build_norm(self, prefix, input_4d, norm_type, norm_eps):
    """Normalize/scale the input."""
    # Define possible norm types
    norm_type_sum = 'sum'
    # TODO investigate alternative norm, e.g. frobenius norm:
    #frobenius_norm = tf.sqrt(tf.reduce_sum(tf.square(input_values_next), axis=[1, 2, 3], keepdims=True))
    # Layer norm..? There has to be a better norm than this. 
    # https://mlexplained.com/2018/11/30/an-overview-of-normalization-methods-in-deep-learning/
    # https://mlexplained.com/2018/01/10/an-intuitive-explanation-of-why-batch-normalization-really-works-normalization-in-deep-learning-part-1/

    # Interpret norm type:
    if norm_type is None:
      logging.info('%s NO norm', prefix)
      return input_4d

    if norm_type == norm_type_sum:
      logging.info('%s Sum norm', prefix)
      sum_input = tf.reduce_sum(input_4d, axis=[1, 2, 3], keepdims=True) + norm_eps
      norm_input_4d = tf.divide(input_4d, sum_input)
      return norm_input_4d

    logging.info('%s Not recognized norm, so none. ', prefix)
    return input_4d

  def _build_dropout(self, prefix, input_4d, keep_rate):
    if keep_rate < 1.0:
      logging.info('%s Dropout @ rate %f', prefix, keep_rate)
      keep_key = self.get_keep_rate_key(prefix)
      keep_pl = self._dual.add(keep_key, shape=(), default_value=1.0).add_pl(default=True)
      dropout_4d = tf.nn.dropout(input_4d, keep_pl)  # Note, a scaling is applied
      return dropout_4d
    logging.info('%s NO dropout', prefix)
    return input_4d

  def _build_input(self, prefix, input_4d,
                   norm_type, norm_eps, keep_rate,
                   decay_rate, decay_floor, decay_trainable, max_decay_rate):
    """Builds the ops to manage an input. The stages are: Norm, Trace, and Dropout."""
    input_shape_4d = input_4d.get_shape().as_list()

    if self._hparams.input_norm_first:
      # 1. Norm the integrated trace
      norm_4d = self._build_norm(prefix, input_4d, norm_type, norm_eps)

      # 2. Build a decaying trace from the normed input
      trace_4d = self._build_trace(prefix, norm_4d, input_shape_4d, decay_rate, decay_floor, decay_trainable, max_decay_rate)
      final_4d = trace_4d
    else:
      # 1. Build a decaying trace from input
      trace_4d = self._build_trace(prefix, input_4d, input_shape_4d, decay_rate, decay_floor, decay_trainable, max_decay_rate)

      # 2. Norm the integrated trace
      norm_4d = self._build_norm(prefix, trace_4d, norm_type, norm_eps)
      final_4d = norm_4d

    # 3. Store the updated trace:
    trace_key = self.get_trace_key(prefix)
    self._dual.set_op(trace_key, final_4d)

    # 4. Make a sample from the normed trace with optional dropout
    sample_4d = self._build_dropout(prefix, final_4d, keep_rate)

    summarize_traces = False
    if summarize_traces:
      self._dual.set_op(prefix+'-in', input_4d)
      self._dual.set_op(prefix+'-int', trace_4d)
      self._dual.set_op(prefix+'-norm', norm_4d)
      self._dual.set_op(prefix+'-dropo', sample_4d)
    return sample_4d

  def get_target(self):
    target = self._input_values  # Current input
    target_shape = self._input_shape
    if self._hparams.mode == self.mode_predict_target:
      target = self._target_values
      target_shape = self._target_shape
    target_shape[0] = self._hparams.batch_size
    return target, target_shape

  def _build(self):
    """Build the autoencoder network"""

    num_dendrites = self.get_num_dendrites()

    if self.use_freq():
      num_cells = self._hparams.cols * self._hparams.cells_per_col
      self._dual.add(self.freq_col, shape=[self._hparams.cols], default_value=0.0).add_pl()
      self._dual.add(self.freq_cell, shape=[num_cells], default_value=0.0).add_pl()
      self._dual.add(self.usage_cell, shape=[num_cells], default_value=0.0).add_pl()
      self._dual.add(self.usage_col, shape=[self._hparams.cols], default_value=0.0).add_pl()

    self._dual.add(self.lifetime_mask, shape=[num_dendrites], default_value=1.0).add_pl(dtype=tf.bool)

    # ff input update - we work with the PREVIOUS ff input. This code stores the current input for access next time.
    input_shape_list = self._input_values.get_shape().as_list()
    previous_pl = self._dual.add(self.previous, shape=input_shape_list, default_value=0.0).add_pl()
    self._dual.set_op(self.previous, self._input_values, override=True)

    # External input and learning target
    target, target_shape = self.get_target()

    f_input = previous_pl  # ff_input = x_ff(t-1)
    if self._hparams.mode == self.mode_encode_input:  # Autoencode
      f_input = target  # ff_input = x_ff(t)

    with tf.name_scope('encoding'):

      # Forward input
      #f_trace = self._build_ff_conditioning(f_input)
      f_trace = self._build_input(prefix='f',
                                  input_4d=f_input,
                                  norm_type=self._hparams.f_norm_type,
                                  norm_eps=self._hparams.f_norm_eps,
                                  keep_rate=self._hparams.f_keep_rate,
                                  decay_rate=self._hparams.f_decay_rate,
                                  decay_floor=self._hparams.f_decay_floor,
                                  decay_trainable=self._hparams.f_decay_trainable,
                                  max_decay_rate=self._hparams.f_decay_rate_max)

      # Recurrent input
      r_input_shape = SequenceMemoryLayer.get_encoding_shape_4d(input_shape_list, self._hparams)
      r_pl = self._dual.add(self.recurrent, shape=r_input_shape, default_value=0.0).add_pl(default=True)

      #unit_recurrent, unit_recurrent_dropout = self._build_fb_conditioning(r_pl)
      #r_trace = unit_recurrent_dropout
      r_trace = self._build_input(prefix='r',
                                  input_4d=r_pl,
                                  norm_type=self._hparams.rb_norm_type,
                                  norm_eps=self._hparams.rb_norm_eps,
                                  keep_rate=self._hparams.rb_keep_rate,
                                  decay_rate=self._hparams.rb_decay_rate,
                                  decay_floor=self._hparams.rb_decay_floor,
                                  decay_trainable=self._hparams.rb_decay_trainable,
                                  max_decay_rate=self._hparams.rb_decay_rate_max)

      # Backward input
      b_trace = None
      if self.use_feedback():
        # Note feedback will be integrated in other layer, if that's a thing.
        b_input_shape = self._feedback_shape
        b_pl = self._dual.add(self.feedback, shape=b_input_shape, default_value=0.0).add_pl(default=True)

        # Interpolate the feedback to the conv w,h of the current layer
        b_interpolated_size = [r_input_shape[1], r_input_shape[2]]  # note h,w order
        b_interpolated = tf.image.resize_bilinear(b_pl, b_interpolated_size)

        #_, unit_feedback_dropout = self._build_fb_conditioning(feedback_interpolated)
        #b_trace = unit_feedback_dropout
        b_trace = self._build_input(prefix='b',
                                    input_4d=b_interpolated,
                                    norm_type=self._hparams.rb_norm_type,
                                    norm_eps=self._hparams.rb_norm_eps,
                                    keep_rate=self._hparams.rb_keep_rate,
                                    decay_rate=self._hparams.rb_decay_rate,
                                    decay_floor=self._hparams.rb_decay_floor,
                                    decay_trainable=self._hparams.rb_decay_trainable,
                                    max_decay_rate=self._hparams.rb_decay_rate_max)

      # Encoding of dendrites (weighted sums)
      # y(t) = f x_ff(t-1), y(t-1)
      f_encoding_cells_5d, r_encoding_cells_5d, b_encoding_cells_5d = self._build_encoding(f_trace, r_trace, b_trace)

      # # Inhibition placeholder
      # cells_5d_shape = [r_input_shape[0], r_input_shape[1], r_input_shape[2],
      #                   self._hparams.cols, self._hparams.cells_per_col]
      # self._dual.add(self.inhibition, shape=cells_5d_shape, default_value=0.0).add_pl()

      # Sparse masking, and integration of dendrites
      filtered_cells_5d = self._build_filtering(
          f_encoding_cells_5d, r_encoding_cells_5d, b_encoding_cells_5d)  # output y(t) | x_ff(t-1), y(t-1)

      # Inference output fork, doesn't accumulate gradients
      output_encoding_cells_5d = tf.stop_gradient(filtered_cells_5d)  # this is the encoding op
      output_encoding_cells_4d = self._build_cells_5d_to_cells_4d(output_encoding_cells_5d)
      # output_encoding_cells_5d_shape = output_encoding_cells_5d.get_shape().as_list()
      # output_encoding_cells_4d = tf.reshape(output_encoding_cells_5d,
      #                                       [-1, output_encoding_cells_5d_shape[1], output_encoding_cells_5d_shape[2],
      #                                        output_encoding_cells_5d_shape[3] * output_encoding_cells_5d_shape[4]])

      # NOTE: accumulated recurrent is WITHOUT dropout, so different selections can be made.
      #output_integrated_cells_4d = self._build_decay_and_integrate(unit_recurrent, output_encoding_cells_4d)
      #self._dual.set_op(self.encoding, output_integrated_cells_4d)  # if feedback decay is 0, then == current encoding
      self._dual.set_op(self.encoding, output_encoding_cells_4d)  # Encoding is current spikes only, now. No integration.

      # DEPRECATED
      # Generate inputs and targets for prediction
      # -----------------------------------------------------------------
      # with tf.name_scope('prediction'):

      #   # Generate what would be the feedback input next time - ie apply the norm
      #   if self._hparams.predictor_integrate_input:
      #     prediction_encoding = output_integrated_cells_4d  # WITH integration and decay
      #   else:
      #     prediction_encoding = output_encoding_cells_4d  # WITHOUT integration and decay

      #   # Norm prediction input
      #   unit_prediction_encoding = self._build_sum_norm(prediction_encoding, self._hparams.predictor_norm_input)

      #   # Input values for prediction (no dropout)
      #   prediction_input_values = unit_prediction_encoding  # predict t | y_t = f(t-1, t-2, ... )
      #   prediction_input_shape = recurrent_input_shape  # Default

      #   # Set these ops/properties:
      #   self._dual.set_op(self.prediction_input, prediction_input_values)
      #   self._dual.get(self.prediction_input).set_shape(prediction_input_shape)
      # DEPRECATED

    # Decode: predict x_ff(t) | y(t)
    # -----------------------------------------------------------------
    # NB: y(t) doesnt depend on x_ff(t)
    # y(t) = f( x_ff(t-1) , y(t-1) OK
    with tf.name_scope('decoding'):
      # TODO Consider removing this bottleneck to allow diff cells in a col to predict different cols.
      output_encoding_cols_4d = tf.reduce_max(filtered_cells_5d, axis=-1)

      # decode: predict x_t given y_t where
      # y_t = encoding of x_t-1 and y_t-1
      decoding, decoding_logits = self._build_decoding(target_shape, output_encoding_cols_4d)
      self._dual.set_op(self.decoding, decoding) # This is the thing that's optimized by the memory itself
      self._dual.set_op(self.decoding_logits, decoding_logits) # This is the thing that's optimized by the memory itself

      # Debug the per-sample prediction error inherent to the memory.
      prediction_error = tf.abs(target - decoding)
      #sum_abs_error = tf.reduce_sum(prediction_error, axis=[1, 2, 3])
      sum_abs_error = tf.reduce_sum(prediction_error)
      self._dual.set_op(self.sum_abs_error, sum_abs_error)

    # Managing history
    # -----------------------------------------------------------------
    self._build_history_update()

  def _build_encoding(self, f_input, r_input_cells_4d, b_input_cells_4d):  # pylint: disable=W0221
    """Build the encoder"""

    # Forward
    f_encoding_cells_5d = None
    with tf.name_scope('forward'):
      f_encoding_cols_4d = self._build_forward_encoding(self._input_shape, f_input)  #, self._hparams.cols)
      f_encoding_cells_5d = self._build_cols_4d_to_cells_5d(f_encoding_cols_4d)  # Old way - tiled
      #f_encoding_cells_5d = self._build_cols_4d_to_cols_5d(f_encoding_cols_4d)  # New way - broadcast
      self._dual.set_op(self.encoding_f, f_encoding_cells_5d)

    num_cells = self.get_num_cells()
    shape_cells_5d = f_encoding_cells_5d.get_shape().as_list()
    shape_cells_4d = [shape_cells_5d[0], shape_cells_5d[1], shape_cells_5d[2], num_cells]

    # Recurrent
    r_encoding_cells_5d = None
    with tf.name_scope('recurrent'):
      r_encoding_cells_4d = self._build_recurrent_encoding(shape_cells_4d, r_input_cells_4d)
      r_encoding_cells_5d = self._build_cells_4d_to_cells_5d(r_encoding_cells_4d)
      self._dual.set_op(self.encoding_r, r_encoding_cells_5d)

    # Feedback
    b_encoding_cells_5d = None
    with tf.name_scope('feedback'):
      if self.use_feedback():
        feedback_depth = self._feedback_shape[3]
        shape_feedback_4d = [shape_cells_5d[0], shape_cells_5d[1], shape_cells_5d[2], feedback_depth]
        b_encoding_cells_4d = self._build_feedback_encoding(shape_feedback_4d, b_input_cells_4d)
        b_encoding_cells_5d = self._build_cells_4d_to_cells_5d(b_encoding_cells_4d)
        self._dual.set_op(self.encoding_b, b_encoding_cells_5d)

    return f_encoding_cells_5d, r_encoding_cells_5d, b_encoding_cells_5d

  def _build_initializers(self, prefix, w_init_type, b_init_type, initial_sd):
    """Override to change initializer for particular Variables. Scope allows different behaviour for specific Variable."""
    # # Initializer has a mean of 0.0
    # #kernel_initializer = tf.truncated_normal_initializer(stddev=initial_sd)  # Older
    # kernel_initializer = tf.random_normal_initializer(stddev=initial_sd)
    # bias_initializer = kernel_initializer

    # # NB: Perhaps should init biases to zero- at least for decoding?
    # # if scope == 'decoding':
    # #   bias_initializer = tf.zeros_initializer
    logging.info('Variable prefix "%s" has W init type: "%s" and b init type: "%s" (sd=%f)', prefix, w_init_type, b_init_type, initial_sd)

    w_init = tf_get_kernel_initializer(init_type=w_init_type, initial_sd=initial_sd)
    if b_init_type is not None:
      b_init = tf_get_kernel_initializer(init_type=b_init_type, initial_sd=initial_sd)
    else:
      b_init = None
    return w_init, b_init

  def _build_conv_encoding(self, input_shape, input_tensor, field_stride, field_height, field_width, output_depth,
                           w_init_type, b_init_type, 
                           add_bias, initial_sd, scope):
    """Build encoding ops for the forward path."""
    input_depth = input_shape[3]
    strides = [1, field_stride, field_stride, 1]
    conv_variable_shape = [
        field_height,
        field_width,
        input_depth,
        output_depth  # number of filters
    ]

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE, auxiliary_name_scope=False):

      kernel_initializer, bias_initializer = self._build_initializers(scope, w_init_type, b_init_type, initial_sd)
      weights = tf.get_variable(shape=conv_variable_shape,
                                initializer=kernel_initializer, name='kernel')

      convolved = tf.nn.conv2d(input_tensor, weights, strides, padding='SAME', name='convolved') # zero padded

      bias = None
      if add_bias:
        # Note: bias has nonzero default initializer if enabled
        bias = tf.get_variable(shape=[output_depth], name='bias', initializer=bias_initializer, trainable=True)
        convolved = tf.nn.bias_add(convolved, bias)

      return convolved, weights, bias

  def _build_forward_encoding(self, input_shape, input_tensor):
    """Build encoding ops for the forward path."""
    scope = 'forward'
    output_depth = self._hparams.cols
    field_stride = self._hparams.filters_field_stride
    field_height = self._hparams.filters_field_height
    field_width = self._hparams.filters_field_width
    add_bias = self._hparams.f_bias
    w_init_type = self._hparams.f_init_type
    b_init_type = self._hparams.f_bias_init_type
    initial_sd = self._hparams.f_init_sd  #0.03
    convolved, self._weights_f, self._bias_f = self._build_conv_encoding(input_shape, input_tensor, field_stride,
                                                                         field_height, field_width, output_depth,
                                                                         w_init_type, b_init_type,
                                                                         add_bias, initial_sd, scope)
    return convolved

  def _build_recurrent_encoding(self, input_shape, input_tensor):
    """Build encoding ops for the forward path."""
    scope = 'recurrent'
    output_depth = self._hparams.cols * self._hparams.cells_per_col
    field_stride = 1  # i.e. feed to itself
    field_height = 1  # i.e. same position
    field_width = 1  # i.e. same position
    add_bias = self._hparams.r_bias
    w_init_type = self._hparams.r_init_type
    b_init_type = self._hparams.r_bias_init_type
    initial_sd = self._hparams.r_init_sd  # Not previously defined
    convolved, self._weights_r, self._bias_r = self._build_conv_encoding(input_shape, input_tensor, field_stride,
                                                                         field_height, field_width, output_depth,
                                                                         w_init_type, b_init_type,
                                                                         add_bias, initial_sd, scope)
    return convolved

  def _build_feedback_encoding(self, input_shape, input_tensor):
    """Build encoding ops for the forward path."""
    scope = 'feedback'
    output_depth = self._hparams.cols * self._hparams.cells_per_col
    field_stride = 1  # i.e. feed to itself
    field_height = 1  # i.e. same position
    field_width = 1  # i.e. same position
    add_bias = self._hparams.b_bias
    w_init_type = self._hparams.b_init_type
    b_init_type = self._hparams.b_bias_init_type
    initial_sd = self._hparams.b_init_sd  # Not previously defined
    convolved, self._weights_b, self._bias_b = self._build_conv_encoding(input_shape, input_tensor, field_stride,
                                                                         field_height, field_width, output_depth,
                                                                         w_init_type, b_init_type,
                                                                         add_bias, initial_sd, scope)
    return convolved

  def _build_cols_mask(self, h, w, ranking_input_cols_4d):
    """Build columns mask."""

    # ---------------------------------------------------------------------------------------------------------------
    # TOP-k RANKING
    # ---------------------------------------------------------------------------------------------------------------
    k = int(self._hparams.sparsity)
    batch_size = self._hparams.batch_size
    hidden_size = self._hparams.cols
    top_k_mask_cols_4d = tf_build_top_k_mask_4d_op(ranking_input_cols_4d, k, batch_size, h, w, hidden_size)

    # ---------------------------------------------------------------------------------------------------------------
    # Combine the two masks - lifetime sparse, and top-k-excluding-threshold
    # ---------------------------------------------------------------------------------------------------------------
    if self._hparams.lifetime_sparsity_cols:
      # ---------------------------------------------------------------------------------------------------------------
      # TOP-1 over cells - lifetime sparsity [in winning Cols only.]
      # ---------------------------------------------------------------------------------------------------------------
      # At each batch, h, w, and col, find the top-1 cell.
      # Don't reduce over cells - dendrite is only valid for its associated cell.
      top_1_input_cols_4d = tf.reduce_max(ranking_input_cols_4d, axis=[0, 1, 2], keepdims=True)
      top_1_mask_bool_cols_4d = tf.greater_equal(ranking_input_cols_4d, top_1_input_cols_4d)
      top_1_mask_cols_4d = tf.to_float(top_1_mask_bool_cols_4d)

      # ---------------------------------------------------------------------------------------------------------------
      # Exclude naturally winning cols from the lifetime sparsity mask.
      # ---------------------------------------------------------------------------------------------------------------
      top_k_inv_cols_4d = 1.0 - tf.reduce_max(top_k_mask_cols_4d, axis=[0, 1, 2], keepdims=True)
      top_1_mask_cols_4d = tf.multiply(top_1_mask_cols_4d, top_k_inv_cols_4d) # removes lifetime bits from winning cols

      either_mask_cols_4d = tf.maximum(top_k_mask_cols_4d, top_1_mask_cols_4d)  # OR(a, b)
    else:
      either_mask_cols_4d = top_k_mask_cols_4d

    either_mask_cols_5d = self._build_cols_4d_to_cells_5d(either_mask_cols_4d)

    return either_mask_cols_5d

  def _build_tiebreaker_masks(self, h, w):
    """Build tiebreaker masks."""

    # Deal with the special case that there's no feedback and all the cells and dends are equal
    # Build a special mask to select ONLY the 1st cell/dend in each col, in the event that all the inputs are identical
    shape_cells_5d = [self._hparams.batch_size, h, w, self._hparams.cols, self._hparams.cells_per_col]

    np_cells_5d_1st = np.zeros(shape_cells_5d) # 1 for first cell / dendrite in the col
    np_cells_5d_all = np.ones(shape_cells_5d)

    for b in range(shape_cells_5d[0]):
      for y in range(shape_cells_5d[1]):
        for x in range(shape_cells_5d[2]):
          for col in range(shape_cells_5d[3]):
            np_cells_5d_1st[b][y][x][col][0] = 1 # 1st cell+dend

    self._mask_1st_cells_5d = tf.constant(np_cells_5d_1st, dtype=tf.float32)
    self._mask_all_cells_5d = tf.constant(np_cells_5d_all, dtype=tf.float32)

  def _build_cells_mask(self, h, w, ranking_input_cells_5d, mask_cols_5d):
    """Build top-1 cells per col mask. A simpler form without tiebreakers or lifetime features."""

    # Find best cell in each col
    # 0 1 2 3   4
    # b,h,w,col,cell
    max_cells_5d = tf.reduce_max(ranking_input_cells_5d, axis=[4], keepdims=True)
    best_cells_5d_bool = tf.greater_equal(ranking_input_cells_5d, max_cells_5d)  # =1 if bit is best in the col
    best_cells_5d = tf.to_float(best_cells_5d_bool)
    mask_cells_5d = mask_cols_5d * best_cells_5d  # best-in-col AND topk-cols
    return mask_cells_5d

  def _build_dendrite_mask(self, h, w, ranking_input_cells_5d, mask_cols_5d, lifetime_sparsity_dends,
                           lifetime_mask_dend_1d):
    """Build dendrite mask (from when we had many dends per cell)."""
    del h, w

    # Find the max value in each col, by reducing over cells and dends
    max_cells_5d = tf.reduce_max(ranking_input_cells_5d, axis=[4], keepdims=True) # 1 if bit is best in the col
    min_cells_5d = tf.reduce_min(ranking_input_cells_5d, axis=[4], keepdims=True)
    rng_cells_5d = max_cells_5d - min_cells_5d

    # Tie-breaking using special masks
    # Detect that the input is all identical, therefore dends have identical response
    zero_rng_cells_5d_bool = tf.equal(rng_cells_5d, 0.0) # if zero, then 1
    zero_rng_cells_5d_bool = tf.tile(zero_rng_cells_5d_bool, [1, 1, 1, 1, self._hparams.cells_per_col])
    if0_mask_cells_5d_bool = tf.where(zero_rng_cells_5d_bool, self._mask_1st_cells_5d, self._mask_all_cells_5d)

    # Below: Calculate the best dendrite of all the dendrites on all the cells for this Col.
    # i.e. the "winner" Dend of the Col
    best_cells_5d_bool = tf.greater_equal(ranking_input_cells_5d, max_cells_5d)

    # Exclude cells from cols that weren't selected
    masked_ranking_input_cells_5d = ranking_input_cells_5d * mask_cols_5d

    # Lifetime sparsity
    if lifetime_sparsity_dends:
      top_1_max_cells_5d = tf.reduce_max(masked_ranking_input_dend_6d, axis=[0, 1, 2], keepdims=True)
      top_1_cells_5d_bool = tf.greater_equal(masked_ranking_input_cells_5d, top_1_max_cells_5d)

      # exclude dends with actual uses from lifetime training
      shape_cells_5d_111 = [1, 1, 1, self._hparams.cols, self._hparams.cells_per_col]

      lifetime_mask_cells_5d = tf.reshape(lifetime_mask_dend_1d, shape=shape_cells_5d_111)
      top_1_cells_5d_bool = tf.logical_and(top_1_cells_5d_bool, lifetime_mask_cells_5d)

      top1_best_cells_5d_bool = tf.logical_or(top_1_cells_5d_bool, best_cells_5d_bool)
    else:
      top1_best_cells_5d_bool = best_cells_5d_bool

    # Now combine this mask with the Top-k cols mask, giving a combined mask:
    mask_cols_5d_bool = tf.greater(mask_cols_5d, 0.0) # Convert to bool
    mask_cells_5d_bool = tf.logical_and(top1_best_cells_5d_bool, mask_cols_5d_bool)
    mask_cells_5d = tf.to_float(mask_cells_5d_bool)

    edge_cells_5d_bool = mask_cells_5d * if0_mask_cells_5d_bool

    return edge_cells_5d_bool

  def _build_cols_4d_to_cells_5d(self, cols_4d):
    """[b, h,w, cols]=4d ---> [b, h,w, cols,cells]=5d"""
    cols_5d = tf.expand_dims(cols_4d, -1)
    cells_5d = tf.tile(cols_5d, [1, 1, 1, 1, self._hparams.cells_per_col])
    return cells_5d

  def _build_cols_4d_to_cols_5d(self, cols_4d):
    """[b, h,w, cols]=4d ---> [b, h,w, cols,1]=5d"""
    cols_5d = tf.expand_dims(cols_4d, -1)
    return cols_5d

  def _build_cells_4d_to_cells_5d(self, cells_4d):
    """[b, h,w, (cols*cells)]=4d ---> [b, h,w, cols,cells]=5d"""
    cells_4d_shape = cells_4d.get_shape().as_list()
    cells_5d_shape = [cells_4d_shape[0], cells_4d_shape[1], cells_4d_shape[2], self._hparams.cols,
                      self._hparams.cells_per_col]
    cells_5d = tf.reshape(cells_4d, cells_5d_shape)
    return cells_5d

  def _build_cells_5d_to_cells_4d(self, cells_5d):
    """[b, h,w, cols,cells]=5d ---> [b, h,w, (cols*cells)]=4d"""
    cells_5d_shape = cells_5d.get_shape().as_list()
    cells_4d = tf.reshape(cells_5d,
                          [-1, cells_5d_shape[1], cells_5d_shape[2],
                           cells_5d_shape[3] * cells_5d_shape[4]])
    return cells_4d

  def _build_inhibition(self, i_encoding_cells_5d):
    """Applies inhibition to the specified encoding."""

    # ---------------------------------------------------------------------------------------------------------------
    # INHIBITION - retard the activation of dendrites that fired recently.
    # inhibition shape = [b,h,w,col,cell,d]
    # ---------------------------------------------------------------------------------------------------------------
    # Refractory period
    # Inhibition is max (1) and decays to min (0)
    # So refractory weight is 1-1=0, returning to 1 over time.
    inhibition_cells_5d_pl = self._dual.get_pl(self.inhibition)
    refractory_cells_5d = 1.0 - inhibition_cells_5d_pl # ie inh.=1, ref.=0 . Inverted

    # Pos encoding (only used for ranking)
    # Find min in each batch sample
    # This is wrong for conv.? Should be min in each conv location.
    # But - it doesn't matter?
    min_i_encoding_cells_5d = tf.reduce_min(i_encoding_cells_5d, axis=[1, 2, 3, 4], keepdims=True) # may be negative
    pos_i_encoding_cells_5d = (i_encoding_cells_5d - min_i_encoding_cells_5d) + 1.0 # shift to +ve range, ensure min value is nonzero

    # Optional hidden dropout
    if self._hparams.hidden_keep_rate < 1.0:
      hidden_keep_pl = self._dual.add(self.hidden_keep, shape=(), default_value=1.0).add_pl(default=True)
      dropout_cells_5d = tf.nn.dropout(refractory_cells_5d, hidden_keep_pl)  # Note, a scaling is applied
    else:
      dropout_cells_5d = refractory_cells_5d

    # Apply inhibition/refraction
    #refracted_cells_5d = pos_i_encoding_cells_5d * refractory_cells_5d
    refracted_cells_5d = pos_i_encoding_cells_5d * dropout_cells_5d

    return refracted_cells_5d

  def _update_boost(self, batch_type, session):
    """Assign the boost variable after a frequency update. Only in training mode."""
    if batch_type != self.training:
      #print('Not updating boost - wrong batch type: ', batch_type)
      return
    #print('YES updating boost - training batch type: ', batch_type)

    # Suspend boost changes when training has finished.
    do_training = tf_do_training(batch_type, self._hparams.training_interval, self._training_batch_count, name=self.name)
    if not do_training:
      return

    freq_cell = self._dual.get(self.freq_cell)
    freq_cell_pl = freq_cell.get_pl()  # [cols * cells_per_col] 1d
    freq_cell_values = freq_cell.get_values()

    feed_dict = {
        freq_cell_pl: freq_cell_values
    }

    boost_a = self.boost_assign
    fetches = {
        boost_a: self._dual.get_op(boost_a)
    }

    # Update the variable
    fetched = session.run(fetches, feed_dict=feed_dict)

    # Off graph copy of boost values
    boost_values = fetched[boost_a]
    self._dual.set_values(self.boost, boost_values)
    #print('New boost: ', boost_values)

  def _build_update_boost(self):
    """Builds the boost Variable an an assign op to be used periodically""" 
    boost_factor_pl = self._dual.add(self.boost_factor, shape=(), default_value=self._hparams.boost_factor).add_pl(default=True)
    freq_cell_pl = self._dual.get_pl(self.freq_cell)  # [cols * cells_per_col] 1d
    num_cells = self._hparams.cols * self._hparams.cells_per_col
    freq_target = self._hparams.sparsity / num_cells  # k/n where n is num cells
    # e.g. cols=600 cell/col=3 num_cells=1800
    # sparsity = k = 95
    # 95/1800 = 0.052777778  ie 5%
    # Say freq = 0
    # exp(0.052-0) = 1.053375743
    # Numenta code: 
    #  boost_factors = tf.exp((target_density - duty_cycles) * boost_strength)
    boost_cells_1d = tf.math.exp((freq_target - freq_cell_pl) * boost_factor_pl)
    boost_shape_1d = [num_cells]
    boost_values = np.ones(num_cells)
    # boost_shape_1d = [num_cells]
    # boost_v = tf.Variable(initial_value=boost_values, shape=boost_shape_1d, trainable=False, dtype=tf.float32)
    boost_v = tf.Variable(initial_value=boost_values, trainable=False, dtype=tf.float32)
    boost_a = boost_v.assign(boost_cells_1d)
    self._dual.set_op(self.boost_assign, boost_a)
    self._dual.set_op(self.boost, boost_v)

  def _build_boosting(self, i_encoding_cells_5d):
    """Builds boost-related features"""
    self._build_update_boost()
    boost_cells_1d = self._dual.get_op(self.boost)  # Retrieve the variable
    boost_shape_5d = [1, 1, 1, self._hparams.cols, self._hparams.cells_per_col]
    boost_cells_5d = tf.reshape(boost_cells_1d, boost_shape_5d)

    # Optional hidden dropout
    if self._hparams.hidden_keep_rate < 1.0:
      hidden_keep_pl = self._dual.add(self.hidden_keep, shape=(), default_value=1.0).add_pl(default=True)
      dropout_cells_5d = tf.nn.dropout(boost_cells_5d, hidden_keep_pl)  # Note, a scaling is applied
    else:
      dropout_cells_5d = boost_cells_5d

    boosted_cell_5d = i_encoding_cells_5d * dropout_cells_5d
    return boosted_cell_5d

  # The code below didn't allow the boost to be stored as a Variable for later reloading
  # def _build_boosting(self, i_encoding_cells_5d):
  #   boost_shape = [1, 1, 1, self._hparams.cols, self._hparams.cells_per_col]
  #   freq_cell_pl = self._dual.get_pl(self.freq_cell)  # [cols * cells_per_col] 1d
  #   num_cells = self._hparams.cols * self._hparams.cells_per_col
  #   freq_target = self._hparams.sparsity / num_cells  # k/n where n is num cells
  #   # e.g. cols=600 cell/col=3 num_cells=1800
  #   # sparsity = k = 95
  #   # 95/1800 = 0.052777778  ie 5%
  #   # Say freq = 0
  #   # exp(0.052-0) = 1.053375743
  #   boost_cells_1d = tf.math.exp(freq_target - freq_cell_pl) * self._hparams.boost_factor
  #   self._dual.set_op('boost', boost_cells_1d)
  #   boost_cells_5d = tf.reshape(boost_cells_1d, boost_shape)
  #   boosted_cell_5d = i_encoding_cells_5d * boost_cells_5d
  #   return boosted_cell_5d

  def use_boosting(self):
    """Returns True iff boost_factor > 0"""
    if self._hparams.boost_factor > 0.0:
      return True
    return False

  def _build_filtering(self, f_encoding_cells_5d, r_encoding_cells_5d, b_encoding_cells_5d):
    """Build filtering/masking for specified encoding."""

    # ---------------------------------------------------------------------------------------------------------------
    # Inhibition placeholder
    # ---------------------------------------------------------------------------------------------------------------
    cells_5d_shape = r_encoding_cells_5d.get_shape().as_list()
    h = cells_5d_shape[1]
    w = cells_5d_shape[2]

    #cells_5d_shape = [r_input_shape[0], r_input_shape[1], r_input_shape[2],
    #                  self._hparams.cols, self._hparams.cells_per_col]
    self._dual.add(self.inhibition, shape=cells_5d_shape, default_value=0.0).add_pl()

    # ---------------------------------------------------------------------------------------------------------------
    # INTEGRATE DENDRITES
    # ---------------------------------------------------------------------------------------------------------------
    i_encoding_cells_5d = f_encoding_cells_5d + r_encoding_cells_5d  # i = integrated
    if self.use_feedback():
      i_encoding_cells_5d = i_encoding_cells_5d + b_encoding_cells_5d


    # ---------------------------------------------------------------------------------------------------------------
    # RANKING METRIC
    # ---------------------------------------------------------------------------------------------------------------
    if self.use_boosting():
      ranking_input_cells_5d = self._build_boosting(i_encoding_cells_5d)
    else:
      ranking_input_cells_5d = self._build_inhibition(i_encoding_cells_5d)

    # reduce to cols
    # take best(=max) dendrite per cell, and best cell per col.
    #ranking_input_cells_5d = refracted_cells_5d
    ranking_input_cols_4d = tf.reduce_max(ranking_input_cells_5d, axis=[4])

    # ---------------------------------------------------------------------------------------------------------------
    # TOP-k RANKING (Cols)
    # ---------------------------------------------------------------------------------------------------------------
    mask_cols_5d = self._build_cols_mask(h, w, ranking_input_cols_4d)

    # ---------------------------------------------------------------------------------------------------------------
    # TOP-1 RANKING (Dend)
    # Reduce over dendrites AND cells to find the best dendrite at each b, h, w, col = 0,1,2,3
    # ---------------------------------------------------------------------------------------------------------------
    self._build_tiebreaker_masks(h, w)
    #lifetime_mask_dend_1d = self._dual.get_pl(self.lifetime_mask)
    #training_lifetime_sparsity_dends = self._hparams.lifetime_sparsity_dends
    #testing_lifetime_sparsity_dends = False
    # training_mask_cells_5d = self._build_dendrite_mask(h, w, ranking_input_cells_5d, mask_cols_5d,
    #                                                    training_lifetime_sparsity_dends, lifetime_mask_dend_1d)
    # testing_mask_cells_5d = self._build_dendrite_mask(h, w, ranking_input_cells_5d, mask_cols_5d,
    #                                                   testing_lifetime_sparsity_dends, lifetime_mask_dend_1d)
    #mask_cells_5d = tf.stop_gradient(self._build_cells_mask(h, w, ranking_input_cells_5d, mask_cols_5d))
    mask_cells_5d = tf.stop_gradient(self._build_dendrite_mask(h, w, ranking_input_cells_5d, mask_cols_5d, False, False))
    #testing_mask_cells_5d = training_mask_cells_5d = mask_cells_5d
    #testing_mask_cells_5d = training_mask_cells_5d = mask_cols_5d
    self._dual.set_op(self.encoding_mask, mask_cells_5d)

    # Produce the final filtering with these masks
    filtered_cells_5d = self._build_nonlinearities(
        f_encoding_cells_5d, r_encoding_cells_5d, b_encoding_cells_5d, mask_cells_5d)

    #testing_filtered_cells_5d = training_filtered_cells_5d

    #if not self.use_boosting():
    self._build_update_inhibition(mask_cells_5d, filtered_cells_5d)
    #self._build_update_boosting(training_mask_cells_5d)

    # # Update cells' inhibition
    # inhibition_cells_5d = inhibition_cells_5d_pl * self._hparams.inhibition_decay # decay old inh
    # inhibition_cells_5d = tf.maximum(training_mask_cells_5d, inhibition_cells_5d)  # this should be per batch sample not only per dend
    # self._dual.set_op(self.inhibition, inhibition_cells_5d, override=True)

    # Update usage (test mask doesnt include lifetime bits)
    if self.use_freq():
      self._build_update_usage(mask_cells_5d)

    return filtered_cells_5d

  def _build_update_inhibition(self, mask_cells_5d, masked_cells_5d):
    # Update cells' inhibition
    inhibition_cells_5d_pl = self._dual.get_pl(self.inhibition)
    inhibition_cells_5d = inhibition_cells_5d_pl * self._hparams.inhibition_decay # decay old inh

    if self._hparams.inhibition_with_mask:
      # Inhibit with mask
      inhibition_cells_5d = tf.maximum(mask_cells_5d, inhibition_cells_5d)
    else:
      # Inhibit with value
      inhibition_cells_5d = tf.maximum(masked_cells_5d, inhibition_cells_5d)
    self._dual.set_op(self.inhibition, inhibition_cells_5d)

  # def _build_update_boosting(self, training_mask_cells_5d):
  #   # Update cells' boost given activity
  #   boost_cells_5d_pl = self._dual.get_pl(self.inhibition)
  #   self._dual.set_op(self.inhibition, boost_cells_5d, override=True)

  def _build_update_usage(self, mask_cells_5d):
    """Build graph op to update usage."""

    # ---------------------------------------------------------------------------------------------------------------
    # USAGE UPDATE - per cell
    # Reduce over the batch, w, and h.
    # Note that any training samples will be included, including lifetime sparsity
    # ---------------------------------------------------------------------------------------------------------------

    usage_col_1d = tf.reduce_sum(mask_cells_5d, axis=[0, 1, 2, 4]) # reduce over b,h,w: 0,1,2, 4 leaving col (3)
    usage_col_pl = self._dual.get(self.usage_col).get_pl()
    usage_col_op = usage_col_pl + usage_col_1d
    self._dual.set_op(self.usage_col, usage_col_op, override=True)

    num_cells = self._hparams.cols * self._hparams.cells_per_col
    usage_cell_2d = tf.reduce_sum(mask_cells_5d, axis=[0, 1, 2]) # reduce over b,h,w: 0,1,2 leaving col,cell (3,4)
    usage_cell_1d = tf.reshape(usage_cell_2d, shape=[num_cells])
    usage_cell_pl = self._dual.get(self.usage_cell).get_pl()
    usage_cell_op = usage_cell_pl + usage_cell_1d
    self._dual.set_op(self.usage_cell, usage_cell_op, override=True)

  def _build_nonlinearities(self, f_encoding_cells_5d, r_encoding_cells_5d, b_encoding_cells_5d, mask_cells_5d):
    """Mask the encodings, sum them, and apply nonlinearity. Don't backprop into the mask."""

    #mask_cells_5d_sg = tf.stop_gradient(mask_cells_5d)
    #self._dual.set_op(self.encoding_mask, mask_cells_5d_sg)

    # Apply masks
    #f_masked_cells_5d = f_encoding_cells_5d * mask_cells_5d
    #r_masked_cells_5d = r_encoding_cells_5d * mask_cells_5d

    # Combine forward and feedback encoding
    #hidden_sum_cells_5d = f_masked_cells_5d + r_masked_cells_5d
    hidden_sum_cells_5d = f_encoding_cells_5d + r_encoding_cells_5d

    if self.use_feedback():
      #b_masked_cells_5d = b_encoding_cells_5d * mask_cells_5d
      hidden_sum_cells_5d = hidden_sum_cells_5d + b_encoding_cells_5d

    # Nonlinearity (1)
    masked_sum_cells_5d = hidden_sum_cells_5d * mask_cells_5d

    # Nonlinearity (2)
    hidden_transfer_cells_5d, _ = activation_fn(masked_sum_cells_5d, self._hparams.hidden_nonlinearity)

    return hidden_transfer_cells_5d#, hidden_transfer_cells_5d

  def _build_dense_decoding(self, target_shape, hidden_tensor, name='decoding'):
    """Build a fully connected decoder."""

    target_area = np.prod(target_shape[1:])
    hidden_area = np.prod(hidden_tensor.get_shape()[1:])
    hidden_reshape = tf.reshape(hidden_tensor, shape=[self._hparams.batch_size, hidden_area])

    w_init_type = self._hparams.d_init_type
    b_init_type = self._hparams.d_bias_init_type
    kernel_initializer, bias_initializer = self._build_initializers(name, w_init_type, b_init_type, self._hparams.d_init_sd)

    # Make the decoding layer
    # Note: We use our own nonlinearity, elsewhere.
    use_bias = self._hparams.d_bias
    decoding_weighted_sum = tf.layers.dense(
        inputs=hidden_reshape, units=target_area,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        activation=None, use_bias=use_bias, name=name)

    with tf.variable_scope(name, reuse=True):
      self._weights_d = tf.get_variable('kernel')

      if use_bias:
        self._bias_d = tf.get_variable('bias')

    return decoding_weighted_sum

  def _build_conv_decoding(self, target_shape, hidden_tensor, name='decoding'):
    """Build a convolutional decoder."""

    strides = [1, self._hparams.filters_field_stride, self._hparams.filters_field_stride, 1]
    conv_variable_shape = [
        self._hparams.filters_field_height,
        self._hparams.filters_field_width,
        self._input_shape[3],
        self._hparams.cols  # number of filters
    ]

    deconv_shape = target_shape
    deconv_shape[0] = self._hparams.batch_size

    w_init_type = self._hparams.d_init_type
    b_init_type = self._hparams.d_bias_init_type
    kernel_initializer, bias_initializer = self._build_initializers(name, w_init_type, b_init_type, self._hparams.d_init_sd)

    with tf.variable_scope(name):
      self._weights_d = tf.get_variable(
          shape=conv_variable_shape,
          initializer=kernel_initializer, name='kernel')

      if self._hparams.d_bias:
        self._bias_d = tf.get_variable(
            shape=target_shape[1:],
            initializer=bias_initializer, name='bias')

      deconvolved = tf.nn.conv2d_transpose(
          hidden_tensor, self._weights_d, output_shape=deconv_shape,
          strides=strides, padding='SAME', name='deconvolved')
      logging.debug(deconvolved)

      # Reconstruction of the input, in 3d
      if self._hparams.d_bias:
        decoding_weighted_sum = tf.add(deconvolved, self._bias_d, name='deconvolved_biased')
      else:
        decoding_weighted_sum = deconvolved
      logging.debug(decoding_weighted_sum)

      return decoding_weighted_sum

  def _build_decoding(self, target_shape, hidden_tensor):  # pylint: disable=W0613
    """Build the decoder (optionally without using tied weights)"""

    decode_layer_name = 'decoding'

    # Build a decoder with untied weights
    # -----------------------------------------------------------------
    if self._hparams.decode_mode == 'conv':
      decoding_weighted_sum = self._build_conv_decoding(target_shape, hidden_tensor, decode_layer_name)
    elif self._hparams.decode_mode == 'fc':
      decoding_weighted_sum = self._build_dense_decoding(target_shape, hidden_tensor, decode_layer_name)
    else:
      raise NotImplementedError('Decoding mode not supported: ' + self._hparams.decode_mode)

    decode_transfer, _ = activation_fn(decoding_weighted_sum, self._hparams.decode_nonlinearity)
    decoding_logits_reshape = tf.reshape(decoding_weighted_sum, target_shape)
    decoding_transfer_reshape = tf.reshape(decode_transfer, target_shape)

    return decoding_transfer_reshape, decoding_logits_reshape

  def _build_l2_loss(self, dendrite, w, b, scale, other_losses):
    """Builds an L2 loss op for specified parameters for regularization."""
    if scale <= 0.0:
      logging.info('Dendrite: %s. Skipping, L2=0.', dendrite)
      return other_losses

    if w is None:
      logging.info('Dendrite: %s. Skipping, W=None.', dendrite)
      return other_losses

    logging.info('Dendrite: %s. Adding L2 loss (W): %s', dendrite, str(scale))
    l2_loss_w = tf.nn.l2_loss(w)
    if b is not None:
      logging.info('Dendrite: %s. Adding L2 loss (b): %s', dendrite, str(scale))
      l2_loss_b = tf.nn.l2_loss(b)
      l2_loss = tf.reduce_sum(l2_loss_w) + tf.reduce_sum(l2_loss_b)
    else:
      l2_loss = tf.reduce_sum(l2_loss_w)

    l2_loss_scaled = l2_loss * scale

    if other_losses is None:
      all_losses = l2_loss_scaled
    else:
      all_losses = other_losses + l2_loss_scaled
    return all_losses

  def _build_extra_losses(self):
    """Defines some extra criteria to optimize - in this case regularization"""
    l2_losses = None
    l2_losses = self._build_l2_loss('f', self._weights_f, self._bias_f, self._hparams.f_l2, l2_losses)
    l2_losses = self._build_l2_loss('r', self._weights_r, self._bias_r, self._hparams.r_l2, l2_losses)
    l2_losses = self._build_l2_loss('b', self._weights_b, self._bias_b, self._hparams.b_l2, l2_losses)
    l2_losses = self._build_l2_loss('d', self._weights_d, self._bias_d, self._hparams.d_l2, l2_losses)
    return l2_losses

  def _build_optimizer(self):
    """Setup the training operations"""
    #target = self._input_values
    target, _ = self.get_target()
    prediction = self.get_op(self.decoding)
    prediction_logits = self.get_op(self.decoding_logits)
    with tf.variable_scope('optimizer', reuse=tf.AUTO_REUSE):
      self._optimizer = tf_create_optimizer(self._hparams)

      prediction_loss_op = self._build_loss_fn(target, prediction, prediction_logits)
      extra_loss_op = self._build_extra_losses()  # e.g. regularization
      loss_op = prediction_loss_op
      if extra_loss_op is not None:
        all_losses = []
        all_losses.append(prediction_loss_op)
        all_losses.append(extra_loss_op)
        loss_op = tf.add_n(all_losses)
      self._dual.set_op(self.loss, loss_op)

      # Does the counter for the optimizer step affect momentum calculations in Adam?
      # Apparently not... 
      # It's own step (accurate counting)
      #step_variable = tf.Variable(0, name='training_step', trainable=False)
      #training_op = self._optimizer.minimize(loss_op, global_step=step_variable, name='training_op')
      # Shared step (n* counting)
      training_op = self._optimizer.minimize(loss_op, global_step=tf.train.get_or_create_global_step(), name='training_op')
      self._dual.set_op(self.training, training_op)

  def _build_loss_fn(self, target, output, output_logits=None):
    """Build the loss function with specified type: mse, sigmoid-ce, softmax-ce."""
    if self._hparams.loss_type == 'mse':
      return tf.losses.mean_squared_error(target, output)
      # # Due to concern about the way the reduction is done in N-dimensional tensors, explicitly make it 1-d.
      # image_shape = target.shape.as_list()
      # image_size = np.prod(image_shape[1:])
      # vector_shape = [image_shape[0], image_size]
      # target_vector = tf.reshape(target, vector_shape)
      # output_vector = tf.reshape(output, vector_shape)
      # return tf.losses.mean_squared_error(target_vector, output_vector)
      #return tf.losses.mean_squared_error(target, output, reduction=Reduction.SUM_BY_NONZERO_WEIGHTS)
      #return tf.losses.mean_squared_error(target, output, reduction=tf.losses.Reduction.MEAN)

    if self._hparams.loss_type == 'sigmoid-ce':
      return tf.losses.sigmoid_cross_entropy(multi_class_labels=target, logits=output_logits)

    if self._hparams.loss_type == 'softmax-ce':
      return tf.losses.softmax_cross_entropy(onehot_labels=target, logits=output_logits)

    raise NotImplementedError('Loss function not implemented: ' + str(self._hparams.loss_type))

  # COMPONENT INTERFACE ------------------------------------------------------------------
  def update_feed_dict(self, feed_dict, batch_type='training'):
    """Update the feed dict."""

    # Update these dual entities in the normal manner
    # Feedback stays on-graph
    names = [self.inhibition, self.previous, self.recurrent, self.lifetime_mask]
    self._dual.update_feed_dict(feed_dict, names)

    #print('prev =-----------------------', self._dual.get_values(self.previous))

    # Optional frequency monitoring
    if self.use_freq():
      freq_names = [self.usage_cell, self.usage_col, self.freq_cell, self.freq_col]
      self._dual.update_feed_dict(feed_dict, freq_names)

    # Optional feedback
    if self.use_feedback():
      feedback = self._dual.get(self.feedback)
      feedback_pl = feedback.get_pl()
      feedback_values = feedback.get_values()
      feed_dict.update({
          feedback_pl: feedback_values
      })

    # Dendrite traces
    if self._hparams.f_decay_rate > 0.0:
      trace_key = self.get_trace_key('f')
      self._dual.update_feed_dict(feed_dict, [trace_key])

    if self._hparams.rb_decay_rate > 0.0:
      trace_key = self.get_trace_key('r')
      self._dual.update_feed_dict(feed_dict, [trace_key])

      if self.use_feedback():
        trace_key = self.get_trace_key('b')
        self._dual.update_feed_dict(feed_dict, [trace_key])

    # Adjust feedback keep rate by batch type
    f_keep_rate = 1.0
    rb_keep_rate = 1.0
    hidden_keep_rate = 1.0  # Encoding
    if batch_type == self.training:
      f_keep_rate = self._hparams.f_keep_rate  # Training
      rb_keep_rate = self._hparams.rb_keep_rate  # Training
      hidden_keep_rate = self._hparams.hidden_keep_rate  # Training
    logging.debug('Forward keep rate: %s', str(f_keep_rate))
    logging.debug('Recurrent keep rate: %s', str(rb_keep_rate))
    logging.debug('Backward keep rate: %s', str(rb_keep_rate))
    logging.debug('Hidden keep rate: %s', str(hidden_keep_rate))

    # Placeholder for feedback keep rate
    def update_feed_dict_keep_rate(self, prefix, keep_rate):
      keep_rate_key = self.get_keep_rate_key(prefix)
      keep = self._dual.get(keep_rate_key)
      keep_pl = keep.get_pl()
      feed_dict.update({
          keep_pl: keep_rate
      })

    if self._hparams.f_keep_rate < 1.0:
      update_feed_dict_keep_rate(self, 'f', f_keep_rate)
    if self._hparams.rb_keep_rate < 1.0:
      update_feed_dict_keep_rate(self, 'r', rb_keep_rate)
      if self.use_feedback():
        update_feed_dict_keep_rate(self, 'b', rb_keep_rate)
    if self._hparams.hidden_keep_rate < 1.0:
      hidden_keep = self._dual.get(self.hidden_keep)
      hidden_keep_pl = hidden_keep.get_pl()
      feed_dict.update({
          hidden_keep_pl: hidden_keep_rate
      })

  def add_fetches(self, fetches, batch_type='training'):
    """Adds ops that will get evaluated."""
    # Feedback stays on-graph
    names = [self.previous, self.inhibition,
             self.encoding, self.decoding, self.encoding_mask,
             self.sum_abs_error, self.loss]

    self._dual.add_fetches(fetches, names)

    if self.use_freq():
      freq_names = [self.usage_cell, self.usage_col]
      self._dual.add_fetches(fetches, freq_names)

    # Dendrite traces
    if self._hparams.f_decay_rate > 0.0:
      trace_key = self.get_trace_key('f')
      self._dual.add_fetches(fetches, [trace_key])

    if self._hparams.rb_decay_rate > 0.0:
      trace_key = self.get_trace_key('r')
      self._dual.add_fetches(fetches, [trace_key])

      if self.use_feedback():
        trace_key = self.get_trace_key('b')
        self._dual.add_fetches(fetches, [trace_key])

    do_training = tf_do_training(batch_type, self._hparams.training_interval, self._training_batch_count,
                                 name=self.name)
    if do_training:
      fetches[self.name].update({
          self.training: self._dual.get_op(self.training)
      })

    # Summaries
    super().add_fetches(fetches, batch_type)

  def set_fetches(self, fetched, batch_type='training'):
    """Store updated tensors"""
    if batch_type == self.training:
      self._training_batch_count += 1

    # Loss (not a tensor)
    self_fetched = fetched[self.name]
    self._loss = self_fetched[self.loss]

    ###########################################################################
    # Exact loss calculation over time, for comparison other impl.
    ###########################################################################
    # if batch_type == self.training:
    #   samples = 1000
    #   if self._loss_mean is None:
    #     self._loss_mean = []
    #   self._loss_mean.append(self._loss)
    #   num_samples = len(self._loss_mean)
    #   if (num_samples % 100) == 0:  # Print progress
    #     print('RSM loss samples: ', num_samples)
    #   if num_samples == samples:  # Print mean loss and reset
    #     mean = np.mean(self._loss_mean)
    #     self._loss_mean = None
    #     print('Mean RSM layer loss ==========> ', mean, ' (N=', num_samples, ')')
    ###########################################################################
    # Exact loss calculation over time, for comparison other impl.
    ###########################################################################

    # Feedback stays on-graph
    names = [self.previous, self.inhibition,
             self.encoding, self.decoding, self.encoding_mask,
             self.sum_abs_error, self.loss]

    self._dual.set_fetches(fetched, names)

    if self.use_freq():
      freq_names = [self.usage_cell, self.usage_col]
      self._dual.set_fetches(fetched, freq_names)

    # Dendrite traces
    if self._hparams.f_decay_rate > 0.0:
      trace_key = self.get_trace_key('f')
      self._dual.set_fetches(fetched, [trace_key])

    if self._hparams.rb_decay_rate > 0.0:
      trace_key = self.get_trace_key('r')
      self._dual.set_fetches(fetched, [trace_key])

      if self.use_feedback():
        trace_key = self.get_trace_key('b')
        self._dual.set_fetches(fetched, [trace_key])

    # Summaries
    super().set_fetches(fetched, batch_type)

  def _hidden_image_summary_shape(self):
    """image shape of hidden layer for summary."""
    volume = np.prod(self._encoding_shape[1:])
    square_image_shape, _ = image_utils.square_image_shape_from_1d(volume)
    return square_image_shape

  def _build_summaries(self, batch_type, max_outputs=3):
    """Build the summaries for TensorBoard."""

    #max_outputs=1
    logging.debug('layer: %s, batch type: %s', self.name, batch_type)

    summaries = []

    previous_pl = self._dual.get_pl(self.previous)
    encoding_op = self.get_op(self.encoding)
    encoding_mask_op = self._dual.get_op(self.encoding_mask)
    f_encoding_op = self.get_op(self.encoding_f)
    r_encoding_op = self.get_op(self.encoding_r)
    decoding_op = self.get_op(self.decoding)

    # Encoding
    # ---------------------------------------------------------------------------
    hidden_shape_4d = self._hidden_image_summary_shape()  # [batches, height=1, width=filters, 1]

    if self._hparams.summarize_encoding:
      # Output Encoding (combined forward + feedback)
      encoding_reshape = tf.reshape(encoding_op, hidden_shape_4d)
      encoding_summary_op = tf.summary.image(self.encoding, encoding_reshape, max_outputs=max_outputs)
      summaries.append(encoding_summary_op)

      # Output Encoding (combined forward + feedback)
      encoding_mask_reshape = tf.reshape(encoding_mask_op, hidden_shape_4d)
      encoding_summary_op = tf.summary.image(self.encoding_mask, encoding_mask_reshape, max_outputs=max_outputs)
      summaries.append(encoding_summary_op)

      # Forward Encoding
      f_encoding_reshape = tf.reshape(f_encoding_op, hidden_shape_4d)
      f_encoding_summary_op = tf.summary.image(self.encoding_f, f_encoding_reshape, max_outputs=max_outputs)
      summaries.append(f_encoding_summary_op)

      # Recurrent Encoding
      r_encoding_reshape = tf.reshape(r_encoding_op, hidden_shape_4d)
      r_encoding_summary_op = tf.summary.image(self.encoding_r, r_encoding_reshape, max_outputs=max_outputs)
      summaries.append(r_encoding_summary_op)

    # Decoding (i.e. prediction, and decode target)
    if self._hparams.summarize_decoding:
      summary_input_shape = image_utils.get_image_summary_shape(self._input_shape)

      # Target, reconstruction and prediction
      input_summary_reshape = tf.reshape(self._input_values, summary_input_shape)
      previous_summary_reshape = tf.reshape(previous_pl, summary_input_shape)
      decoding_summary_reshape = tf.reshape(decoding_op, summary_input_shape)

      concat = True
      concat_axis = 1  # 1 = Y

      if concat:
        summary_target = tf.concat(
            [input_summary_reshape, previous_summary_reshape, decoding_summary_reshape], axis=concat_axis)
        concat_summary_op = tf.summary.image('decoding-prediction', summary_target, max_outputs=max_outputs)
        summaries.append(concat_summary_op)
      else:
        input_summary_op = tf.summary.image('input', input_summary_reshape, max_outputs=max_outputs)
        summaries.append(input_summary_op)

        previous_summary_op = tf.summary.image(self.previous, previous_summary_reshape, max_outputs=max_outputs)
        summaries.append(previous_summary_op)

        decoding_summary_op = tf.summary.image(self.decoding, decoding_summary_reshape, max_outputs=max_outputs)
        summaries.append(decoding_summary_op)

    # Reconstruction-prediction loss
    memory_loss_summary = tf.summary.scalar(self.loss, self._dual.get_op(self.loss))
    summaries.append(memory_loss_summary)

    # Statistics
    # ---------------------------------------------------------------------------

    # Input statistics
    if self._hparams.summarize_input:
      input_stats_summary = tf_build_stats_summaries(self._input_values, 'input-stats')
      summaries.append(input_stats_summary)

    # Weights statistics
    if self._hparams.summarize_weights:
      w_f = self._weights_f
      w_r = self._weights_r
      w_d = self._weights_d
      summaries.append(tf.summary.scalar('Wf', tf.reduce_sum(tf.abs(w_f))))
      summaries.append(tf.summary.scalar('Wr', tf.reduce_sum(tf.abs(w_r))))
      summaries.append(tf.summary.scalar('Wd', tf.reduce_sum(tf.abs(w_d))))

      summaries.append(tf.summary.histogram('w_f_hist', w_f))
      summaries.append(tf.summary.histogram('w_r_hist', w_r))
      summaries.append(tf.summary.histogram('w_d_hist', w_d))

      # weight histograms
      wf_per_column = tf.reduce_sum(tf.abs(w_f), [0, 1, 2])  # over the columns (conv filters)
      summaries.append(tf.summary.histogram('wf_per_column', wf_per_column))
      wr_per_cell = tf.reduce_sum(tf.abs(w_r), [1])   # wb shape = [cells, dendrites]
      summaries.append(tf.summary.histogram('wb_per_cell', wr_per_cell))

      # weight images
      wf_cols = tf.reshape(w_f, [-1, self._hparams.cols])  # [weight per col, cols]
      wf_cols_summary_shape = image_utils.make_image_summary_shape_from_2d(wf_cols)
      wf_cols_reshaped = tf.reshape(wf_cols, wf_cols_summary_shape)
      wf_cols_summary_op = tf.summary.image('wf_cols', wf_cols_reshaped, max_outputs=max_outputs)
      summaries.append(wf_cols_summary_op)

      wb_cells = tf.reshape(w_r, [-1, self._hparams.cols * self._hparams.cells_per_col])  # [weight per col, cols]
      wb_cells_summary_shape = image_utils.make_image_summary_shape_from_2d(wb_cells)
      wb_cells_reshaped = tf.reshape(wb_cells, wb_cells_summary_shape)
      wb_cells_summary_op = tf.summary.image('wr_cells', wb_cells_reshaped, max_outputs=max_outputs)
      summaries.append(wb_cells_summary_op)

    # Dendrite traces
    summarize_traces = False
    if summarize_traces:
      def build_trace_image_summary(self, key, summaries, summary_shape):
        op = self._dual.get(key).get_op()
        #summary_shape = [300,60,80,1]
        op_reshape = tf.reshape(op, summary_shape)
        summary_op = tf.summary.image(prefix+suffix, op_reshape)
        summaries.append(summary_op)

      prefix = 'r'
      suffix = '-in'
      build_trace_image_summary(self, prefix+suffix, summaries, hidden_shape_4d)
      suffix = '-int'
      build_trace_image_summary(self, prefix+suffix, summaries, hidden_shape_4d)
      suffix = '-norm'
      build_trace_image_summary(self, prefix+suffix, summaries, hidden_shape_4d)
      suffix = '-dropo'
      build_trace_image_summary(self, prefix+suffix, summaries, hidden_shape_4d)

    # Utilization of resources
    if self._hparams.summarize_freq:
      freq_cell = self._dual.get(self.freq_cell).get_pl()
      freq_col = self._dual.get(self.freq_col).get_pl()

      summaries.append(tf.summary.histogram('freq_cell_hist', freq_cell))
      summaries.append(tf.summary.histogram('freq_col_hist', freq_col))

      freq_max_at_summary = tf.summary.scalar('freq_max_at', tf.argmax(freq_cell))
      summaries.append(freq_max_at_summary)

      freq_max_summary = tf.summary.scalar('freq_max', tf.reduce_max(freq_cell))
      summaries.append(freq_max_summary)

      lifetime_mask = self._dual.get(self.lifetime_mask).get_pl()
      lifetime_mask_summary = tf.summary.scalar('lifetime-mask-sum', tf.reduce_sum(tf.to_float(lifetime_mask)))
      summaries.append(lifetime_mask_summary)

      if self.use_boosting():
        boost_cell = self._dual.get_op(self.boost)  # Now a variable
        summaries.append(tf.summary.histogram('boost_cell_hist', boost_cell))
      else:
        inhibition = self._dual.get_op(self.inhibition)  # Now a variable
        summaries.append(tf.summary.histogram('inhibition_cell_hist', inhibition))

    return summaries
