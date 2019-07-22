# Copyright (C) 2018 Project AGI
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

import sys
import logging
import math

import os
from os.path import dirname, abspath

import numpy as np
import tensorflow as tf

from pagi.components.summary_component import SummaryComponent
from pagi.components.conv_autoencoder_component import ConvAutoencoderComponent

from pagi.utils import image_utils
from pagi.utils.dual import DualData
from pagi.utils.layer_utils import type_activation_fn
from pagi.utils.layer_utils import activation_fn
from pagi.utils.tf_utils import tf_build_stats_summaries
from pagi.utils.tf_utils import tf_build_top_k_mask_4d_op
from pagi.utils.tf_utils import tf_build_varying_top_k_mask_4d_op
from pagi.utils.tf_utils import tf_build_perplexity
from pagi.utils.tf_utils import tf_create_optimizer
from pagi.utils.tf_utils import tf_do_training
from pagi.utils.tf_utils import tf_random_mask


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
        momentum=0.9,  # Ignore if adam
        momentum_nesterov=False,

        # General options
        training_interval=[0,-1],  # [0,-1] means forever
        autoencode=False,
        hidden_nonlinearity='tanh', # used for hidden layer only
        inhibition_decay=0.1,  # controls refractory period

        predictor_norm_input=True,
        predictor_integrate_input=False,  # default: only current cells

        # Feedback
        feedback_decay_rate=0.0,  # Optional integrated/exp decay feedback
        feedback_keep_rate=1.0,  # Optional dropout on feedback
        feedback_norm=True,  # Option to normalize feedback
        feedback_norm_eps=0.00000001,  # Prevents feedback norm /0

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

        # Sparse parameters:
        sparsity=25,
        lifetime_sparsity_dends=False,
        lifetime_sparsity_cols=False,

        # Bias (shouldn't need to change)
        i_scale = 1.0,
        i_bias = 0.0,
        ff_bias=False,
        fb_bias=False,
        decode_bias=True,

        # Initialization
        f_sd=0.03,
        r_sd=0.03,
        b_sd=0.03,

        # Summaries
        summarize_input=False,
        summarize_encoding=False,
        summarize_decoding=False,
        summarize_weights=False,
        summarize_freq=False
    )

  def __init__(self):
    super().__init__()

  # Static names
  loss = 'loss'
  training = 'training'
  encoding = 'encoding'

  usage = 'usage'
  usage_col = 'usage-col'
  usage_cell = 'usage-cell'
  freq = 'freq'
  freq_col = 'freq-col'
  freq_cell = 'freq-cell'

  prediction_input = 'prediction-input'
  feedback_keep = 'feedback-keep'
  lifetime_mask = 'lifetime-mask'
  encoding_mask = 'encoding-mask'
  encoding = 'encoding'
  decoding = 'decoding'
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

  def update_statistics(self, session):  # pylint: disable=W0613
    if self.use_freq():
      self._update_freq()
      self._update_lifetime_mask()   # sensitive to minimum frequency

  def _update_freq(self):
    """Updates the cell utilisation frequency from usage."""
    if ((self._freq_update_count % self._hparams.freq_update_interval) == 0) and (self._freq_update_count > 0):
      self._freq_update_count = 0

      #self._update_freq_with_usage(self.usage, self.freq)
      self._update_freq_with_usage(self.usage_cell, self.freq_cell)
      self._update_freq_with_usage(self.usage_col, self.freq_col)

    self._freq_update_count += 1

  def _update_freq_with_usage(self, usage_key, freq_key):

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

      Fa = np.multiply(a, freq)
      fb = np.multiply(b, freq_old)

      freq_new = np.add(Fa, fb) # now the frequency has been updated
      self._dual.set_values(freq_key, freq_new)

  def _update_lifetime_mask(self):
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
    if self._hparams.autoencode is True:
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
      return

    if history_forgetting_probability < 1.0:
      p0 = history_forgetting_probability
      history_mask = tf_random_mask(p0, shape=(self._hparams.batch_size))
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

    fetched = session.run(fetches, feed_dict=feed_dict)

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

  def get_num_dendrites(self):
    #num_dendrites = self._hparams.cols * self._hparams.cells_per_col * self._hparams.dends_per_cell
    return self.get_num_cells()

  def get_num_cells(self):
    num_cells = self._hparams.cols * self._hparams.cells_per_col
    return num_cells

  def get_encoding_shape_4d(input_shape, hparams):
    return ConvAutoencoderComponent.get_convolved_shape(input_shape,
                                                   hparams.filters_field_height,
                                                   hparams.filters_field_width,
                                                   hparams.filters_field_stride,
                                                   hparams.cols * hparams.cells_per_col,
                                                   padding='SAME')

  def build(self, input_values, input_shape, hparams, name='rsm', encoding_shape=None, feedback_shape=None):  # pylint: disable=W0221
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

    self.name = name
    self._hparams = hparams
    self._freq_update_count = 0
    self._training_batch_count = 0

    #self._get_feedback_weights_op = None

    self._input_shape = input_shape
    self._input_values = input_values

    self._encoding_shape = encoding_shape
    if self._encoding_shape is None:
      self._encoding_shape = SequenceMemoryLayer.get_encoding_shape_4d(input_shape, self._hparams)

    self._feedback_shape = feedback_shape

    logging.info('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    logging.info('Layer "' + self.name + '"')
    logging.info('# Cols ' + str(self._hparams.cols))
    logging.info('# Cells/Col ' + str(self._hparams.cells_per_col))
    logging.info('# Total cells ' + str(self._hparams.cols * self._hparams.cells_per_col))
    logging.info('Input shape: ' + str(self._input_shape))
    logging.info('Encoding shape: ' + str(self._encoding_shape))
    if self.use_feedback():
      logging.info('Feedback shape: ' + str(self._feedback_shape))
    else:
      logging.info('Feedback shape: N/A')
    logging.info('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
      self._build()
      self._build_optimizer()

    self.reset()

    #self.build_summaries()

  def _build_history_update(self):
    """Builds graph ops to update the history tensors given a history mask"""

    # previous_input_shape = self._input_values.get_shape().as_list()
    # feedback_input_shape = self._create_encoding_shape_4d(previous_input_shape)
    # dend_6d_shape = [feedback_input_shape[0], feedback_input_shape[1], feedback_input_shape[2], self._hparams.cols, self._hparams.cells_per_col, self._hparams.dends_per_cell]
    #print( "batch size = ", self._hparams.batch_size)
    history_pl = self._dual.add(self.history, shape=[self._hparams.batch_size], default_value=1.0).add_pl()
    #print( "hist pl = ", history_pl)
    history_4d = tf.reshape(history_pl, [self._hparams.batch_size, 1, 1, 1])
    history_5d = tf.reshape(history_pl, [self._hparams.batch_size, 1, 1, 1, 1])
    #history_6d = tf.reshape(history_pl, [self._hparams.batch_size, 1, 1, 1, 1, 1])

    previous_pl = self._dual.get_pl(self.previous)
    inhibition_pl = self._dual.get_pl(self.inhibition)
    recurrent_pl = self._dual.get_pl(self.recurrent)

    previous_masked = tf.multiply(previous_pl, history_4d)
    #inhibition_masked = tf.multiply(inhibition_pl, history_6d)
    inhibition_masked = tf.multiply(inhibition_pl, history_5d)
    recurrent_masked = tf.multiply(recurrent_pl, history_4d)

    self._dual.set_op(self.previous_mask, previous_masked)
    self._dual.set_op(self.inhibition_mask, inhibition_masked)
    self._dual.set_op(self.recurrent_mask, recurrent_masked)

    if self.use_feedback():
      feedback_pl = self._dual.get_pl(self.feedback)
      feedback_masked = tf.multiply(feedback_pl, history_4d)
      self._dual.set_op(self.feedback_mask, feedback_masked)

  def _build_decay_and_integrate(self, feedback_old, feedback_now):
    """Builds graph ops to update feedback structures"""
    if self._hparams.feedback_decay_rate == 0.0:
      #self.set_feedback_values(output)
      return feedback_now

    # Integrate feedback over time, exponentially weighted decay.
    # Do this both for train and test.
    # Additive
    #feedback_new = (feedback_old * self._hparams.feedback_decay_rate) + feedback_now
    # Maximum
    feedback_new = tf.maximum(feedback_old * self._hparams.feedback_decay_rate, feedback_now)
    return feedback_new

  def _build_sum_norm(self, input_4d, do_norm, eps=0.00000000001):
    # Optionally apply a norm to make input constant sum
    # NOTE: Assuming here it is CORRECT to norm over conv w,h 
    if do_norm is True:
      sum_input = tf.reduce_sum(input_4d, axis=[1, 2, 3], keepdims=True) + eps
      unit_input_4d = tf.divide(input_4d, sum_input)
    else:
      unit_input_4d = input_4d
    return unit_input_4d

  def _build_fb_conditioning(self, feedback_input_4d):

    # Optionally apply a norm to make input constant sum
    unit_feedback = self._build_sum_norm(feedback_input_4d, self._hparams.feedback_norm)
    # if self._hparams.feedback_norm is True:
    #   sum_feedback_values = tf.reduce_sum(feedback_input_4d, axis=[1, 2, 3], keepdims=True) + eps
    #   unit_feedback_values = tf.divide(feedback_input_4d, sum_feedback_values)
    # else:
    #   unit_feedback_values = feedback_input_4d

    # Dropout AFTER norm. There will be a scaling factor inside dropout 
    if self._hparams.feedback_keep_rate < 1.0:
      feedback_keep_pl = self._dual.add(self.feedback_keep, shape=(), default_value=1.0).add_pl(default=True)
      unit_feedback_dropout = tf.nn.dropout(unit_feedback, feedback_keep_pl)  # Note, a scaling is applied
    else:
      unit_feedback_dropout = unit_feedback

    # Return both values to be used as necessary
    return unit_feedback, unit_feedback_dropout

  def _build_ff_conditioning(self, f_input):
    # if self._hparams.autoencode is True:
    #   ff_input = self._input_values
    # else:
    #   previous_pl = self._dual.get_pl(self.previous)
    #   ff_input = previous_pl # ff_input = x_ff(t-1)

    # TODO could add ff dropout here.

    # TODO What norm should we do for stacked layers? Is zero special?
    # Perhaps I should batch-norm.

    # Adjust range - e.g. if range is 0 <= x <= 3, then
    # 
    if self._hparams.i_scale != 1.0:
      f_input = f_input * self._hparams.i_scale

    if self._hparams.i_bias != 0.0:
       f_input = f_input + self._hparams.i_bias

    return f_input

  def _build(self):
    """Build the autoencoder network"""

    num_dendrites = self.get_num_dendrites()

    if self.use_freq():
      self._dual.add(self.freq_col, shape=[self._hparams.cols], default_value=0.0).add_pl()
      self._dual.add(self.freq_cell, shape=[self._hparams.cols * self._hparams.cells_per_col], default_value=0.0).add_pl()
      #self._dual.add(self.freq, shape=[num_dendrites], default_value=0.0).add_pl()
      #self._dual.add(self.usage, shape=[num_dendrites], default_value=0.0).add_pl()
      self._dual.add(self.usage_cell, shape=[self._hparams.cols * self._hparams.cells_per_col], default_value=0.0).add_pl()
      self._dual.add(self.usage_col, shape=[self._hparams.cols], default_value=0.0).add_pl()

    self._dual.add(self.lifetime_mask, shape=[num_dendrites], default_value=1.0).add_pl(dtype=tf.bool)

    # ff input update - we work with the PREVIOUS ff input. This code stores the current input for access next time.
    input_shape_list = self._input_values.get_shape().as_list()
    previous_pl = self._dual.add(self.previous,shape=input_shape_list, default_value=0.0).add_pl()
    previous_op = self._dual.set_op(self.previous, self._input_values)

    # FF input
    ff_target = self._input_values
    if self._hparams.autoencode is True:
      ff_input = ff_target  # ff_input = x_ff(t)
    else:
      ff_input = previous_pl  # ff_input = x_ff(t-1)
    ff_input = self._build_ff_conditioning(ff_input)

    with tf.name_scope('encoding'):

      # Recurrent input
      recurrent_input_shape = SequenceMemoryLayer.get_encoding_shape_4d(input_shape_list, self._hparams)
      dend_5d_shape = [recurrent_input_shape[0], recurrent_input_shape[1], recurrent_input_shape[2],
                       self._hparams.cols, self._hparams.cells_per_col]

      recurrent_pl = self._dual.add(self.recurrent, shape=recurrent_input_shape, default_value=0.0).add_pl(default=True)
      unit_recurrent, unit_recurrent_dropout = self._build_fb_conditioning(recurrent_pl)
      r_unit_dropout = unit_recurrent_dropout

      # Inhibition (used later, but same shape)
      inhibition_pl = self._dual.add(self.inhibition, shape=dend_5d_shape, default_value=0.0).add_pl()

      # FB input
      b_unit_dropout = None
      if self.use_feedback():
        # Note feedback will be integrated in other layer, if that's a thing.
        feedback_input_shape = self._feedback_shape
        feedback_pl = self._dual.add(self.feedback, shape=feedback_input_shape, default_value=0.0).add_pl(default=True)

        # Interpolate the feedback to the conv w,h of the current layer
        interpolated_size = [recurrent_input_shape[1], recurrent_input_shape[2]]  # note h,w order
        feedback_interpolated = tf.image.resize_bilinear(feedback_pl, interpolated_size)
        unit_feedback, unit_feedback_dropout = self._build_fb_conditioning(feedback_interpolated)
        b_unit_dropout = unit_feedback_dropout

      # Encoding of dendrites (weighted sums)
      f_encoding_cells_5d, r_encoding_cells_5d, b_encoding_cells_5d = self._build_encoding(
          ff_input, r_unit_dropout, b_unit_dropout)  # y(t) = f x_ff(t-1), y(t-1)

      # Sparse masking, and integration of dendrites
      training_filtered_cells_5d, testing_filtered_cells_5d = self._build_filtering(
          f_encoding_cells_5d, r_encoding_cells_5d, b_encoding_cells_5d)  # output y(t) | x_ff(t-1), y(t-1)

      # Inference output fork, doesn't accumulate gradients
      output_encoding_cells_5d = tf.stop_gradient(testing_filtered_cells_5d)  # this is the encoding op
      output_encoding_cells_5d_shape = output_encoding_cells_5d.get_shape().as_list()
      output_encoding_cells_4d = tf.reshape(output_encoding_cells_5d,
                                            [-1, output_encoding_cells_5d_shape[1], output_encoding_cells_5d_shape[2],
                                             output_encoding_cells_5d_shape[3] * output_encoding_cells_5d_shape[4]])

      # NOTE: accumulated recurrent is WITHOUT dropout, so different selections can be made.
      output_integrated_cells_4d = self._build_decay_and_integrate(unit_recurrent, output_encoding_cells_4d)
      self._dual.set_op(self.encoding, output_integrated_cells_4d)  # if feedback decay is 0, then == current encoding

      # Generate inputs and targets for prediction
      # -----------------------------------------------------------------
      with tf.name_scope('prediction'):

        # Generate what would be the feedback input next time - ie apply the norm
        if self._hparams.predictor_integrate_input:
          prediction_encoding = output_integrated_cells_4d  # WITH integration and decay
        else:
          prediction_encoding = output_encoding_cells_4d  # WITHOUT integration and decay

        # Norm prediction input
        unit_prediction_encoding = self._build_sum_norm(prediction_encoding, self._hparams.predictor_norm_input)

        # Input values for prediction (no dropout)
        prediction_input_values = unit_prediction_encoding  # predict t | y_t = f(t-1, t-2, ... )
        prediction_input_shape = recurrent_input_shape  # Default

        # Set these ops/properties:
        self._dual.set_op(self.prediction_input, prediction_input_values)
        self._dual.get(self.prediction_input).set_shape(prediction_input_shape)

    # Decode: predict x_ff(t) | y(t)
    # -----------------------------------------------------------------
    # NB: y(t) doesnt depend on x_ff(t)
    # y(t) = f( x_ff(t-1) , y(t-1) OK
    with tf.name_scope('decoding'):
      output_encoding_cols_4d = tf.reduce_max(training_filtered_cells_5d, axis=-1)

      # decode: predict x_t given y_t where
      # y_t = encoding of x_t-1 and y_t-1
      decoding = self._build_decoding(ff_target, output_encoding_cols_4d)
      self._dual.set_op(self.decoding, decoding) # This is the thing that's optimized by the memory itself

      # Debug the per-sample prediction error inherent to the memory.
      prediction_error = tf.abs(ff_target - decoding)
      sum_abs_error = tf.reduce_sum(prediction_error, axis=[1,2,3])
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
      f_encoding_cells_5d = self._build_cols_4d_to_cells_5d(f_encoding_cols_4d)
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

  def _build_conv_encoding(self, 
                           input_shape, 
                           input_tensor, 
                           field_stride,
                           field_height,
                           field_width,
                           output_depth,
                           add_bias,
                           initial_sd,
                           scope):
    """Build encoding ops for the forward path."""
    input_depth = input_shape[3]
    #strides = [1, field_stride, field_stride, input_depth]
    strides = [1, field_stride, field_stride, 1]
    conv_variable_shape = [
        field_height,
        field_width,
        input_depth,
        output_depth  # number of filters
    ]

    # print('input tensor: ', input_tensor)
    # print('input_shape: ', input_shape)
    # print('conv_variable_shape: ', conv_variable_shape)
    # print('strides: ', strides)
    # print('output_depth: ', output_depth)

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE, auxiliary_name_scope=False):

      weights = tf.get_variable(shape=conv_variable_shape,
                                initializer=tf.truncated_normal_initializer(stddev=initial_sd), name='kernel')

      convolved = tf.nn.conv2d(input_tensor, weights, strides, padding='SAME', name='convolved') # zero padded
      bias = None

      if add_bias:
        # Note: bias has nonzero default initializer if enabled
        bias = tf.get_variable(shape=[output_depth], name='bias', trainable=True)
        convolved = tf.nn.bias_add(convolved, bias)

      return convolved, weights, bias

  # def _build_feedback_encoding(self, input_shape, input_tensor, filters):
  #   """Build encoding ops for feedback path."""
  #   input_area = np.prod(input_shape[1:])
  #   batch_input_shape = (-1, input_area)
  #   input_vector = tf.reshape(input_tensor, batch_input_shape)
  #   weighted_sum = tf.layers.dense(inputs=input_vector, units=filters, activation=None, use_bias=self._hparams.fb_bias, name='feedback')
  #   return weighted_sum

  # def _build_get_feedback_weights_op(self):
  #   if self._get_feedback_weights_op is None:
  #     with tf.variable_scope('feedback', reuse=True):
  #       self._get_feedback_weights_op = tf.get_variable('kernel')
  #       logging.debug('weights: %s', self._get_feedback_weights_op)  # shape=(784, 1024) = input, cells

  def _build_forward_encoding(self, input_shape, input_tensor):
    """Build encoding ops for the forward path."""
    output_depth = self._hparams.cols
    field_stride = self._hparams.filters_field_stride
    field_height = self._hparams.filters_field_height
    field_width  = self._hparams.filters_field_width
    add_bias = self._hparams.ff_bias
    initial_sd = self._hparams.f_sd  #0.03
    scope = 'forward'
    convolved, self._weights_f, self._bias_f = self._build_conv_encoding(input_shape, input_tensor, field_stride, field_height, field_width, output_depth, add_bias, initial_sd, scope)
    return convolved

  def _build_recurrent_encoding(self, input_shape, input_tensor):
    """Build encoding ops for the forward path."""
    output_depth = self._hparams.cols * self._hparams.cells_per_col
    field_stride = 1  # i.e. feed to itself
    field_height = 1  # i.e. same position
    field_width  = 1  # i.e. same position
    add_bias = self._hparams.fb_bias
    initial_sd = self._hparams.r_sd  # Not previously defined
    scope = 'recurrent'
    convolved, self._weights_r, self._bias_r = self._build_conv_encoding(input_shape, input_tensor, field_stride, field_height, field_width, output_depth, add_bias, initial_sd, scope)
    return convolved

  def _build_feedback_encoding(self, input_shape, input_tensor):
    """Build encoding ops for the forward path."""
    output_depth = self._hparams.cols * self._hparams.cells_per_col
    field_stride = 1  # i.e. feed to itself
    field_height = 1  # i.e. same position
    field_width  = 1  # i.e. same position
    add_bias = self._hparams.fb_bias
    initial_sd = self._hparams.b_sd  # Not previously defined
    scope = 'feedback'
    convolved, self._weights_r, self._bias_r = self._build_conv_encoding(input_shape, input_tensor, field_stride, field_height, field_width, output_depth, add_bias, initial_sd, scope)
    return convolved

  def _build_cols_mask(self, h, w, ranking_input_cols_4d):
    # ---------------------------------------------------------------------------------------------------------------
    # TOP-k RANKING
    # ---------------------------------------------------------------------------------------------------------------
    k = int(self._hparams.sparsity)
    batch_size = self._hparams.batch_size
    hidden_size = self._hparams.cols
    top_k_mask_cols_4d = tf_build_top_k_mask_4d_op(ranking_input_cols_4d, k, batch_size, h, w, hidden_size)

    # ---------------------------------------------------------------------------------------------------------------
    # TOP-1 over cells - lifetime sparsity [in winning Cols only.]
    # ---------------------------------------------------------------------------------------------------------------
    # At each batch, h, w, and col, find the top-1 cell. 
    # Don't reduce over cells - dendrite is only valid for its associated cell. 
    top_1_input_cols_4d = tf.reduce_max(ranking_input_cols_4d, axis=[0,1,2], keepdims=True)
    top_1_mask_bool_cols_4d = tf.greater_equal(ranking_input_cols_4d, top_1_input_cols_4d)
    top_1_mask_cols_4d = tf.to_float(top_1_mask_bool_cols_4d)

    # ---------------------------------------------------------------------------------------------------------------
    # Exclude naturally winning cols from the lifetime sparsity mask.
    # ---------------------------------------------------------------------------------------------------------------
    top_k_inv_cols_4d = 1.0 - tf.reduce_max(top_k_mask_cols_4d, axis=[0,1,2], keepdims=True)
    top_1_mask_cols_4d = tf.multiply(top_1_mask_cols_4d, top_k_inv_cols_4d) # removes lifetime bits from winning cols

    # ---------------------------------------------------------------------------------------------------------------
    # Combine the two masks - lifetime sparse, and top-k-excluding-threshold
    # ---------------------------------------------------------------------------------------------------------------
    #either_mask_cols_4d = tf.logical_or(top_k_mask_cols_4d, top_1_mask_cols_4d)
    if self._hparams.lifetime_sparsity_cols:
      either_mask_cols_4d = tf.maximum(top_k_mask_cols_4d, top_1_mask_cols_4d)
    else:
      either_mask_cols_4d = top_k_mask_cols_4d

    either_mask_cols_5d = self._build_cols_4d_to_cells_5d(either_mask_cols_4d)
    # either_mask_cols_5d = tf.tile(tf.expand_dims(either_mask_cols_4d, -1),
    #                        [1, 1, 1, 1, self._hparams.cells_per_col])
    # either_mask_cols_6d = tf.tile(tf.expand_dims(either_mask_cols_5d, -1), # Expand to 6d
    #                        [1, 1, 1, 1, 1, self._hparams.dends_per_cell])

    return either_mask_cols_5d

  def _build_tiebreaker_masks(self, h, w):

    # Deal with the special case that there's no feedback and all the cells and dends are equal
    # Build a special mask to select ONLY the 1st cell/dend in each col, in the event that all the inputs are identical
    #shape_dend_6d = [self._hparams.batch_size, h, w, self._hparams.cols, self._hparams.cells_per_col, self._hparams.dends_per_cell]
    shape_cells_5d = [self._hparams.batch_size, h, w, self._hparams.cols, self._hparams.cells_per_col]

    np_cells_5d_1st = np.zeros(shape_cells_5d) # 1 for first cell / dendrite in the col
    np_cells_5d_all = np.ones(shape_cells_5d)

    for b in range(shape_cells_5d[0]):
      for y in range(shape_cells_5d[1]):
        for x in range(shape_cells_5d[2]):
          for col in range(shape_cells_5d[3]):
            #for cell in range(shape_dend_6d[4]):
            #  for dend in range(shape_dend_6d[5]):
            #np_dend_6d_1st[b][y][x][col][0][0] = 1 # 1st cell+dend
            np_cells_5d_1st[b][y][x][col][0] = 1 # 1st cell+dend

    self._mask_1st_cells_5d = tf.constant(np_cells_5d_1st, dtype=tf.float32)
    self._mask_all_cells_5d = tf.constant(np_cells_5d_all, dtype=tf.float32)

  def _build_dendrite_mask(self, h, w, ranking_input_cells_5d, mask_cols_5d, lifetime_sparsity_dends, lifetime_mask_dend_1d):

    # Find the max value in each col, by reducing over cells and dends
    # max_col_6d = tf.reduce_max(ranking_input_dend_6d, axis=[4, 5], keepdims=True) # 1 if bit is best in the col
    # min_col_6d = tf.reduce_min(ranking_input_dend_6d, axis=[4, 5], keepdims=True)
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
    #top1_dend_6d = tf.reduce_max(ranking_input_dend_6d, axis=[4, 5], keepdims=True) # max value per CELL, of ONLY the samples to which this is the batch-max for the COL.
    best_cells_5d_bool = tf.greater_equal(ranking_input_cells_5d, max_cells_5d)

    # Exclude cells from cols that weren't selected
    masked_ranking_input_cells_5d = ranking_input_cells_5d * mask_cols_5d

    # Lifetime sparsity
    if lifetime_sparsity_dends:
      top_1_max_cells_5d = tf.reduce_max(masked_ranking_input_dend_6d, axis=[0,1,2], keepdims=True)
      top_1_cells_5d_bool = tf.greater_equal(masked_ranking_input_cells_5d, top_1_max_cells_5d)

      # exclude dends with actual uses from lifetime training
      #shape_dend_6d_111 = [1, 1, 1, self._hparams.cols, self._hparams.cells_per_col, self._hparams.dends_per_cell]
      shape_cells_5d_111 = [1, 1, 1, self._hparams.cols, self._hparams.cells_per_col]

      #lifetime_mask_cells_5d = tf.reshape(lifetime_mask_dend_1d, shape=shape_dend_6d_111)
      lifetime_mask_cells_5d = tf.reshape(lifetime_mask_dend_1d, shape=shape_cells_5d_111)
      top_1_cells_5d_bool = tf.logical_and(top_1_cells_5d_bool, lifetime_mask_cells_5d)

      top1_best_cells_5d_bool = tf.logical_or(top_1_cells_5d_bool, best_cells_5d_bool)
    else:
      top1_best_cells_5d_bool = best_cells_5d_bool

    # Now combine this mask with the Top-k cols mask, giving a combined mask:
    mask_cols_5d_bool = tf.greater(mask_cols_5d, 0.0) # Convert to bool 
    #mask_dend_6d_bool = tf.logical_and(best_dend_6d_bool, mask_cols_6d_bool)
    mask_cells_5d_bool = tf.logical_and(top1_best_cells_5d_bool, mask_cols_5d_bool)
    mask_cells_5d = tf.to_float(mask_cells_5d_bool)

    edge_cells_5d_bool = mask_cells_5d * if0_mask_cells_5d_bool 

    #mask_dend_6d = top_1_mask_dend_6d
    #mask_dend_6d = tf.maximum(top_1_mask_dend_6d, 0.0)
    #mask_dend_6d = tf.maximum(top_1_mask_dend_6d, batch_max_mask_dend_6d)
    return edge_cells_5d_bool

  def _build_cols_4d_to_cells_5d(self, cols_4d):
    cols_5d = tf.expand_dims(cols_4d, -1)
    cells_5d = tf.tile(cols_5d, [1, 1, 1, 1, self._hparams.cells_per_col])
    return cells_5d

  def _build_cells_4d_to_cells_5d(self, cells_4d):
    cells_4d_shape = cells_4d.get_shape().as_list()
    cells_5d_shape = [cells_4d_shape[0], cells_4d_shape[1], cells_4d_shape[2], self._hparams.cols, self._hparams.cells_per_col]
    cells_5d = tf.reshape(cells_4d, cells_5d_shape)
    return cells_5d

  def _build_filtering(self, f_encoding_cells_5d, r_encoding_cells_5d, b_encoding_cells_5d):
    """Build filtering/masking for specified encoding."""
    #hidden_size = self._hparams.cols
    shape_cells_5d = f_encoding_cells_5d.get_shape().as_list()
    h = shape_cells_5d[1]
    w = shape_cells_5d[2]
    #shape_dend_6d = [self._hparams.batch_size, h, w, self._hparams.cols, self._hparams.cells_per_col, self._hparams.dends_per_cell]
    #shape_cells_5d = [self._hparams.batch_size, h, w, self._hparams.cols, self._hparams.cells_per_col]


    # ---------------------------------------------------------------------------------------------------------------
    # Expand FF to 6d to work out per-dendrite weighted sums
    # Then mask in dend 6d space, and reduce back for cols values.
    # ---------------------------------------------------------------------------------------------------------------
    #f_encoding_cols_5d = self._build_cols_4d_to_cells_5d(f_encoding_cols_4d)
    # tf.tile(tf.expand_dims(f_encoding_cols_4d, -1), # Expand to 5d
    #                        [1, 1, 1, 1, self._hparams.cells_per_col])
    # f_encoding_cols_6d = tf.tile(tf.expand_dims(f_encoding_cols_5d, -1), # Expand to 6d
    #                        [1, 1, 1, 1, 1, self._hparams.dends_per_cell])


    # ---------------------------------------------------------------------------------------------------------------
    # INHIBITION - retard the activation of dendrites that fired recently.
    # inhibition shape = [b,h,w,col,cell,d]
    # ---------------------------------------------------------------------------------------------------------------
    i_encoding_cells_5d = f_encoding_cells_5d + r_encoding_cells_5d  # i = integrated
    if self.use_feedback():
      i_encoding_cells_5d = i_encoding_cells_5d + b_encoding_cells_5d

    # Refractory period
    # Inhibition is max (1) and decays to min (0)
    # So refractory weight is 1-1=0, returning to 1 over time.
    inhibition_cells_5d_pl = self._dual.get_pl(self.inhibition)
    refractory_cells_5d = 1.0 - inhibition_cells_5d_pl # ie inh.=1, ref.=0 . Inverted

    # Pos encoding (only used for ranking)
    # Find min in each batch sample
    # This is wrong for conv.? Should be min in each conv location. 
    # But - it doesn't matter?
    min_i_encoding_cells_5d = tf.reduce_min(i_encoding_cells_5d, axis=[1,2,3,4], keepdims=True) # may be negative
    pos_i_encoding_cells_5d = (i_encoding_cells_5d - min_i_encoding_cells_5d) + 1.0 # shift to +ve range, ensure min value is nonzero

    # Apply inhibition/refraction
    refracted_cells_5d = pos_i_encoding_cells_5d * refractory_cells_5d

    # reduce to cols
    # take best(=max) dendrite per cell, and best cell per col.
    ranking_input_cols_4d = tf.reduce_max(refracted_cells_5d, axis=[4])
    ranking_input_cells_5d = refracted_cells_5d

    # ---------------------------------------------------------------------------------------------------------------
    # TOP-k RANKING (Cols)
    # ---------------------------------------------------------------------------------------------------------------
    #either_mask_cols_6d = self._build_cols_mask(h, w, refracted_cols_4d)
    mask_cols_5d = self._build_cols_mask(h, w, ranking_input_cols_4d)

    # ---------------------------------------------------------------------------------------------------------------
    # TOP-1 RANKING (Dend)
    # Reduce over dendrites AND cells to find the best dendrite at each b, h, w, col = 0,1,2,3
    # ---------------------------------------------------------------------------------------------------------------
    self._build_tiebreaker_masks(h, w)
    lifetime_mask_dend_1d = self._dual.get_pl(self.lifetime_mask)
    training_lifetime_sparsity_dends = self._hparams.lifetime_sparsity_dends
    testing_lifetime_sparsity_dends = False
    training_mask_cells_5d = self._build_dendrite_mask(h, w, ranking_input_cells_5d, mask_cols_5d, training_lifetime_sparsity_dends, lifetime_mask_dend_1d)
    testing_mask_cells_5d = self._build_dendrite_mask(h, w, ranking_input_cells_5d, mask_cols_5d, testing_lifetime_sparsity_dends, lifetime_mask_dend_1d)

    # Update cells' inhibition
    inhibition_cells_5d = inhibition_cells_5d_pl * self._hparams.inhibition_decay # decay old inh
    inhibition_cells_5d = tf.maximum(training_mask_cells_5d, inhibition_cells_5d) #this should be per batch sample not only per dend
    self._dual.set_op(self.inhibition, inhibition_cells_5d)

    # Update usage (test mask doesnt include lifetime bits)
    if self.use_freq():
      self._build_update_usage(testing_mask_cells_5d)

    # Produce the final filtering with these masks
    training_filtered_cells_5d, testing_filtered_cells_5d = self._build_nonlinearities(f_encoding_cells_5d, r_encoding_cells_5d, b_encoding_cells_5d, training_mask_cells_5d)

    return training_filtered_cells_5d, testing_filtered_cells_5d

  def _build_update_usage(self, mask_cells_5d):

    # ---------------------------------------------------------------------------------------------------------------
    # USAGE UPDATE - per dendrite
    # Reduce over the batch, w, and h.
    # Note that any training samples will be included, including lifetime sparsity
    # ---------------------------------------------------------------------------------------------------------------
    # num_dendrites = self.get_num_dendrites()
    # usage_dend_3d = tf.reduce_sum(mask_dend_6d, axis=[0,1,2]) # reduce over b,h,w leaving col,cell,dend
    # usage_dend_1d = tf.reshape(usage_dend_3d, shape=[num_dendrites])
    # usage_pl = self._dual.get(self.usage).get_pl()
    # usage_op = usage_pl + usage_dend_1d
    # self._dual.set_op(self.usage, usage_op)

    usage_col_1d = tf.reduce_sum(mask_cells_5d, axis=[0,1,2, 4]) # reduce over b,h,w: 0,1,2, 4 leaving col (3)
    usage_col_pl = self._dual.get(self.usage_col).get_pl()
    usage_col_op = usage_col_pl + usage_col_1d
    self._dual.set_op(self.usage_col, usage_col_op)

    num_cells = self._hparams.cols * self._hparams.cells_per_col
    usage_cell_2d = tf.reduce_sum(mask_cells_5d, axis=[0,1,2]) # reduce over b,h,w: 0,1,2 leaving col,cell (3,4)
    usage_cell_1d = tf.reshape(usage_cell_2d, shape=[num_cells])
    usage_cell_pl = self._dual.get(self.usage_cell).get_pl()
    usage_cell_op = usage_cell_pl + usage_cell_1d
    self._dual.set_op(self.usage_cell, usage_cell_op)

  def _build_nonlinearities(self, f_encoding_cells_5d, r_encoding_cells_5d, b_encoding_cells_5d, mask_cells_5d):
    """Mask the encodings, sum them, and apply nonlinearity. Don't backprop into the mask."""

    # Expand cols to cells 5d 
    #f_encoding_cells_5d = self._build_cols_4d_to_cells_5d(f_encoding_cols_4d)
    # f_encoding_cells_5d = tf.tile(tf.expand_dims(f_encoding_cols_4d, -1),
    #                        [1, 1, 1, 1, self._hparams.cells_per_col])

    mask_cells_5d = tf.stop_gradient(mask_cells_5d)
    self._dual.set_op(self.encoding_mask, mask_cells_5d)

    # Apply masks
    f_masked_cells_5d = f_encoding_cells_5d * mask_cells_5d
    r_masked_cells_5d = r_encoding_cells_5d * mask_cells_5d

    # Combine forward and feedback encoding
    hidden_sum_cells_5d = f_masked_cells_5d + r_masked_cells_5d

    if self.use_feedback():
      b_masked_cells_5d = b_encoding_cells_5d * mask_cells_5d
      hidden_sum_cells_5d = hidden_sum_cells_5d + b_masked_cells_5d

    # Nonlinearity
    # if self._hparams.hidden_nonlinearity == 'tanh':
    #   hidden_transfer_cells_5d = tf.tanh(hidden_sum_cells_5d)
    # else:
    hidden_transfer_cells_5d, _ = activation_fn(hidden_sum_cells_5d, self._hparams.hidden_nonlinearity)
    return hidden_transfer_cells_5d, hidden_transfer_cells_5d

  def _build_decoding(self, input_tensor, hidden_tensor):  # pylint: disable=W0613
    """Build the decoder (optionally without using tied weights)"""

    # Untied weights
    # -----------------------------------------------------------------
    deconv_shape = self._input_shape
    deconv_shape[0] = self._hparams.batch_size
    deconv_area = np.prod(deconv_shape[1:])
    hidden_area = np.prod(hidden_tensor.get_shape()[1:])

    hidden_reshape = tf.reshape(hidden_tensor, shape=[self._hparams.batch_size, hidden_area])
    decoding_weighted_sum = tf.layers.dense(
        inputs=hidden_reshape, units=deconv_area,
        activation=None, use_bias=self._hparams.decode_bias, name='deconvolved')  # Note we use our own nonlinearity, elsewhere.

    decoding_reshape = tf.reshape(decoding_weighted_sum, deconv_shape, name='decoding_reshape')
    logging.debug(decoding_reshape)

    return decoding_reshape

  def _build_optimizer(self):
    """Setup the training operations"""
    target = self._input_values
    prediction = self.get_op(self.decoding)
    with tf.variable_scope('optimizer', reuse=tf.AUTO_REUSE):
      self._optimizer = tf_create_optimizer(self._hparams)
      loss_op_1 = self._build_loss_fn(target, prediction)
      self._dual.set_op(self.loss, loss_op_1)
      training_op = self._optimizer.minimize(loss_op_1, global_step=tf.train.get_or_create_global_step(), name='training_op')
      self._dual.set_op(self.training, training_op)

  def _build_loss_fn(self, target, output):
    if self._hparams.loss_type == 'mse':
      return tf.losses.mean_squared_error(target, output)
    elif self._hparams.loss_type == 'sxe':
       return tf.losses.sigmoid_cross_entropy(target, output)
    else:
      raise NotImplementedError('Loss function not implemented: ' + str(self._hparams.loss_type))

  # COMPONENT INTERFACE ------------------------------------------------------------------
  def update_feed_dict(self, feed_dict, batch_type='training'):

    # Update these dual entities in the normal manner
    # Feedback stays on-graph
    names = [self.inhibition, self.previous, self.recurrent, self.lifetime_mask] 
    self._dual.update_feed_dict(feed_dict, names)

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

    # Adjust feedback keep rate by batch type
    feedback_keep_rate = 1.0  # Encoding
    if batch_type == self.training:
      feedback_keep_rate = self._hparams.feedback_keep_rate  # Training    
    logging.debug('Feedback keep rate: ' + str(feedback_keep_rate))

    # Placeholder for feedback keep rate
    if self._hparams.feedback_keep_rate < 1.0:
      feedback_keep = self._dual.get(self.feedback_keep)
      feedback_keep_pl = feedback_keep.get_pl()
      feed_dict.update({
          feedback_keep_pl: feedback_keep_rate
      })

  def add_fetches(self, fetches, batch_type='training'):
    """Adds ops that will get evaluated."""
    # Feedback stays on-graph
    names = [self.previous,
             self.encoding, self.decoding, self.encoding_mask,
             self.sum_abs_error, self.loss]
             # self.usage_cell, self.usage_col]
             # #self.usage, self.usage_cell, self.usage_col]

    # if not self._hparams.autoencode:
    #   names.append(self.inhibition)
    names.append(self.inhibition)

    self._dual.add_fetches(fetches, names)

    if self.use_freq():
      freq_names = [self.usage_cell, self.usage_col]
      self._dual.add_fetches(fetches, freq_names)

    do_training = tf_do_training(batch_type, self._hparams.training_interval, self._training_batch_count, name=self.name)
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

    # Feedback stays on-graph
    names = [self.previous,
             self.encoding, self.decoding, self.encoding_mask,
             self.sum_abs_error, self.loss]
             # self.usage_cell, self.usage_col, self.loss]
             # #self.usage, self.usage_cell, self.usage_col, self.loss]

    # if not self._hparams.autoencode:
    #   names.append(self.inhibition)

    names.append(self.inhibition)

    self._dual.set_fetches(fetched, names)

    if self.use_freq():
      freq_names = [self.usage_cell, self.usage_col]
      self._dual.set_fetches(fetched, freq_names)

    # Summaries
    super().set_fetches(fetched, batch_type)

  def _hidden_image_summary_shape(self):
    """image shape of hidden layer for summary."""
    volume = np.prod(self._encoding_shape[1:])
    square_image_shape, _ = image_utils.square_image_shape_from_1d(volume)
    return square_image_shape

  def _build_summaries(self, batch_type, max_outputs=3):
    """Build the summaries for TensorBoard."""

    logging.debug('layer: ' + self.name + ' batch type: ' + batch_type )

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
      w_b = self._weights_r #self._get_feedback_weights_op
      summaries.append(tf.summary.scalar('Wf', tf.reduce_sum(tf.abs(w_f))))
      summaries.append(tf.summary.scalar('Wb', tf.reduce_sum(tf.abs(w_b))))

      summaries.append(tf.summary.histogram('w_f_hist', w_f))
      summaries.append(tf.summary.histogram('w_b_hist', w_b))

      # weight histograms
      wf_per_column = tf.reduce_sum(tf.abs(w_f), [0, 1, 2])  # over the columns (conv filters)
      summaries.append(tf.summary.histogram('wf_per_column', wf_per_column))
      wb_per_cell = tf.reduce_sum(tf.abs(w_b), [1])   # wb shape = [cells, dendrites]
      summaries.append(tf.summary.histogram('wb_per_cell', wb_per_cell))
      wb_per_dendrite = tf.reduce_sum(tf.abs(w_b), [0])
      summaries.append(tf.summary.histogram('wb_per_dendrite', wb_per_dendrite))

      # weight images
      wf_cols = tf.reshape(w_f, [-1, self._hparams.cols])  # [weight per col, cols]
      wf_cols_summary_shape = image_utils.make_image_summary_shape_from_2d(wf_cols)
      wf_cols_reshaped = tf.reshape(wf_cols, wf_cols_summary_shape)
      wf_cols_summary_op = tf.summary.image('wf_cols', wf_cols_reshaped, max_outputs=max_outputs)
      summaries.append(wf_cols_summary_op)

      wb_cells = tf.reshape(w_b, [-1, self._hparams.cols * self._hparams.cells_per_col])  # [weight per col, cols]
      #wb_cells = tf.transpose(w_b)    # columns of image are cells (groups of cells comprise a col)
      wb_cells_summary_shape = image_utils.make_image_summary_shape_from_2d(wb_cells)
      wb_cells_reshaped = tf.reshape(wb_cells, wb_cells_summary_shape)
      wb_cells_summary_op = tf.summary.image('wb_cells', wb_cells_reshaped, max_outputs=max_outputs)
      summaries.append(wb_cells_summary_op)

    # Utilization of resources
    if self._hparams.summarize_freq:
      #usage = self._dual.get_op(self.usage)
      #freq = self._dual.get(self.freq).get_pl()
      freq_cell = self._dual.get(self.freq_cell).get_pl()
      freq_col = self._dual.get(self.freq_col).get_pl()

      #summaries.append(tf.summary.histogram('usage_hist', usage))  # not v interesting

      #summaries.append(tf.summary.histogram('freq_dend_hist', freq))
      #summaries.append(tf.summary.histogram('freq_hist', freq))
      summaries.append(tf.summary.histogram('freq_cell_hist', freq_cell))
      summaries.append(tf.summary.histogram('freq_col_hist', freq_col))

      freq_max_at_summary = tf.summary.scalar('freq_max_at', tf.argmax(freq_cell))
      summaries.append(freq_max_at_summary)

      freq_max_summary = tf.summary.scalar('freq_max', tf.reduce_max(freq_cell))
      summaries.append(freq_max_summary)

      lifetime_mask = self._dual.get(self.lifetime_mask).get_pl()
      lifetime_mask_summary = tf.summary.scalar('lifetime-mask-sum', tf.reduce_sum(tf.to_float(lifetime_mask)))
      summaries.append(lifetime_mask_summary)

    return summaries
