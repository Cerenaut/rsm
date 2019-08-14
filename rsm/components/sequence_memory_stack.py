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

"""SequenceMemoryStack class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import tensorflow as tf

from pagi.utils.tf_utils import tf_build_interpolate_distributions
from pagi.utils.tf_utils import tf_build_cross_entropy

from pagi.utils.np_utils import np_uniform

from pagi.components.summary_component import SummaryComponent

from rsm.components.sequence_memory_layer import SequenceMemoryLayer
from rsm.components.predictor_component import PredictorComponent


class SequenceMemoryStack(SummaryComponent):
  """
  A stack architecture of sequence memory layers
  """

  # Static names
  file = 'file'
  cache = 'cache'

  prediction = 'prediction'
  prediction_loss = 'prediction-loss'

  ensemble_top_1 = 'ensemble-top-1'
  ensemble_distribution = 'ensemble-distribution'
  ensemble_loss_sum = 'ensemble-loss-sum'
  ensemble_perplexity = 'ensemble-perplexity'

  connectivity_ff = 'ff'  # Feed-forward hierarchy ONLY.
  connectivity_bi = 'bi'  # Bidirectional hierarchy

  @staticmethod
  def default_hparams():
    """Builds an HParam object with default hyperparameters."""
    return tf.contrib.training.HParams(
        num_layers=1,
        prediction_layer=-1,  # default is top of stack
        connectivity='ff',

        # Optimizer - applies to layers and predictor
        optimizer='adam',
        loss_type='mse',
        learning_rate=0.0005,
        batch_size=80,
        momentum=0.9,
        momentum_nesterov=False,

        # Cache predictor
        cache_decay=0.9,
        cache_smart=False,

        # Ensemble
        decode_mass=0.0,  # Generate a prediction directly from the RSM
        file_mass=0.0,  # A distribution loaded from external file
        cache_mass=0.0,  # Cache of older inputs
        uniform_mass=0.0,
        input_mass=0.0,
        layer_mass=1.0,  # Default to only use layer
        ensemble_norm_eps=0.0000000001,  # 0.0001%

        mode = 'predict-input',
        #autoencode=False,

        # Memory options
        memory_summarize_input=False,
        memory_summarize_encoding=False,
        memory_summarize_decoding=False,
        memory_summarize_weights=False,
        memory_summarize_freq=False,
        memory_training_interval=[0, -1],

        # Geometry. A special value of -1 can be used to generate a 1x1 output (non-conv)
        filters_field_width=[28],
        filters_field_height=[28],
        filters_field_stride=[28],
        pool_size=[1],  # Per layer. 1 = no pooling

        cols=[160],
        cells_per_col=[3],  # 480 = 160 columns * 3 cells

        # Predictor
        predictor_training_interval=[0, -1],
        predictor_hidden_size=[200],
        predictor_nonlinearity=['leaky-relu', 'leaky-relu'],
        predictor_optimize='accuracy',  # reconstruction, accuracy
        predictor_loss_type='cross-entropy',
        predictor_keep_rate=1.0,
        predictor_l2_regularizer=0.0,
        predictor_label_smoothing=0.0,

        # Memory predictor options
        predictor_integrate_input=False,
        predictor_norm_input=True,

        # Regularization, 0=Off
        l2_f=[0.0],
        l2_r=[0.0],
        l2_b=[0.0],

        # Control statistics
        freq_update_interval=10,
        freq_learning_rate=0.1,
        freq_min=0.05, # used by lifetime sparsity mask

        hidden_nonlinearity='tanh', # used for hidden layer only
        decode_nonlinearity=['none'], # Used for decoding

        inhibition_decay=[0.1],  # controls refractory period
        feedback_decay_rate=[0.0],  # Optional integrated/exp decay feedback
        feedback_keep_rate=[1.0],  # Optional dropout on feedback
        feedback_norm=[True],

        # Sparse parameters:
        sparsity=[25],
        lifetime_sparsity_dends=False,
        lifetime_sparsity_cols=False,

        summarize_distributions=False
    )

  def update_statistics(self, session):  # pylint: disable=W0613
    """Called after a batch"""
    layers = self.get_num_layers()
    for i in range(layers):
      layer = self.get_layer(i)
      layer.update_statistics(session)

  def forget_history(self, session, history_forgetting_probability, clear_previous=False):
    """Called before a batch. Stochastically forget the recurrent history."""
    layers = self.get_num_layers()
    for i in range(layers):
      layer = self.get_layer(i)
      layer.forget_history(session, history_forgetting_probability, clear_previous)

  def update_history(self, session, history_mask, clear_previous=True):
    """Called before a batch. The external data defines a break in the data."""
    layers = self.get_num_layers()
    for i in range(layers):
      layer = self.get_layer(i)
      layer.update_history(session, history_mask, clear_previous)

  def update_recurrent_and_feedback(self):
    self.update_recurrent()
    self.update_feedback()

  def update_feedback(self):
    """If connectivity is bidirectional..."""
    layers = self.get_num_layers()
    for i in range(layers-1):  # e.g. 0,1,2 = 3 layers
      upper = i +1
      lower = i
      logging.info('Copying feedback from layer %s to layer %s', str(upper), str(lower))
      upper_layer = self.get_layer(upper)
      lower_layer = self.get_layer(lower)
      feedback_values = upper_layer.get_values(SequenceMemoryLayer.encoding)
      lower_layer.set_feedback(feedback_values)

  def update_recurrent(self):
    layers = self.get_num_layers()
    for i in range(layers):
      layer = self.get_layer(i)
      layer.update_recurrent()

  def get_num_layers(self):
    return self._hparams.num_layers

  def get_layers(self):
    return self._layers

  def get_layer(self, layer=None):
    return self._layers[layer]

  def get_prediction_layer(self):
    if self._hparams.prediction_layer < 0:
      layers = self.get_num_layers()
      return layers - 1
    return self._hparams.prediction_layer

  def get_loss(self):
    if self._hparams.predictor_optimize == 'accuracy':
      return self.get_values(SequenceMemoryStack.ensemble_loss_sum)
    return self.get_values(SequenceMemoryStack.prediction_loss)

  def build(self, input_values, input_shape, label_values, label_shape, hparams, decoder=None, name='rsm-stack'):  # pylint: disable=W0221
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
    self._decoder = decoder  # optional

    self._input_shape = input_shape
    self._label_shape = label_shape
    self._label_values = label_values

    with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):

      self._build_layers(input_values, input_shape, label_values, label_shape)

      # Predictor may be asked to generate a target image rather than a classification
      predictor_target_values = input_values
      predictor_target_shape = input_shape

      # Build a predictor based on memory input
      if self._hparams.layer_mass > 0.0:
        logging.info('Building layer predictor...')
        layer_predictor_name = 'layer-p'
        layer_predictor_input, layer_predictor_input_shape = self._build_layer_prediction_input()
        self._layer_predictor = self._build_predictor(layer_predictor_input, layer_predictor_input_shape, label_values,
                                                      label_shape, predictor_target_values, predictor_target_shape,
                                                      layer_predictor_name)
      else:
        logging.info('NOT building layer predictor.')

      # Build another prediction from the input (1st order)
      if self._hparams.input_mass > 0.0:
        logging.info('Building input predictor...')
        input_predictor_name = 'input-p'
        input_predictor_input, input_predictor_input_shape = self._build_input_prediction_input()
        self._input_predictor = self._build_predictor(input_predictor_input, input_predictor_input_shape, label_values,
                                                      label_shape, predictor_target_values, predictor_target_shape,
                                                      input_predictor_name)
      else:
        logging.info('NOT building input predictor.')

      if self._hparams.cache_mass > 0.0:
        self._build_cache_pl(label_values)
      else:
        logging.info('NOT building cache.')

      # Now build an ensemble prediction from these predictions
      if self._hparams.predictor_optimize == 'accuracy':
        self._build_ensemble_prediction()
      else:
        prediction_layer_idx = self.get_prediction_layer()
        prediction_layer = self._layers[prediction_layer_idx]

        prediction_decoding = prediction_layer.get_op(SequenceMemoryLayer.decoding)
        prediction_loss = prediction_layer.get_op(SequenceMemoryLayer.loss)

        self._dual.set_op(self.prediction, prediction_decoding)
        self._dual.set_op(self.prediction_loss, prediction_loss)


  def _build_decode_prediction(self):
    """Transform the decoding-prediction into a prediction over the classes"""
    prediction_layer_idx = self.get_prediction_layer()
    prediction_layer = self._layers[prediction_layer_idx]
    decoding = prediction_layer.get_op(SequenceMemoryLayer.decoding)  # a prediction
    prediction_logits = self._decoder.build(decoding)
    return prediction_logits

  def _build_ensemble_prediction(self):
    """Builds ensemble prediction."""
    logging.info('Building ensemble...')
    distributions = []
    distribution_mass = []
    num_classes = self._label_shape[-1]

    if self._hparams.decode_mass > 0.0:
      print('decoding...')
      decode_distribution = self._build_decode_prediction()
      print('one hot', decode_distribution)
      # decode_sum = tf.reduce_sum(decode_distribution, axis=1, keepdims=True)# + eps
      # decode_norm = decode_distribution / decode_sum
      # distributions.append(decode_norm)
      distributions.append(decode_distribution)
      distribution_mass.append(self._hparams.decode_mass)

    if self._hparams.input_mass > 0.0:
      input_prediction = self._input_predictor.get_op(PredictorComponent.prediction_softmax)
      distributions.append(input_prediction)
      distribution_mass.append(self._hparams.input_mass)

    if self._hparams.uniform_mass > 0.0:
      uniform = np_uniform(num_classes)
      distributions.append(uniform)
      distribution_mass.append(self._hparams.uniform_mass)

    if self._hparams.file_mass > 0.0:
      file_pl = self._dual.add(self.file, shape=[self._hparams.batch_size, num_classes], default_value=0.0).add_pl()
      file_sum = tf.reduce_sum(file_pl, axis=1, keepdims=True)# + eps
      file_norm = file_pl / file_sum
      distributions.append(file_norm)
      distribution_mass.append(self._hparams.file_mass)

    if self._hparams.cache_mass > 0.0:
      cache_pl = self._dual.get_pl(self.cache)
      cache_sum = tf.reduce_sum(cache_pl, axis=1, keepdims=True) + self._hparams.ensemble_norm_eps
      cache_norm = cache_pl / cache_sum
      distributions.append(cache_norm)  # Use the old cache, not with new label ofc
      distribution_mass.append(self._hparams.cache_mass)

    if self._hparams.layer_mass > 0.0:
      layer_prediction = self._layer_predictor.get_op(PredictorComponent.prediction_softmax) #prediction_softmax_op()
      distributions.append(layer_prediction)
      distribution_mass.append(self._hparams.layer_mass)

    # Build the final distribution, calculate loss
    ensemble_prediction = tf_build_interpolate_distributions(distributions, distribution_mass, num_classes)
    cross_entropy_loss = tf_build_cross_entropy(self._label_values, ensemble_prediction)  # Calculate the loss
    cross_entropy_mean = tf.reduce_mean(cross_entropy_loss)
    ensemble_perplexity = tf.exp(cross_entropy_mean)  # instantaneous perplexity (exaggerated)
    ensemble_cross_entropy_sum = tf.reduce_sum(cross_entropy_loss)
    ensemble_prediction_max = tf.argmax(ensemble_prediction, 1)

    self._dual.set_op(self.ensemble_distribution, ensemble_prediction)
    self._dual.set_op(self.ensemble_top_1, ensemble_prediction_max)
    self._dual.set_op(self.ensemble_perplexity, ensemble_perplexity)
    self._dual.set_op(self.ensemble_loss_sum, ensemble_cross_entropy_sum)

    if self._hparams.cache_mass > 0.0:
      self._build_cache_op(self._label_values, ensemble_prediction)

  def _build_cache_pl(self, labels):
    logging.info('Building cache placeholder...')
    num_classes = labels.get_shape().as_list()[1]
    cache_pl = self._dual.add(self.cache, shape=[self._hparams.batch_size, num_classes], default_value=0.0).add_pl()
    return cache_pl

  def _build_cache_op(self, labels, prediction):
    if self._hparams.cache_smart:
      self._build_smart_cache_op(labels, prediction)
    else:
      self._build_simple_cache_op(labels, prediction)

  def _build_simple_cache_op(self, labels, prediction):
    """Builds a simple caching operation."""
    del prediction

    logging.info('Building simple cache op...')

    cache_increase = labels
    cache_pl = self._dual.get_pl(self.cache)
    cache_decay = cache_pl * self._hparams.cache_decay
    cache_op = tf.maximum(cache_decay, cache_increase)  # Sets to 1 if label set
    self._dual.set_op(self.cache, cache_op)
    return cache_pl

  def _build_smart_cache_op(self, labels, prediction):
    """
    Builds a smart caching operation.

    Surprise = "how much mass was predicted before the true label"
    If surprise is more, then other values will decay faster due to normalization. We will cache the new value.
    If surprise is less, other cache values are retained, not caching the new value.
                     X
    0.1, 0.09, 0.05, 0.02  S = 0.1+0.09+0.05 = 0.24

               X
    0.1, 0.09, 0.05, 0.02  S = 0.1+0.09+0.05 = 0.19

         X
    0.1, 0.09, 0.05, 0.02  S = 0.1
    """
    logging.info('Building smart cache op...')

    masked_prediction = labels * prediction  # now only a value where label is true (TP). All FP mass is zero.
    predicted_mass = tf.reduce_sum(masked_prediction, axis=1, keepdims=True)  # The predicted mass of the true label (scalar per batch)
    more_likely_bool = tf.greater(prediction, predicted_mass)  # bool. Mask if the predicted mass was greater than the true label
    more_likely_mask = tf.to_float(more_likely_bool)
    predicted_more_likely_mass = prediction * more_likely_mask  # Mask out mass that was less than predicted true label mass
    surprise = tf.reduce_sum(predicted_more_likely_mass, axis=1, keepdims=True)  # now a vector of batch size x 1

    cache_increase = labels * surprise  # weight the increase by the surprise

    cache_pl = self._dual.get_pl(self.cache)
    cache_decay = cache_pl * self._hparams.cache_decay
    cache_op = tf.maximum(cache_decay, cache_increase)  # Smart cache
    self._dual.set_op(self.cache, cache_op)
    return cache_op

  def _build_layer_prediction_input(self):
    prediction_layer_idx = self.get_prediction_layer()
    prediction_layer = self._layers[prediction_layer_idx]
    prediction_input = prediction_layer.get_op(SequenceMemoryLayer.prediction_input)
    prediction_input_shape = prediction_layer.get_shape(SequenceMemoryLayer.prediction_input)
    return prediction_input, prediction_input_shape

  def _build_input_prediction_input(self):
    prediction_layer_idx = self.get_prediction_layer()
    prediction_layer = self._layers[prediction_layer_idx]
    prediction_input = prediction_layer.get_op(SequenceMemoryLayer.previous)
    prediction_input_shape = prediction_layer.get_shape(SequenceMemoryLayer.previous)
    return prediction_input, prediction_input_shape

  def _build_layers(self, input_values, input_shape, label_values, label_shape):
    """Build the RSM layers."""
    logging.info('Building layers...')
    self._layers = []

    layers = self.get_num_layers()
    layers_hparams = []
    layers_shapes = []

    layer_input_values = input_values
    layer_input_shape = input_shape

    # Explicitly specify batch size
    if layer_input_shape[0] < 0:
      layer_input_shape[0] = self._hparams.batch_size

    # Compute geometry of all layers
    for i in range(layers):

      layer_hparams = SequenceMemoryLayer.default_hparams()

      # copy and override parameters
      layer_hparams.optimizer = self._hparams.optimizer
      layer_hparams.loss_type = self._hparams.loss_type
      layer_hparams.learning_rate = self._hparams.learning_rate
      layer_hparams.batch_size = self._hparams.batch_size
      layer_hparams.momentum = self._hparams.momentum
      layer_hparams.momentum_nesterov = self._hparams.momentum_nesterov

      layer_hparams.mode = self._hparams.mode
      #layer_hparams.autoencode = self._hparams.autoencode

      layer_hparams.summarize_input = self._hparams.memory_summarize_input
      layer_hparams.summarize_encoding = self._hparams.memory_summarize_encoding
      layer_hparams.summarize_decoding = self._hparams.memory_summarize_decoding
      layer_hparams.summarize_weights = self._hparams.memory_summarize_weights
      layer_hparams.summarize_freq = self._hparams.memory_summarize_freq

      layer_hparams.training_interval = self._hparams.memory_training_interval

      layer_hparams.hidden_nonlinearity = self._hparams.hidden_nonlinearity

      layer_hparams.predictor_use_input = False
      layer_hparams.predictor_inc_input = False

      # Compute conv geometry
      ih = layer_input_shape[1]
      iw = layer_input_shape[2]
      fh = self._hparams.filters_field_height[i]
      fw = self._hparams.filters_field_width[i]
      fs = self._hparams.filters_field_stride[i]

      if fh < 0:
        fh = ih
      if fw < 0:
        fw = iw
      if fs < 0:
        fs = max(fh, fw)

      layer_hparams.filters_field_height = fh
      layer_hparams.filters_field_width = fw
      layer_hparams.filters_field_stride = fs

      # Depth dimension - num filters
      layer_hparams.cols = self._hparams.cols[i]
      layer_hparams.cells_per_col = self._hparams.cells_per_col[i]

      layer_hparams.freq_update_interval = self._hparams.freq_update_interval
      layer_hparams.freq_learning_rate = self._hparams.freq_learning_rate
      layer_hparams.freq_min = self._hparams.freq_min

      layer_hparams.predictor_norm_input = self._hparams.predictor_norm_input
      layer_hparams.predictor_integrate_input = self._hparams.predictor_integrate_input

      layer_hparams.l2_f = self._hparams.l2_f[i]
      layer_hparams.l2_r = self._hparams.l2_r[i]
      layer_hparams.l2_b = self._hparams.l2_b[i]

      layer_hparams.decode_nonlinearity = self._hparams.decode_nonlinearity[i]
      layer_hparams.inhibition_decay = self._hparams.inhibition_decay[i]
      layer_hparams.feedback_decay_rate = self._hparams.feedback_decay_rate[i]
      layer_hparams.feedback_keep_rate = self._hparams.feedback_keep_rate[i]
      layer_hparams.feedback_norm = self._hparams.feedback_norm[i]

      layer_hparams.sparsity = self._hparams.sparsity[i]
      layer_hparams.lifetime_sparsity_dends = self._hparams.lifetime_sparsity_dends
      layer_hparams.lifetime_sparsity_cols = self._hparams.lifetime_sparsity_cols

      logging.debug('layer: %d h/w/s: %d/%d/%d',
                    i,
                    layer_hparams.filters_field_height,
                    layer_hparams.filters_field_width,
                    layer_hparams.filters_field_stride)

      layer_shape = SequenceMemoryLayer.get_encoding_shape_4d(layer_input_shape, layer_hparams)

      layers_hparams.append(layer_hparams)
      layers_shapes.append(layer_shape)

      layer_input_shape = layer_shape  # for next layer

      # Max-pooling - affects next layer input shape
      pool_size = self._hparams.pool_size[i]
      if pool_size > 1:
        logging.info('Pooling %s:1', str(pool_size))
        layer_input_shape[1] = int(layer_input_shape[1] / pool_size)
        layer_input_shape[2] = int(layer_input_shape[2] / pool_size)

    # 2nd pass - for bi-directional connectivity
    layer_input_values = input_values
    layer_input_shape = input_shape

    for i in range(layers):

      layer_hparams = layers_hparams[i]  # retrieve precalculated hparams

      layer = SequenceMemoryLayer()
      layer_name = 'layer-'+str(i+1)

      layer_feedback_shape = None  # Connectivity FF
      if self._hparams.connectivity == SequenceMemoryStack.connectivity_bi:
        logging.info('Bidirectional connectivity enabled.')
        if i < (layers-1):
          layer_feedback_shape = layers_shapes[i+1]
      else:
        logging.info('Feed-forward connectivity enabled.')


      layer.build(layer_input_values, layer_input_shape, layer_hparams, name=layer_name, encoding_shape=None,
                  feedback_shape=layer_feedback_shape, target_shape=label_shape, target_values=label_values)

      self._layers.append(layer)

      # link layers
      # This means it'll update with the latest state of input in lower layer WRT current input
      output_encoding = layer.get_op(SequenceMemoryLayer.encoding) # 4d, protected with StopGradient
      layer_input_values = output_encoding

      pool_size = self._hparams.pool_size[i]
      if pool_size > 1:
        logging.info('Pooling %s:1', str(pool_size))
        pool_sizes = [1, pool_size, pool_size, 1]
        pool_strides = [1, pool_size, pool_size, 1]
        layer_input_values = tf.nn.max_pool(output_encoding, pool_sizes, pool_strides, padding='SAME')

      #print( "output encoding, ", output_encoding)
      layer_input_shape = layer_input_values.shape.as_list()

  def _build_predictor(self, prediction_input, prediction_input_shape, label_values, label_shape, target_values,
                       target_shape, name='p'):
    """Build the predictor using outputs from RSM layers."""

    # Build the predictor
    predictor = PredictorComponent()
    predictor_hparams = predictor.default_hparams()

    predictor_hparams.optimizer = self._hparams.optimizer
    predictor_hparams.optimize = self._hparams.predictor_optimize
    predictor_hparams.loss_type = self._hparams.predictor_loss_type
    predictor_hparams.learning_rate = self._hparams.learning_rate
    predictor_hparams.batch_size = self._hparams.batch_size
    predictor_hparams.momentum = self._hparams.momentum
    predictor_hparams.momentum_nesterov = self._hparams.momentum_nesterov

    predictor_hparams.uniform_mass = 0.0
    predictor_hparams.training_interval = self._hparams.predictor_training_interval
    predictor_hparams.nonlinearity = self._hparams.predictor_nonlinearity
    predictor_hparams.hidden_size = self._hparams.predictor_hidden_size
    predictor_hparams.keep_rate = self._hparams.predictor_keep_rate
    predictor_hparams.l2_regularizer = self._hparams.predictor_l2_regularizer
    predictor_hparams.label_smoothing = self._hparams.predictor_label_smoothing

    predictor.build(prediction_input, prediction_input_shape, label_values, label_shape, target_values, target_shape,
                    predictor_hparams, name=name)
    return predictor

  # BATCH INTERFACE ------------------------------------------------------------------
  def update_feed_dict(self, feed_dict, batch_type='training'):
    """Updates the feed dict in each layer."""
    if self._hparams.decode_mass > 0.0:
      self._decoder.update_feed_dict(feed_dict, batch_type)

    # Layers
    layers = self.get_num_layers()
    for i in range(layers):
      layer = self.get_layer(i)
      layer.update_feed_dict(feed_dict, batch_type)

    # Predictor
    if self._hparams.layer_mass > 0.0:
      self._layer_predictor.update_feed_dict(feed_dict, batch_type)
    if self._hparams.input_mass > 0.0:
      self._input_predictor.update_feed_dict(feed_dict, batch_type)

    if self._hparams.file_mass > 0.0:
      file_dual = self._dual.get(self.file)
      file_pl = file_dual.get_pl()
      file_values = file_dual.get_values()  # assume this is populated somehow by workflow.

      feed_dict.update({
          file_pl: file_values
      })

    # Cache
    if self._hparams.cache_mass > 0.0:
      cache = self._dual.get(SequenceMemoryStack.cache)
      cache_pl = cache.get_pl()
      cache_values = cache.get_values()

      feed_dict.update({
          cache_pl: cache_values
      })

  def add_fetches(self, fetches, batch_type='training'):
    """Add fetches in each layer for session run call."""
    # Layers
    layers = self.get_num_layers()
    for i in range(layers):
      layer = self.get_layer(i)
      layer.add_fetches(fetches, batch_type)

    # Predictors
    if self._hparams.layer_mass > 0.0:
      self._layer_predictor.add_fetches(fetches, batch_type)
    if self._hparams.input_mass > 0.0:
      self._input_predictor.add_fetches(fetches, batch_type)

    # Cache
    fetches[self.name] = {}
    if self._hparams.cache_mass > 0.0:
      fetches[self.name].update({
          SequenceMemoryStack.cache: self._dual.get_op(SequenceMemoryStack.cache),
      })

    # Ensemble
    if self._hparams.predictor_optimize == 'accuracy':
      fetches[self.name].update({
          self.ensemble_distribution: self._dual.get_op(self.ensemble_distribution),
          self.ensemble_top_1: self._dual.get_op(self.ensemble_top_1),
          self.ensemble_perplexity: self._dual.get_op(self.ensemble_perplexity),
          self.ensemble_loss_sum: self._dual.get_op(self.ensemble_loss_sum)
      })
    else:
      fetches[self.name].update({
          self.prediction: self._dual.get_op(self.prediction),
          self.prediction_loss: self._dual.get_op(self.prediction_loss)
      })

    # Summaries
    super().add_fetches(fetches, batch_type)

  def set_fetches(self, fetched, batch_type='training'):
    """Set fetches in each layer."""
    # Layers
    layers = self.get_num_layers()
    for i in range(layers):
      layer = self.get_layer(i)
      layer.set_fetches(fetched, batch_type)

    # Predictors
    if self._hparams.layer_mass > 0.0:
      self._layer_predictor.set_fetches(fetched, batch_type)
    if self._hparams.input_mass > 0.0:
      self._input_predictor.set_fetches(fetched, batch_type)

    names = []

    # Cache
    if self._hparams.cache_mass > 0.0:
      names.append(SequenceMemoryStack.cache)

    # Ensemble
    if self._hparams.predictor_optimize == 'accuracy':
      names.append(self.ensemble_distribution)
      names.append(self.ensemble_top_1)
      names.append(self.ensemble_perplexity)
      names.append(self.ensemble_loss_sum)
    else:
      names.append(self.prediction)
      names.append(self.prediction_loss)

    self._dual.set_fetches(fetched, names)

    # Summaries
    super().set_fetches(fetched, batch_type)

  def write_summaries(self, step, writer, batch_type='training'):
    """Write the TensorBoard summaries for each layer."""
    # Layers
    layers = self.get_num_layers()
    for i in range(layers):
      layer = self.get_layer(i)
      layer.write_summaries(step, writer, batch_type)

    # Predictors
    if self._hparams.layer_mass > 0.0:
      self._layer_predictor.write_summaries(step, writer, batch_type)
    if self._hparams.input_mass > 0.0:
      self._input_predictor.write_summaries(step, writer, batch_type)

    # Summaries
    super().write_summaries(step, writer, batch_type)

  def build_summaries(self, batch_types=None, max_outputs=3, scope=None):
    """Builds the summaries for each layer."""

    # Layers
    layers = self.get_num_layers()
    for i in range(layers):
      layer = self.get_layer(i)
      layer.build_summaries(batch_types)

    # Predictors
    if self._hparams.layer_mass > 0.0:
      self._layer_predictor.build_summaries(batch_types)
    if self._hparams.input_mass > 0.0:
      self._input_predictor.build_summaries(batch_types)

    # Summaries
    super().build_summaries(batch_types, max_outputs, scope)

  def _build_summaries(self, batch_type, max_outputs=3):

    # Ensemble interpolation
    summaries = []

    if self._hparams.summarize_distributions:
      ensemble_perplexity = self._dual.get_op(self.ensemble_perplexity)
      ensemble_cross_entropy_sum = self._dual.get_op(self.ensemble_loss_sum)
      #ensemble_top_1 = self._dual.get_op(self.ensemble_top_1)

      summaries.append(tf.summary.scalar('mean_perplexity', tf.reduce_mean(ensemble_perplexity)))
      summaries.append(tf.summary.scalar(self.ensemble_loss_sum, ensemble_cross_entropy_sum))
      #summaries.append(tf.summary.scalar(self.ensemble_top_1, ensemble_top_1))

      ensemble_distribution = self._dual.get_op(self.ensemble_distribution)
      #ensemble_distribution = tf.Print(ensemble_distribution, [ensemble_distribution], 'DIST ', summarize=48)
      ensemble_shape = ensemble_distribution.get_shape().as_list()
      ensemble_shape_4d = [ensemble_shape[0],1,ensemble_shape[1],1]
      #print('>>>>>', ensemble_shape_4d)
      ensemble_distribution_reshape = tf.reshape(ensemble_distribution, ensemble_shape_4d)
      p_summary_op = tf.summary.image(self.ensemble_distribution, ensemble_distribution_reshape, max_outputs=max_outputs)
      summaries.append(p_summary_op)

    if len(summaries) == 0:
      return None
    return summaries
