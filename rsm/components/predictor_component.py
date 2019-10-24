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

"""PredictorComponent class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np
import tensorflow as tf

from pagi.utils import image_utils
from pagi.utils.layer_utils import type_activation_fn
from pagi.utils.tf_utils import tf_create_optimizer
from pagi.utils.tf_utils import tf_do_training

from pagi.components.summary_component import SummaryComponent


class PredictorComponent(SummaryComponent):
  """
  TODO: Rename to ClassifierComponent (except: name taken.) Maybe NeuralClassifierComponent
  A stack of fully-connected dense layers for a supervised function approximation purpose,
  such as classification.
  """

  # Static names
  training = 'training'
  encoding = 'encoding'

  loss = 'loss'
  keep = 'keep'
  accuracy = 'accuracy'

  prediction = 'prediction'
  prediction_loss = 'prediction-loss'
  prediction_loss_sum = 'prediction-loss-sum'
  prediction_reshape = 'prediction-reshape'
  prediction_softmax = 'prediction-softmax'
  prediction_correct = 'prediction-correct'
  prediction_correct_sum = 'prediction-correct-sum'
  prediction_max = 'prediction-max'
  prediction_perplexity = 'prediction-perplexity'

  @staticmethod
  def default_hparams():
    """Builds an HParam object with default hyperparameters."""
    return tf.contrib.training.HParams(

        # Optimizer options
        training_interval=[0, -1],
        loss_type='cross-entropy',  # or mse
        optimize='accuracy',  # target, accuracy
        optimizer='adam',
        learning_rate=0.0005,

        nonlinearity=['leaky-relu'],
        bias=True,
        init_sd=0.03,

        # Geometry
        batch_size=80,
        hidden_size=[200],

        # Norm
        norm_type = 'sum',  # Or None, currently
        norm_eps = 1.0e-11,

        # Regularization
        keep_rate=1.0,
        l2=0.0,
        label_smoothing=0.0,  # adds a uniform weight to the training target distribution

        # Summary options
        summarize=True,
        summarize_input=False,
        summarize_distribution=False
    )

  def get_loss(self):
    return self._loss

  def reset(self):
    """Reset the trained/learned variables and all other state of the component to a new random state."""
    self._loss = 0.0

    # TODO this should reinitialize all the variables..

  def build(self, input_values, input_shape, label_values, label_shape, target_values, target_shape, hparams,
            name='predictor'):  # pylint: disable=W0221
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
    self._training_batch_count = 0

    self._input_shape = input_shape
    self._input_values = input_values

    self._label_shape = label_shape
    self._label_values = label_values

    self._target_shape = target_shape
    self._target_values = target_values

    with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
      self._build()
      self._build_optimizer()

    self.reset()

  def _build_initializers(self):
    """Override to change the initializer. Returns weights and biases init."""
    # TF Default: glorot_uniform_initializer
    # Source: https://www.tensorflow.org/api_docs/python/tf/layers/Dense

    # "Smart"
    # https://adventuresinmachinelearning.com/weight-initialization-tutorial-tensorflow/
    #w_factor = 1.0  # factor=1.0 for Xavier, 2.0 for He
    #w_mode = 'FAN_IN'
    #w_mode = 'FAN_AVG'
    #kernel_initializer = tf.contrib.layers.variance_scaling_initializer(factor=w_factor, mode=w_mode, uniform=False)

    # Normal
    init_sd = self._hparams.init_sd
    kernel_initializer = tf.random_normal_initializer(stddev=init_sd)
    bias_initializer = kernel_initializer
    return kernel_initializer, bias_initializer

  def _build_input_norm(self, input_4d, input_shape_4d):
    """Normalize/scale the input using the sum of the inputs."""
    # Optionally apply a norm to make input constant sum
    # NOTE: Assuming here it is CORRECT to norm over conv w,h
    if self._hparams.norm_type is not None:
      if self._hparams.norm_type == 'sum':
        eps = self._hparams.norm_eps
        sum_input = tf.reduce_sum(input_4d, axis=[1, 2, 3], keepdims=True) + eps
        norm_input_4d = tf.divide(input_4d, sum_input)

        # TODO investigate alternative norm, e.g. frobenius norm:
        #frobenius_norm = tf.sqrt(tf.reduce_sum(tf.square(input_values_next), axis=[1, 2, 3], keepdims=True))
        # Layer norm..? There has to be a better norm than this.
        # https://mlexplained.com/2018/11/30/an-overview-of-normalization-methods-in-deep-learning/
        # https://mlexplained.com/2018/01/10/an-intuitive-explanation-of-why-batch-normalization-really-works-normalization-in-deep-learning-part-1/
    else:
      norm_input_4d = input_4d
    return norm_input_4d

  def _build(self):
    """Build the autoencoder network"""

    # Norm inputs before flattening (e.g. to exploit local norm concepts)
    input_values = self._build_input_norm(self._input_values, self._input_shape)

    # Flatten inputs
    input_volume = np.prod(self._input_shape[1:])
    input_shape_1d = [self._hparams.batch_size, input_volume]
    input_values_1d = tf.reshape(input_values, input_shape_1d)

    target_shape_list = self._target_values.get_shape().as_list()
    target_volume = np.prod(target_shape_list[1:])

    # Calculate output layer size
    num_classes = self._label_shape[-1]
    output_layer_size = 0
    if self._hparams.optimize == self.accuracy:
      output_layer_size = num_classes  # Number of classes
    else:
      output_layer_size = target_volume  # Number of pixels


    # Define all layer sizes
    hidden_layer_sizes = self._hparams.hidden_size  # An array
    layer_sizes = hidden_layer_sizes.copy()
    layer_sizes.append(output_layer_size)
    self.layer_sizes = layer_sizes
    self.num_layers = len(layer_sizes)
    self.layers = []

    for i, layer_size in enumerate(layer_sizes):
      activation = type_activation_fn(self._hparams.nonlinearity[i])
      kernel_initializer, bias_initializer = self._build_initializers()

      layer = tf.layers.Dense(layer_size,
                              activation=activation,
                              kernel_initializer=kernel_initializer,
                              bias_initializer=bias_initializer,
                              use_bias=self._hparams.bias,
                              name='prediction_layer_' + str(i + 1))
      logging.info('Predictor layer: %d has size: %d and fn: %s ', i, layer_size, str(activation))
      self.layers.append(layer)

    # Optional dropout of input bits at each layer
    keep_pl = tf.placeholder_with_default(1.0, shape=())
    self._dual.add(self.keep).set_pl(keep_pl)

    # Connect layers
    layer_input = input_values_1d

    for i in range(self.num_layers):

      layer = self.layers[i]
      if self._hparams.keep_rate < 1.0:
        layer_input = tf.nn.dropout(layer_input, keep_pl)  # Note, a scaling is applied

      layer_output = layer(layer_input)
      layer_input = layer_output

    prediction = layer_input  # output of last layer (1d)
    prediction_sg = tf.stop_gradient(prediction)
    self._dual.set_op(self.prediction, prediction_sg)

    softmax = tf.nn.softmax(logits=prediction_sg)
    self._dual.set_op(self.prediction_softmax, softmax)

    prediction_loss = None  # Mean
    prediction_loss_sum = None

    if self._hparams.optimize == self.accuracy:
      # We should predict the current labels l_t without using the current input x_t
      # l_t is the label corresponding to x_t so it isn't a challenge to go from x_t to l_t
      # y_t = f( x_t-1,  y_t-1 )
      # p_t = f( y_t )
      # predict l_t | p_t
      logging.info('Predictor optimizing classification accuracy.')
      labels = self._label_values

      if self._hparams.label_smoothing > 0:
        logging.info('Predictor using label smoothing: ')
        ce_loss = tf.losses.softmax_cross_entropy(logits=prediction, onehot_labels=labels,
                                                  label_smoothing=self._hparams.label_smoothing)
      else:
        logging.info('Predictor NOT using label smoothing: ')
        ce_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=labels)

      ce_loss_sum = tf.reduce_sum(ce_loss)
      ce_loss_mean = tf.reduce_mean(ce_loss)
      prediction_loss = ce_loss_mean
      prediction_loss_sum = ce_loss_sum
      self._dual.set_op(self.prediction_loss_sum, prediction_loss_sum)

      # Correctness metrics
      prediction_max = tf.argmax(prediction_sg, 1)
      self._dual.set_op(self.prediction_max, prediction_max)

      labels_max = tf.argmax(labels, 1)
      correct_predictions = tf.equal(labels_max, prediction_max)
      correct_predictions = tf.cast(correct_predictions, tf.float32)
      self._dual.set_op(self.prediction_correct, correct_predictions)

      sum_correct_predictions = tf.reduce_sum(correct_predictions)
      self._dual.set_op(self.prediction_correct_sum, sum_correct_predictions)

      # Perplexity metrics
      perplexity = tf.exp(ce_loss_mean)
      self._dual.set_op(self.prediction_perplexity, perplexity)

    else:  # predictor constructs some arbitrary thing
      # We should predict the current input x_t without using the current input x_t, instead only x_t-1
      # y_t = f( x_t-1,  y_t-1 )
      # y_t-1 = f( x_t-2, y_t-2 )
      # p_t = f( y_t )
      # predict x_t | p_t
      #         x_t | y_t
      #         x_t | x_t-1, y_t-1
      logging.info('Predictor optimizing reconstruction (MSE loss).')

      # Predictions may need reshaping:
      prediction_reshape = tf.reshape(prediction, self._target_shape)
      self._dual.set_op(self.prediction_reshape, tf.stop_gradient(prediction_reshape))
      prediction_loss = tf.losses.mean_squared_error(self._target_values, prediction_reshape)

    # Both paths: Set the prediction loss
    self._dual.set_op(self.prediction_loss, prediction_loss)

    # Both paths: Regularization
    if self._hparams.l2 == 0.0:
      logging.info('NOT adding L2 weight regularization to predictor.')
      self._dual.set_op(self.loss, prediction_loss)

    else:  # L2 weight regularization
      logging.info('Adding L2 weight regularization to predictor.')
      all_losses = []
      all_losses.append(prediction_loss)

      layer_input_shape = self._input_shape

      for i in range(self.num_layers):
        layer = self.layers[i]

        logging.debug('Pred. layer ' + str(i) + ' input shape: ' + str(layer_input_shape))

        layer_output_shape = [self._hparams.batch_size, self.layer_sizes[i]]
        logging.debug('Pred. layer ' + str(i) + ' output shape: ' + str(layer_output_shape))

        num_inputs = np.prod(layer_input_shape[1:])
        num_cells = layer.weights[1].get_shape().as_list()[0]
        logging.debug('Pred. layer ' + str(i) + ' #inputs: ' + str(num_inputs))
        logging.debug('Pred. layer ' + str(i) + ' #cells: ' + str(num_cells))
        logging.debug('Pred. layer ' + str(i) + ' #weights: ' + str(layer.weights))
        l2_loss_w = tf.nn.l2_loss(layer.weights[0])
        l2_loss_b = tf.nn.l2_loss(layer.weights[1])
        l2_loss = tf.reduce_sum(l2_loss_w) + tf.reduce_sum(l2_loss_b)

        # Make the L2 loss scaling invariant to number of weights. the +1 is for biases
        #l2_normalizer = 1.0 / (num_cells * (num_inputs+1.0))
        #l2_scale = l2_normalizer * self._hparams.l2
        #l2_loss_scaled = l2_loss * l2_scale
        l2_loss_scaled = l2_loss * self._hparams.l2
        all_losses.append(l2_loss_scaled)

        i += 1
        layer_input_shape = layer_output_shape

      all_losses_op = tf.add_n(all_losses)
      self._dual.set_op(self.loss, all_losses_op)

  def _build_optimizer(self):
    """Setup the training operations"""
    with tf.variable_scope('optimizer', reuse=tf.AUTO_REUSE):
      self._optimizer = tf_create_optimizer(self._hparams)
      loss_op = self._dual.get_op(self.loss)
      training_op = self._optimizer.minimize(loss_op, global_step=tf.train.get_or_create_global_step(),
                                             name='training_op')
      self._dual.set_op(self.training, training_op)

  # INTERFACE ------------------------------------------------------------------
  def update_feed_dict(self, feed_dict, batch_type='training'):
    """Update the feed dict for the session call."""
    keep_rate = None
    if batch_type == self.training:
      keep_rate = self._hparams.keep_rate # reduced rate during training
    if batch_type == self.encoding:
      keep_rate = 1.0 # No dropout at test time
    logging.debug('Predictor keep rate: %f', keep_rate)

    keep = self._dual.get(self.keep)
    keep_pl = keep.get_pl()

    feed_dict.update({
        keep_pl: keep_rate
    })

  def add_fetches(self, fetches, batch_type='training'):
    """Add the fetches for the session call."""
    # Predict
    fetches[self.name] = {
        self.loss: self._dual.get_op(self.loss),
        self.prediction: self._dual.get_op(self.prediction)
    }

    # Classify
    if self._hparams.optimize == self.accuracy:
      fetches[self.name].update({
          self.prediction_loss_sum: self._dual.get_op(self.prediction_loss_sum),
          self.prediction_correct: self._dual.get_op(self.prediction_correct),
          self.prediction_max: self._dual.get_op(self.prediction_max),
          self.prediction_softmax: self._dual.get_op(self.prediction_softmax),
          self.prediction_perplexity: self._dual.get_op(self.prediction_perplexity),
          'label_values': self._label_values
      })

    # Decide whether to train. Default training_interval: [0,-1]
    do_training = tf_do_training(batch_type, self._hparams.training_interval, self._training_batch_count,
                                 name=self.name)
    if do_training:
      fetches[self.name].update({
          self.training: self._dual.get_op(self.training)
      })

    # Summaries
    super().add_fetches(fetches, batch_type)

  def set_fetches(self, fetched, batch_type='training'):
    """Set the fetches from the output of the session call."""
    if batch_type == self.training:
      self._training_batch_count += 1

    self_fetched = fetched[self.name]
    self._loss = self_fetched[self.loss]

    names = [self.prediction]
    if self._hparams.optimize == self.accuracy:
      names.append(self.prediction_loss_sum)
      names.append(self.prediction_correct)
      names.append(self.prediction_max)
      names.append(self.prediction_softmax)
      names.append(self.prediction_perplexity)

    self._dual.set_fetches(fetched, names)

    super().set_fetches(fetched, batch_type)

  def _build_summaries(self, batch_type, max_outputs=3):
    """Build the summaries for TensorBoard."""
    del batch_type

    # Summary parameters
    max_perplexity = 1000.0
    concat = True  # Makes it easier to compare input and decode/reconstruction output together
    concat_axis = 1  # 1 = Y

    summaries = []
    if not self._hparams.summarize:
      return summaries

    # Input labels
    if self._hparams.summarize_input:
      labels = tf.reshape(self._label_values, shape=[-1, 1, self._label_values.get_shape()[1], 1])
      logging.debug('Input labels shape: %s', str(labels))
      labels_summary_op = tf.summary.image('labels', labels, max_outputs=max_outputs)
      summaries.append(labels_summary_op)

      # Input values (to do the prediction with)
      logging.debug('Input values shape: %s', str(self._input_shape))
      summary_input_shape = self._build_summary_input_shape(self._input_shape)
      input_reshape = tf.reshape(self._input_values, summary_input_shape)
      input_summary_op = tf.summary.image('input-summary', input_reshape, max_outputs=max_outputs)
      summaries.append(input_summary_op)

    prediction_op = None
    if self._hparams.summarize_distribution:
      # The actual raw prediction
      prediction_op = self.get_prediction_op()  # 1d

      # reshape raw prediction to labels shape.
      summary_prediction_shape = [-1, 1, self._label_values.get_shape()[1], 1]
      prediction = tf.reshape(prediction_op, shape=summary_prediction_shape)
      prediction_summary_op = tf.summary.image('prediction-summary', prediction, max_outputs=max_outputs)
      summaries.append(prediction_summary_op)

    # Prediction loss
    if self._dual.get(self.prediction_loss):
      prediction_loss_summary = tf.summary.scalar(self.prediction_loss, self._dual.get_op(self.prediction_loss))
      summaries.append(prediction_loss_summary)

    # Mode-specific summaries
    if self._hparams.optimize == self.accuracy:
      if self._dual.get(self.prediction_correct_sum):
        total_correct_predictions_summary = tf.summary.scalar(self.prediction_correct_sum,
                                                              self._dual.get_op(self.prediction_correct_sum))
        summaries.append(total_correct_predictions_summary)

      if self._dual.get(self.prediction_perplexity):
        perplexity = self._dual.get_op(self.prediction_perplexity)
        perplexity_clipped = tf.minimum(max_perplexity, perplexity)
        perplexity_summary = tf.summary.scalar(self.prediction_perplexity, perplexity_clipped)
        summaries.append(perplexity_summary)

    else: # reconstruct a target tensor

      # Work out the ideal summary shape, given the actual target shape.
      summary_target_shape = self._build_summary_target_shape(self._target_shape)
      prediction = self._dual.get_op(self.prediction_reshape)  # Already target shape
      prediction_reshape = tf.reshape(prediction, summary_target_shape)
      target_reshape = tf.reshape(self._target_values, summary_target_shape)

      if concat:
        summary_target = tf.concat(
            [target_reshape, prediction_reshape], axis=concat_axis)
        target_summary_op = tf.summary.image('target-prediction', summary_target, max_outputs=max_outputs)
        summaries.append(target_summary_op)
      else:
        target_summary_op = tf.summary.image('target', target_reshape, max_outputs=max_outputs)
        summaries.append(target_summary_op)

        reconstruction_summary_op = tf.summary.image('reconstruction', prediction_reshape, max_outputs=max_outputs)
        summaries.append(reconstruction_summary_op)

        if prediction_op is not None:
          prediction_reconstruction_op = tf.summary.image('prediction-recon-summary', prediction,
                                                          max_outputs=max_outputs)
          summaries.append(prediction_reconstruction_op)

    return summaries

  def _build_summary_target_shape(self, target_shape):
    summary_shape = image_utils.get_image_summary_shape(target_shape)
    return summary_shape

  def _build_summary_input_shape(self, input_shape):
    summary_shape = image_utils.get_image_summary_shape(input_shape)
    return summary_shape
