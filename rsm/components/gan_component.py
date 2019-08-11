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

"""GANComponent class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from pagi.components.summary_component import SummaryComponent

from pagi.utils import image_utils
from pagi.utils.layer_utils import type_activation_fn


class GANComponent(SummaryComponent):
  """
  Sequence Memory layer component based on K-Sparse Convolutional Autoencoder with variable-order feedback dendrites.
  """

  @staticmethod
  def default_hparams():
    """Builds an HParam object with default hyperparameters."""
    return tf.contrib.training.HParams(

        batch_size=80,

        # Optimizer options
        optimizer='adam',
        momentum=0.9,  # Ignore if adam

        # Generator-specific options
        generator_num_layers=2,
        generator_filters=[32, 64],
        generator_filters_field_width=[6, 6],
        generator_filters_field_height=[6, 6],
        generator_filters_field_stride=[3, 3],
        generator_nonlinearity=['relu', 'relu'],
        generator_loss_lambda=0.9,
        generator_learning_rate=0.0005,

        # Discriminator-specific options
        discriminator_num_layers=2,
        discriminator_filters=[32, 64],
        discriminator_filters_field_width=[6, 6],
        discriminator_filters_field_height=[6, 6],
        discriminator_filters_field_stride=[3, 3],
        discriminator_nonlinearity=['leaky-relu', 'leaky-relu'],
        discriminator_learning_rate=0.0005
    )

  class BaseNetwork:
    """A base class with common methods for generator and discriminator."""
    def __init__(self, hparams, output_size, output_nonlinearity, name):
      self.name = name
      self.hparams = hparams
      self.output_size = output_size
      self.output_nonlinearity = output_nonlinearity

      self.layers = self.build_layers()

    def build_layers(self):
      """Build the network."""
      with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE, auxiliary_name_scope=False):
        layers = []

        for i in range(self.hparams.num_layers):
          layer = tf.layers.Conv2D(
              filters=self.hparams.filters[i],
              kernel_size=[
                  self.hparams.filters_field_height[i],
                  self.hparams.filters_field_width[i]
              ],
              strides=self.hparams.filters_field_stride[i],
              activation=type_activation_fn(self.hparams.nonlinearity[i])
          )
          layers.append(layer)

        output_layer = tf.layers.Dense(self.output_size, activation=None)
        layers.append(output_layer)

        return layers

    def __call__(self, inputs):
      """
      Args:
        inputs: A tensor with shape = (b, h, w, c)
      Returns:
        It return the logits and the prediction with nonlinearity.
      """
      with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE, auxiliary_name_scope=False):
        outputs = inputs
        for layer in self.layers:
          outputs = layer(outputs)
        return outputs, self.output_nonlinearity(outputs)

    def trainable_variables(self):
      return tf.trainable_variables(self.name)

  class Generator(BaseNetwork):
    """Builds the generator network."""
    def __init__(self, hparams, output_size, output_nonlinearity=tf.nn.sigmoid, name='generator'):
      generator_hparams = tf.contrib.training.HParams(
          num_layers=hparams.generator_num_layers,
          filters=hparams.generator_filters,
          filters_field_height=hparams.generator_filters_field_height,
          filters_field_width=hparams.generator_filters_field_width,
          filters_field_stride=hparams.generator_filters_field_stride,
          nonlinearity=hparams.generator_nonlinearity
      )

      super(GANComponent.Generator, self).__init__(
          generator_hparams, output_size, output_nonlinearity, name)

    def loss(self, fake_logits):
      return tf.reduce_mean(
          tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake_logits),
                                                  logits=fake_logits,
                                                  name='gen_adv_loss')
      )

  class Discriminator(BaseNetwork):
    """Builds the discriminator network."""
    def __init__(self, hparams, output_size=1, output_nonlinearity=tf.nn.sigmoid, name='discriminator'):
      discriminator_hparams = tf.contrib.training.HParams(
          num_layers=hparams.discriminator_num_layers,
          filters=hparams.discriminator_filters,
          filters_field_height=hparams.discriminator_filters_field_height,
          filters_field_width=hparams.discriminator_filters_field_width,
          filters_field_stride=hparams.discriminator_filters_field_stride,
          nonlinearity=hparams.discriminator_nonlinearity
      )

      super(GANComponent.Discriminator, self).__init__(
          discriminator_hparams, output_size, output_nonlinearity, name)

    def loss(self, real_logits, fake_logits):
      """Build the discriminator loss."""
      real_loss = tf.reduce_mean(
          tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real_logits),
                                                  logits=real_logits,
                                                  name='real_loss')
      )

      fake_loss = tf.reduce_mean(
          tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake_logits),
                                                  logits=fake_logits,
                                                  name='fake_loss')
      )

      return real_loss + fake_loss

  def build(self, gen_input_shape, real_input_shape, hparams, name='gan', encoding_shape=None):
    """Initializes the model parameters.

    Args:
        gen_input_shape: The shape of the generator input, for display (internal is vectorized)
        real_input_shape: The shape of the real input, for display (internal is vectorized)
        hparams: The hyperparameters for the model as tf.contrib.training.HParams.
        name: A globally unique graph name used as a prefix for all tensors and ops.
        encoding_shape: The shape to be used to display encoded (hidden layer) structures
    """
    self.name = name
    self.gen_name = 'generator'
    self.disc_name = 'discriminator'

    self._hparams = hparams
    self._gen_input_shape = gen_input_shape
    self._real_input_shape = real_input_shape
    self._encoding_shape = encoding_shape

    with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
      with tf.name_scope('inputs'):
        gen_inputs_pl = self._dual.add('gen_inputs', shape=gen_input_shape, default_value=0.0).add_pl()
        real_inputs_pl = self._dual.add('real_inputs', shape=real_input_shape, default_value=0.0).add_pl()

      with tf.name_scope(self.gen_name):
        self._generator = self.Generator(hparams=self._hparams,
                                         output_size=np.prod(real_input_shape[1:]),
                                         output_nonlinearity=tf.nn.sigmoid,
                                         name=self.gen_name)

        _, fake_output = self._generator(gen_inputs_pl)
        fake_output = tf.reshape(fake_output, real_input_shape)

        self._dual.set_op('fake_output', fake_output)

      with tf.name_scope(self.disc_name):
        self._discriminator = self.Discriminator(hparams=self._hparams,
                                                 output_size=1,
                                                 output_nonlinearity=tf.nn.sigmoid,
                                                 name=self.disc_name)

        real_logits, real_score = self._discriminator(real_inputs_pl)
        fake_logits, fake_score = self._discriminator(fake_output)

        self._dual.set_op('real_score', real_score)
        self._dual.set_op('fake_score', fake_score)

      with tf.name_scope(self.gen_name + '/loss'):
        gen_adv_loss = self._generator.loss(fake_logits)
        gen_mse_loss = tf.losses.mean_squared_error(real_inputs_pl, fake_output)
        gen_total_loss = gen_mse_loss + (self._hparams.generator_loss_lambda * gen_adv_loss)

        self._dual.set_op('gen_adv_loss', gen_adv_loss)
        self._dual.set_op('gen_mse_loss', gen_mse_loss)
        self._dual.set_op('gen_total_loss', gen_total_loss)

      with tf.variable_scope(self.gen_name + '/optimizer'):
        gen_optimizer = tf.train.AdamOptimizer(learning_rate=self._hparams.generator_learning_rate)
        gen_variables = tf.trainable_variables(self.name + '/' + self._generator.name)
        gen_train_op = gen_optimizer.minimize(loss=gen_total_loss,
                                              var_list=gen_variables,
                                              global_step=tf.train.get_or_create_global_step())

        self._dual.set_op('gen_train_op', gen_train_op)

      with tf.name_scope(self.disc_name + '/loss'):
        disc_loss = self._discriminator.loss(real_logits, fake_logits)

        self._dual.set_op('disc_loss', disc_loss)

      with tf.variable_scope(self.disc_name + '/optimizer'):
        disc_optimizer = tf.train.AdamOptimizer(learning_rate=self._hparams.discriminator_learning_rate)
        disc_variables = tf.trainable_variables(self.name + '/' + self._discriminator.name)
        disc_train_op = disc_optimizer.minimize(loss=disc_loss,
                                                var_list=disc_variables,
                                                global_step=tf.train.get_or_create_global_step())

        self._dual.set_op('disc_train_op', disc_train_op)

  def get_loss(self):
    return self._loss

  def add_fetches(self, fetches, batch_type=None):
    """Adds ops that will get evaluated."""
    if batch_type.startswith(self._generator.name + '_'):
      real_batch_type = batch_type.split(self._generator.name + '_')[-1]

      names = ['gen_total_loss', 'fake_output']

      if real_batch_type == 'training':
        names.extend(['gen_train_op'])

    elif batch_type.startswith(self._discriminator.name + '_'):
      real_batch_type = batch_type.split(self._discriminator.name + '_')[-1]

      names = ['disc_loss']

      if real_batch_type == 'training':
        names.extend(['disc_train_op'])

    else:
      raise NotImplementedError('Batch type not supported: ' + batch_type)

    self._dual.add_fetches(fetches, names)

    # Summaries
    super().add_fetches(fetches, real_batch_type)

  def set_fetches(self, fetched, batch_type='training'):
    """Store updated tensors"""

    # Loss (not a tensor)
    if batch_type.startswith(self._generator.name + '_'):
      real_batch_type = batch_type.split(self._generator.name + '_')[-1]

      self._loss = fetched[self.name]['gen_total_loss']

      names = ['fake_output']

    elif batch_type.startswith(self._discriminator.name + '_'):
      real_batch_type = batch_type.split(self._discriminator.name + '_')[-1]

      self._loss = fetched[self.name]['disc_loss']

      names = []

    else:
      raise NotImplementedError('Batch type not supported: ' + batch_type)

    self._dual.set_fetches(fetched, names)

    # Summaries
    super().set_fetches(fetched, real_batch_type)

  def write_summaries(self, step, writer, batch_type='training'):
    if batch_type.startswith(self._generator.name + '_'):
      real_batch_type = batch_type.split(self._generator.name + '_')[-1]
    elif batch_type.startswith(self._discriminator.name + '_'):
      real_batch_type = batch_type.split(self._discriminator.name + '_')[-1]

    super().write_summaries(step, writer, real_batch_type)

  def _build_summaries(self, batch_type=None, max_outputs=3):
    """Builds all summaries."""
    del batch_type

    summaries = []

    summary_gen_input_shape = image_utils.get_image_summary_shape(self._gen_input_shape)
    summary_real_input_shape = image_utils.get_image_summary_shape(self._real_input_shape)

    gen_inputs_summary_reshape = tf.reshape(self._dual.get_pl('gen_inputs'), summary_gen_input_shape)
    real_inputs_summary_reshape = tf.reshape(self._dual.get_pl('real_inputs'), summary_real_input_shape)

    summaries.append(tf.summary.image('gen_inputs', gen_inputs_summary_reshape, max_outputs=max_outputs))
    summaries.append(tf.summary.image('real_inputs', real_inputs_summary_reshape, max_outputs=max_outputs))

    # Concatenate input and reconstruction summaries
    fake_output_summary_reshape = tf.reshape(self._dual.get_op('fake_output'), summary_real_input_shape)
    # summary_reconstruction = tf.concat([input_summary_reshape, decoding_summary_reshape], axis=1)
    reconstruction_summary_op = tf.summary.image('reconstruction', fake_output_summary_reshape,
                                                 max_outputs=max_outputs)
    summaries.append(reconstruction_summary_op)

    # Record the discriminator score using REAL inputs
    real_score_summary = tf.summary.scalar('real_score', tf.reduce_mean(self._dual.get_op('real_score')))
    summaries.append(real_score_summary)

    # Record the discriminator score using FAKE inputs
    fake_score_summary = tf.summary.scalar('fake_score', tf.reduce_mean(self._dual.get_op('fake_score')))
    summaries.append(fake_score_summary)

    # Record the Generator losses
    gen_adv_loss_summary = tf.summary.scalar('gen_adv_loss', self._dual.get_op('gen_adv_loss'))
    summaries.append(gen_adv_loss_summary)

    gen_mse_loss_summary = tf.summary.scalar('gen_mse_loss', self._dual.get_op('gen_mse_loss'))
    summaries.append(gen_mse_loss_summary)

    gen_total_loss_summary = tf.summary.scalar('gen_total_loss', self._dual.get_op('gen_total_loss'))
    summaries.append(gen_total_loss_summary)

    # Record the Discriminator loss
    disc_loss_summary = tf.summary.scalar('disc_loss', self._dual.get_op('disc_loss'))
    summaries.append(disc_loss_summary)


    return summaries
