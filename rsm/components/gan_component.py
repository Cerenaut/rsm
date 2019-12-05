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

import logging

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
        batch_size=128,
        conditional=True,

        # Generator-specific options
        generator_num_layers=2,
        generator_filters=[32, 64],
        generator_filters_field_width=[6, 6],
        generator_filters_field_height=[6, 6],
        generator_filters_field_stride=[3, 3],
        generator_nonlinearity=['leaky_relu', 'leaky_relu'],
        generator_loss_mse_lambda=1.0,
        generator_loss_adv_lambda=0.9,
        generator_learning_rate=0.0005,
        generator_input_size=[28, 28, 1],
        generator_input_nonlinearity='relu',
        generator_output_nonlinearity='tanh',
        generator_autoencoder='encode',

        # Discriminator-specific options
        discriminator_num_layers=2,
        discriminator_filters=[32, 64],
        discriminator_filters_field_width=[6, 6],
        discriminator_filters_field_height=[6, 6],
        discriminator_filters_field_stride=[3, 3],
        discriminator_nonlinearity=['leaky_relu', 'leaky_relu'],
        discriminator_learning_rate=0.0005,
        discriminator_input_size=[28, 28, 1],
        discriminator_input_nonlinearity='relu',
        discriminator_output_nonlinearity='sigmoid',
        discriminator_input_noise=False
    )

  class BaseNetwork:
    """A base class with common methods for generator and discriminator."""
    def __init__(self, hparams, output_size, name):
      self.name = name
      self.hparams = hparams
      self.output_size = output_size

      self.layers = self.build_layers()

    def build_layers(self):
      """Build the network."""
      with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE, auxiliary_name_scope=False):
        layers = []

        initializer = tf.random_normal_initializer(mean=0., stddev=0.02)

        def conv_block(i, fn=tf.layers.Conv2D, nonlinearity='leaky_relu'):
          return fn(
              filters=self.hparams.filters[i],
              kernel_size=[
                  self.hparams.filters_field_height[i],
                  self.hparams.filters_field_width[i]
              ],
              padding='same',
              strides=self.hparams.filters_field_stride[i],
              activation=type_activation_fn(nonlinearity),
              kernel_initializer=initializer
          )

        if self.hparams.autoencoder == 'decode':
          layer_fn = tf.layers.Conv2DTranspose
        else:
          layer_fn = tf.layers.Conv2D

        for i in range(self.hparams.num_layers):
          layer = conv_block(i, layer_fn, nonlinearity=self.hparams.nonlinearity[i])
          layers.append(layer)

        if self.hparams.autoencoder == 'both':
          logging.info('Using autoencoder mode in %s.', self.name)

          # Decoder
          for i in range(self.hparams.num_layers - 1, -1, -1):
            if i == (self.hparams.num_layers - 1):
              continue
            layer = conv_block(i, fn=tf.layers.Conv2DTranspose, nonlinearity=self.hparams.nonlinearity[i])
            layers.append(layer)

        # Build output layer
        if self.hparams.autoencoder in ['decode', 'both']:
          output_layer = tf.layers.Conv2DTranspose(
              filters=1,
              kernel_size=[
                  self.hparams.filters_field_height[-1],
                  self.hparams.filters_field_width[-1]
              ],
              padding='same',
              strides=self.hparams.filters_field_stride[-1],
              activation=None,
              kernel_initializer=initializer
          )
        else:
          flatten_layer = tf.layers.Flatten()
          layers.append(flatten_layer)

          output_layer = tf.layers.Dense(self.output_size, activation=None, kernel_initializer=initializer)

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
          print(outputs)
        output_nonlinearity = type_activation_fn(self.hparams.output_nonlinearity)
        return outputs, output_nonlinearity(outputs)

    def trainable_variables(self):
      return tf.trainable_variables(self.name)

  class Generator(BaseNetwork):
    """Builds the generator network."""
    def __init__(self, hparams, output_size, name='generator'):
      generator_hparams = tf.contrib.training.HParams(
          num_layers=hparams.generator_num_layers,
          filters=hparams.generator_filters,
          filters_field_height=hparams.generator_filters_field_height,
          filters_field_width=hparams.generator_filters_field_width,
          filters_field_stride=hparams.generator_filters_field_stride,
          nonlinearity=hparams.generator_nonlinearity,
          input_size=hparams.generator_input_size,
          input_nonlinearity=hparams.generator_input_nonlinearity,
          output_nonlinearity=hparams.generator_output_nonlinearity,
          conditional=hparams.conditional,
          autoencoder=hparams.generator_autoencoder
      )

      super(GANComponent.Generator, self).__init__(
          generator_hparams, output_size, name)

    def loss(self, fake_logits):
      return tf.reduce_mean(
          tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake_logits),
                                                  logits=fake_logits,
                                                  name='gen_adv_loss')
      )

  class Discriminator(BaseNetwork):
    """Builds the discriminator network."""
    def __init__(self, hparams, output_size=1, name='discriminator'):
      discriminator_hparams = tf.contrib.training.HParams(
          num_layers=hparams.discriminator_num_layers,
          filters=hparams.discriminator_filters,
          filters_field_height=hparams.discriminator_filters_field_height,
          filters_field_width=hparams.discriminator_filters_field_width,
          filters_field_stride=hparams.discriminator_filters_field_stride,
          nonlinearity=hparams.discriminator_nonlinearity,
          input_size=hparams.discriminator_input_size,
          input_nonlinearity=hparams.discriminator_input_nonlinearity,
          output_nonlinearity=hparams.discriminator_output_nonlinearity,
          conditional=hparams.conditional,
          autoencoder=False
      )

      super(GANComponent.Discriminator, self).__init__(
          discriminator_hparams, output_size, name)

    def __call__(self, inputs):
      """
      Args:
        inputs: A tensor with shape = (b, h, w, c)
      Returns:
        It return the logits and the prediction with nonlinearity.
      """
      with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE, auxiliary_name_scope=False):
        gen_inputs, cond_inputs = inputs

        outputs = gen_inputs
        for layer in self.layers[:-2]:
          outputs = layer(outputs)

        if self.hparams.conditional:
          # Use the same size and nonlinearity as previous layer
          _, h, w, c = outputs.shape
          cond_nonlinearity = type_activation_fn(self.hparams.nonlinearity[-1])
          cond_layer = tf.layers.Dense(units=h * w * c, activation=cond_nonlinearity)

          # Process the conditional input, reshape to same size as previous layer
          cond_inputs_2d = tf.layers.flatten(cond_inputs)
          cond_outputs_2d = cond_layer(cond_inputs_2d)
          cond_outputs_4d = tf.reshape(cond_outputs_2d, [-1, h, w, c])

          concat_outputs = tf.concat([outputs, cond_outputs_4d], axis=3)
          outputs = tf.layers.flatten(concat_outputs)

          outputs = self.layers[-1](concat_outputs)
          output_nonlinearity = type_activation_fn(self.hparams.output_nonlinearity)

        return outputs, output_nonlinearity(outputs)

    def loss(self, real_logits, fake_logits):
      """Build the discriminator loss."""
      smoothing_factor = 0.1

      real_labels = tf.ones_like(real_logits) * (1 - smoothing_factor)
      fake_labels = tf.zeros_like(fake_logits)

      real_loss = tf.reduce_mean(
          tf.nn.sigmoid_cross_entropy_with_logits(labels=real_labels,
                                                  logits=real_logits,
                                                  name='real_loss')
      )

      fake_loss = tf.reduce_mean(
          tf.nn.sigmoid_cross_entropy_with_logits(labels=fake_labels,
                                                  logits=fake_logits,
                                                  name='fake_loss')
      )

      return real_loss, fake_loss

  def build(self, gen_input_shape, real_input_shape, condition_shape, hparams, name='gan', encoding_shape=None):
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
    self._condition_shape = condition_shape
    self._encoding_shape = encoding_shape
    self._loss = 0.0

    with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
      scope = (tf.no_op(name='.').name[:-1])

      with tf.name_scope('inputs'):
        gen_inputs_4d = self._dual.add('gen_inputs', shape=gen_input_shape, default_value=0.0).add_pl()
        real_inputs_4d = self._dual.add('real_inputs', shape=real_input_shape, default_value=0.0).add_pl()
        noise_param = self._dual.add('noise_param', shape=[], default_value=0.0).add_pl()

      with tf.name_scope(self.gen_name):
        gen_global_step = tf.Variable(0, trainable=False, name='global_step')

        self._generator = self.Generator(
            hparams=self._hparams,
            output_size=np.prod(real_input_shape[1:]),
            name=self.gen_name)

        gen_input = gen_inputs_4d

        print('gen_input', gen_input)

        _, fake_output = self._generator(gen_input)
        fake_output_4d = tf.reshape(fake_output, real_input_shape)

        self._dual.set_op('fake_output', fake_output_4d)

      with tf.name_scope(self.disc_name):
        disc_global_step = tf.Variable(0, trainable=False, name='global_step')

        self._discriminator = self.Discriminator(
            hparams=self._hparams,
            output_size=1,
            name=self.disc_name)

        disc_real_input = real_inputs_4d
        disc_fake_input = fake_output_4d

        if self._hparams.discriminator_input_noise:
          disc_input_noise = tf.random_normal(shape=tf.shape(disc_real_input), mean=0.0, stddev=noise_param,
                                              dtype=tf.float32)

          disc_real_input = disc_real_input + disc_input_noise
          disc_fake_input = disc_fake_input + disc_input_noise

        print('disc_real_input', disc_real_input)
        print('disc_fake_input', disc_fake_input)

        real_logits, real_score = self._discriminator(inputs=(disc_real_input, gen_inputs_4d))
        fake_logits, fake_score = self._discriminator(inputs=(disc_fake_input, gen_inputs_4d))

        self._dual.set_op('real_score', real_score)
        self._dual.set_op('fake_score', fake_score)

      with tf.name_scope(self.gen_name + '/loss'):
        gen_adv_loss = self._generator.loss(fake_logits)
        gen_mse_loss = tf.losses.mean_squared_error(real_inputs_4d, fake_output_4d)
        gen_total_loss = (self._hparams.generator_loss_mse_lambda * gen_mse_loss) + \
                         (self._hparams.generator_loss_adv_lambda * gen_adv_loss)

        self._dual.set_op('gen_adv_loss', gen_adv_loss)
        self._dual.set_op('gen_mse_loss', gen_mse_loss)
        self._dual.set_op('gen_total_loss', gen_total_loss)

      with tf.variable_scope(self.gen_name + '/optimizer'):
        gen_optimizer = tf.train.AdamOptimizer(learning_rate=self._hparams.generator_learning_rate)
        gen_variables = tf.trainable_variables(scope + self._generator.name)
        gen_train_op = gen_optimizer.minimize(loss=gen_total_loss,
                                              var_list=gen_variables,
                                              global_step=gen_global_step)

        self._dual.set_op('gen_train_op', gen_train_op)

      with tf.name_scope(self.disc_name + '/loss'):
        disc_real_loss, disc_fake_loss = self._discriminator.loss(real_logits, fake_logits)
        disc_total_loss = disc_real_loss + disc_fake_loss

        self._dual.set_op('disc_real_loss', disc_real_loss)
        self._dual.set_op('disc_fake_loss', disc_fake_loss)
        self._dual.set_op('disc_total_loss', disc_total_loss)

      with tf.variable_scope(self.disc_name + '/optimizer'):
        disc_optimizer = tf.train.GradientDescentOptimizer(learning_rate=self._hparams.discriminator_learning_rate)
        disc_variables = tf.trainable_variables(scope + self._discriminator.name)
        disc_train_op = disc_optimizer.minimize(loss=disc_total_loss,
                                                var_list=disc_variables,
                                                global_step=disc_global_step)

        self._dual.set_op('disc_train_op', disc_train_op)

  def get_loss(self):
    return self._loss

  def get_output(self):
    return self.get_values('fake_output')

  def get_output_op(self):
    return self.get_op('fake_output')

  def add_fetches(self, fetches, batch_type=None):
    """Adds ops that will get evaluated."""

    names = ['fake_output']

    if batch_type.startswith(self._generator.name + '_'):
      real_batch_type = batch_type.split(self._generator.name + '_')[-1]

      names.extend(['gen_total_loss'])

      if real_batch_type == 'training':
        names.extend(['gen_train_op'])

    elif batch_type.startswith(self._discriminator.name + '_'):
      real_batch_type = batch_type.split(self._discriminator.name + '_')[-1]

      names.extend(['disc_total_loss'])

      if real_batch_type == 'training':
        names.extend(['disc_train_op'])

    else:
      raise NotImplementedError('Batch type not supported: ' + batch_type)

    self._dual.add_fetches(fetches, names)

    # Summaries
    super().add_fetches(fetches, real_batch_type)

  def set_fetches(self, fetched, batch_type='training'):
    """Store updated tensors"""

    names = ['fake_output']

    # Loss (not a tensor)
    if batch_type.startswith(self._generator.name + '_'):
      real_batch_type = batch_type.split(self._generator.name + '_')[-1]

      self._loss = fetched[self.name]['gen_total_loss']

    elif batch_type.startswith(self._discriminator.name + '_'):
      real_batch_type = batch_type.split(self._discriminator.name + '_')[-1]

      self._loss = fetched[self.name]['disc_total_loss']

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
    reconstruction_summary_op = tf.summary.image('reconstruction', fake_output_summary_reshape,
                                                 max_outputs=max_outputs)
    summaries.append(reconstruction_summary_op)

    # Record the discriminator score using REAL inputs
    real_score_summary = tf.summary.scalar('score_real', tf.reduce_mean(self._dual.get_op('real_score')))
    summaries.append(real_score_summary)

    # Record the discriminator score using FAKE inputs
    fake_score_summary = tf.summary.scalar('score_fake', tf.reduce_mean(self._dual.get_op('fake_score')))
    summaries.append(fake_score_summary)

    # Record the Generator losses
    gen_adv_loss_summary = tf.summary.scalar('gen_adv_loss', self._dual.get_op('gen_adv_loss'))
    summaries.append(gen_adv_loss_summary)

    gen_mse_loss_summary = tf.summary.scalar('gen_mse_loss', self._dual.get_op('gen_mse_loss'))
    summaries.append(gen_mse_loss_summary)

    gen_total_loss_summary = tf.summary.scalar('gen_total_loss', self._dual.get_op('gen_total_loss'))
    summaries.append(gen_total_loss_summary)

    # Record the Discriminator loss
    disc_real_loss_summary = tf.summary.scalar('disc_real_loss', self._dual.get_op('disc_real_loss'))
    summaries.append(disc_real_loss_summary)

    disc_fake_loss_summary = tf.summary.scalar('disc_fake_loss', self._dual.get_op('disc_fake_loss'))
    summaries.append(disc_fake_loss_summary)

    disc_total_loss_summary = tf.summary.scalar('disc_total_loss', self._dual.get_op('disc_total_loss'))
    summaries.append(disc_total_loss_summary)


    return summaries
