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

"""CompositeRSMStack class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import tensorflow as tf

from pagi.utils.hparam_multi import HParamMulti
from pagi.components.composite_component import CompositeComponent
from pagi.components.visual_cortex_component import VisualCortexComponent

from rsm.components.gan_component import GANComponent
from rsm.components.sequence_memory_layer import SequenceMemoryLayer
from rsm.components.sequence_memory_stack import SequenceMemoryStack
from rsm.components.sparse_conv_autoencoder_stack import SparseConvAutoencoderStack


class CompositeRSMStack(CompositeComponent):
  """A composite component consisting of:
    * A reducer: a stack of k-Sparse convolutional autoencoders and
    * A predictor: a stack of RSM layers and
    * A sampler: A GAN. """

  key_reducer = 'reducer'
  key_predictor = 'predictor'
  key_sampler = 'sampler'

  # TODO change these to their functional roles (reducer, predictor, sampler)
  reducer_name = 'ae_stack'
  predictor_name = 'rsm_stack'
  sampler_name = 'gan'

  @staticmethod
  def default_hparams():
    """Builds an HParam object with default hyperparameters."""

    # create component level hparams (this will be a multi hparam, with hparams from sub components)
    batch_size = 64
    hparams = tf.contrib.training.HParams(
        batch_size=batch_size,
        build_ae=False,
        build_rsm=True,
        build_gan=False,

        gan_rsm_input='decoding'
    )

    # create all possible sub component hparams
    # ae_stack = SparseConvAutoencoderStack.default_hparams()
    ae_stack = VisualCortexComponent.default_hparams()
    rsm_stack = SequenceMemoryStack.default_hparams()
    gan_stack = GANComponent.default_hparams()

    subcomponents = [ae_stack, rsm_stack, gan_stack]   # all possible subcomponents

    # default hparams in individual component should be consistent with component level hparams
    HParamMulti.set_hparam_in_subcomponents(subcomponents, 'batch_size', batch_size)

    # add sub components to the composite hparams
    HParamMulti.add(source=ae_stack, multi=hparams, component=CompositeRSMStack.reducer_name)
    HParamMulti.add(source=rsm_stack, multi=hparams, component=CompositeRSMStack.predictor_name)
    HParamMulti.add(source=gan_stack, multi=hparams, component=CompositeRSMStack.sampler_name)

    return hparams

  def get_loss(self):
    return self.get_sub_component('output').get_loss()

  def build(self, input_values, input_shape, label_values, label_shape, hparams, decoder=None,
            name='composite-rsm-stack'):
    """Initializes the model parameters.

    Args:
        input_values: Tensor containing input
        input_shape: The shape of the input, for display (internal is vectorized)
        encoding_shape: The shape to be used to display encoded (hidden layer) structures
        hparams: The hyperparameters for the model as tf.contrib.training.HParams.
        name: A globally unique graph name used as a prefix for all tensors and ops.
    """
    self.name = name
    self._hparams = hparams
    self._input_values = input_values

    with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
      input_values_next = input_values
      input_shape_next = input_shape

      # Build the AE Stack
      if self._hparams.build_ae:
        logging.info('Building AE (dim. reduction)')
        input_values_next, input_shape_next = self._build_ae_stack(input_values_next, input_shape_next)
      else:
        logging.info('NOT building AE (dim. reduction)')

      # Build the RSM Stack
      if self._hparams.build_rsm:
        logging.info('Building RSM (predictor)')
        input_values_next, input_shape_next = self._build_rsm_stack(input_values_next, input_shape_next,
                                                                    label_values, label_shape, decoder)
      else:
        logging.info('NOT building RSM (predictor)')

      # Build the GAN Component
      if self._hparams.build_gan:
        logging.info('Building GAN (sampler)')
        real_input_shape = input_values.get_shape().as_list()
        gen_input_shape = input_shape_next
        condition_shape = input_shape_next

        self._build_gan(gen_input_shape, real_input_shape, condition_shape)
      else:
        logging.info('NOT building GAN (sampler)')

  def _build_ae_stack(self, input_values, input_shape):
    """Builds a stack of k-Sparse convolutional autoencoders."""
    ae_stack = SparseConvAutoencoderStack()
    ae_stack_hparams = SparseConvAutoencoderStack.default_hparams()

    ae_stack = VisualCortexComponent()
    ae_stack_hparams = VisualCortexComponent.default_hparams()

    ae_stack_hparams = HParamMulti.override(multi=self._hparams, target=ae_stack_hparams, component=self.reducer_name)
    ae_stack.build(input_values, input_shape, ae_stack_hparams, self.reducer_name)
    self._add_sub_component(ae_stack, self.reducer_name)

    input_values_next = ae_stack.get_sub_component('output').get_encoding_op()
    input_shape_next = input_values_next.get_shape().as_list()

    return input_values_next, input_shape_next

  def _build_rsm_stack(self, input_values, input_shape, label_values=None, label_shape=None, decoder=None):
    """Builds a stack of RSM layers."""
    rsm_stack = SequenceMemoryStack()
    rsm_stack_hparams = SequenceMemoryStack.default_hparams()

    rsm_stack_hparams = HParamMulti.override(multi=self._hparams, target=rsm_stack_hparams, component=self.predictor_name)
    rsm_stack.build(input_values, input_shape, label_values=label_values, label_shape=label_shape,
                    decoder=decoder, hparams=rsm_stack_hparams, name=self.predictor_name)
    self._add_sub_component(rsm_stack, self.predictor_name)

    self._layers = self.get_sub_component(self.predictor_name).get_layers()

    if self._hparams.gan_rsm_input == 'decoding':
      input_values_next = self._layers[0].get_op(SequenceMemoryLayer.decoding)
    else:
      input_values_next = self._layers[0].get_op(SequenceMemoryLayer.encoding)

    input_shape_next = input_values_next.get_shape().as_list()

    return input_values_next, input_shape_next

  def _build_gan(self, gen_input_shape, real_input_shape, condition_shape):
    """Build a GAN."""
    gan = GANComponent()
    gan_hparams = GANComponent.default_hparams()

    gan_hparams = HParamMulti.override(multi=self._hparams, target=gan_hparams, component=self.sampler_name)
    gan.build(gen_input_shape, real_input_shape, condition_shape, gan_hparams)
    self._add_sub_component(gan, self.sampler_name)

    input_values_next = gan.get_output_op()
    input_shape_next = input_values_next.get_shape().as_list()

    return input_values_next, input_shape_next

  def get_gan_inputs(self):  # TODO rename to get_sampler_input
    """Finds the input data (off-graph) used as input for the GAN"""
    if self._hparams.build_rsm:  # If the RSM exists, use that:
      if self._hparams.gan_rsm_input == 'decoding':
        return self.get_sub_component(CompositeRSMStack.predictor_name).get_layer(0).get_values(SequenceMemoryLayer.decoding)
      return self.get_sub_component(CompositeRSMStack.predictor_name).get_layer(0).get_values(SequenceMemoryLayer.encoding)

    # Else, RSM doesn't exist, look for AE:
    if self._hparams.build_ae:
      return self.get_sub_component(CompositeRSMStack.reducer_name).get_sub_component('output').get_encoding()

    # Else, the GAN takes input directly from the input.
    return self._input_values

  def get_composite_input_next(self):
    """Return the input that would be processed next. This is only meaningful when using RSM predictor, which works 1-step behind the Dataset source"""
    if self._hparams.build_rsm:
      stack = self.get_sub_component(CompositeRSMStack.predictor_name)
      layer = stack.get_layer(0)
      previous = layer.get_dual().get_values(SequenceMemoryLayer.previous)
      return previous
    assert(False)  # This option isn't meaningful unless we're working 1-step behind the Dataset

  def set_composite_input_next(self, input_next):
    """Overwrite the input that would be processed next. This is only meaningful when using RSM predictor, which works 1-step behind the Dataset source"""
    if self._hparams.build_rsm:
      stack = self.get_sub_component(CompositeRSMStack.predictor_name)
      layer = stack.get_layer(0)
      layer.get_dual().set_values(SequenceMemoryLayer.previous, input_next)
      return
    assert(False)  # This option isn't meaningful unless we're working 1-step behind the Dataset

  def get_composite_output(self):
    output = None
    sample = None
    prediction = None
    if self._hparams.build_gan:
      sample = self.get_sub_component(CompositeRSMStack.sampler_name).get_output()
      output = sample
    if self._hparams.build_rsm:
      stack = self.get_sub_component(CompositeRSMStack.predictor_name)
      layer = stack.get_layer(0)
      prediction = layer.get_values(SequenceMemoryLayer.decoding)
      if output is None:  # no GAN
        output = prediction
    return output, prediction, sample

  def get_reducer_component(self):
    return self.get_sub_component(CompositeRSMStack.reducer_name)

  def get_predictor_component(self):
    return self.get_sub_component(CompositeRSMStack.predictor_name)

  def get_sampler_component(self):
    return self.get_sub_component(CompositeRSMStack.sampler_name)

  def get_losses(self):
    """Return some string proxy for the losses or errors being optimized"""
    losses = {}

    reducer = self.get_sub_component(CompositeRSMStack.reducer_name)
    if reducer is not None:
      reducer_losses = reducer.get_loss()
      losses[CompositeRSMStack.key_reducer] = reducer_losses

    predictor = self.get_sub_component(CompositeRSMStack.predictor_name)
    if predictor is not None:
      predictor_losses = predictor.get_loss()
      losses[CompositeRSMStack.key_predictor] = predictor_losses

    sampler = self.get_sub_component(CompositeRSMStack.sampler_name)
    if sampler is not None:
      sampler_losses = sampler.get_losses()
      losses[CompositeRSMStack.key_sampler] = sampler_losses

    return losses

  # --------------------------------------------------------------------------
  # RSM Stack Methods
  # --------------------------------------------------------------------------
  def update_recurrent_and_feedback(self):
    return self.get_sub_component(self.predictor_name).update_recurrent_and_feedback()

  def update_statistics(self, batch_type, session):
    return self.get_sub_component(self.predictor_name).update_statistics(batch_type, session)

  def update_history(self, session, history_mask, clear_previous):
    return self.get_sub_component(self.predictor_name).update_history(session, history_mask, clear_previous)

  # --------------------------------------------------------------------------
  # Composite Component Methods
  # --------------------------------------------------------------------------

  def filter_sub_components(self, batch_type):
    """Filter the subcomponents to use by the batch_type provided. By default, for simple batch types, won't update the GAN."""
    real_batch_type = batch_type
    sub_components = self._sub_components.copy()

    # If there is a GAN, then don't update that subcomponent
    if self._hparams.build_gan:
      sub_components.pop(CompositeRSMStack.sampler_name)

    # If it's a composite batch type, re-add those components to the batch
    if '-' in batch_type:
      parsed_batch_type = batch_type.split('-')
      component_name = parsed_batch_type[0]
      real_batch_type = parsed_batch_type[-1]

      sub_components = {k: v for (k, v) in self._sub_components.items() if k.startswith(component_name)}

    return real_batch_type, sub_components

  def update_feed_dict(self, feed_dict, batch_type='training'):
    filtered_batch_type, filtered_sub_components = self.filter_sub_components(batch_type)

    for name, comp in filtered_sub_components.items():
      comp.update_feed_dict(feed_dict, self._select_batch_type(filtered_batch_type, name))

  def add_fetches(self, fetches, batch_type='training'):
    filtered_batch_type, filtered_sub_components = self.filter_sub_components(batch_type)

    for name, comp in filtered_sub_components.items():
      comp.add_fetches(fetches, self._select_batch_type(filtered_batch_type, name))

  def set_fetches(self, fetched, batch_type='training'):
    filtered_batch_type, filtered_sub_components = self.filter_sub_components(batch_type)

    for name, comp in filtered_sub_components.items():
      comp.set_fetches(fetched, self._select_batch_type(filtered_batch_type, name))

  def write_summaries(self, step, writer, batch_type='training'):
    filtered_batch_type, filtered_sub_components = self.filter_sub_components(batch_type)

    for name, comp in filtered_sub_components.items():
      comp.write_summaries(step, writer, self._select_batch_type(filtered_batch_type, name))
