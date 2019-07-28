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

"""SparseConvAutoencoderStack class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import tensorflow as tf

from pagi.utils import tf_utils
from pagi.utils.dual import DualData
from pagi.components.composite_component import CompositeComponent
from pagi.components.sparse_conv_autoencoder_component import SparseConvAutoencoderComponent

class SparseConvAutoencoderStack(CompositeComponent):
  """A composite component with N layers of k-sparse convolutional autoencoders."""

  @staticmethod
  def default_hparams():
    """Builds an HParam object with default hyperparameters."""
    hparams = tf.contrib.training.HParams()
    component_hparams = SparseConvAutoencoderComponent.default_hparams()
    stack_hparams = ['num_layers', 'batch_size']

    for key, value in component_hparams.values().items():
      if key not in stack_hparams:
        hparams.add_hparam(key, [value])

    hparams.add_hparam('num_layers', 1)
    hparams.add_hparam('batch_size', component_hparams.batch_size)
    hparams.add_hparam('sum_norm', [-1])

    return hparams

  def __init__(self):
    super().__init__()

    self._decoder_summaries = []

  def get_loss(self):
    return self.get_sub_component('output').get_loss()

  def get_output(self):
    return self.get_sub_component('output').get_encoding()

  def build(self, input_values, input_shape, hparams, name='component', encoding_shape=None):
    """Initializes the model parameters.

    Args:
        input_values: Tensor containing input
        input_shape: The shape of the input, for display (internal is vectorized)
        encoding_shape: The shape to be used to display encoded (hidden layer) structures
        hparams: The hyperparameters for the model as tf.contrib.training.HParams.
        name: A globally unique graph name used as a prefix for all tensors and ops.
    """
    self._name = name
    self._hparams = hparams
    self._dual = DualData(self._name)
    self._encoding_shape = encoding_shape

    with tf.variable_scope(self._name, reuse=tf.AUTO_REUSE):
      self._build_layers(input_values, input_shape)

  def _build_layers(self, input_values, input_shape):
    """Builds N layers of k-Sparse convolutional autoencoders."""
    logging.info('Building layers...')

    layers_hparams = []
    layers_shapes = []

    num_layers = self._hparams.num_layers

    layer_input_values = input_values
    layer_input_shape = input_shape

    # Explicitly specify batch size
    if layer_input_shape[0] < 0:
      layer_input_shape[0] = self._hparams.batch_size

    hparams_dict = self._hparams.values()

    # Compute geometry of all layers
    for i in range(num_layers):
      layer_hparams = SparseConvAutoencoderComponent.default_hparams()

      for key, _ in layer_hparams.values().items():
        if key in hparams_dict.keys():
          hparam_value = hparams_dict[key]
          if isinstance(hparams_dict[key], list):
            try:
              hparam_value = hparams_dict[key][i]
            except IndexError:
              hparam_value = hparams_dict[key][0]

          layer_hparams.set_hparam(key, hparam_value)

      layer_shape = SparseConvAutoencoderComponent.get_encoding_shape_4d(layer_input_shape, layer_hparams)

      layers_hparams.append(layer_hparams)
      layers_shapes.append(layer_shape)

      layer_input_shape = layer_shape  # for next layer

    # 2nd pass - for bi-directional connectivity
    layer_input_values = input_values
    layer_input_shape = input_shape

    for i in range(num_layers):
      layer_hparams = layers_hparams[i]  # retrieve precalculated hparams

      layer = SparseConvAutoencoderComponent()
      layer_name = self._name + '/layer-' + str(i + 1)

      layer.build(layer_input_values, layer_input_shape, layer_hparams, name=layer_name, encoding_shape=None)

      self._add_sub_component(layer, layer_name)

      # link layers
      # This means it'll update with the latest state of input in lower layer WRT current input
      output_encoding = layer.get_encoding_op() # 4d, protected with StopGradient
      layer_input_values = output_encoding

      if self._hparams.sum_norm[i] != -1 and self._hparams.sum_norm[i] > 0:
        layer_input_values = tf_utils.tf_normalize_to_k(layer_input_values, k=self._hparams.sum_norm[i], axis=[1, 2, 3])

      layer_input_shape = layer_input_values.shape.as_list()

  def build_secondary_decoding_summaries(self, scope, name):
    """Builds secondary decoding summaries."""
    for decoder_name, comp in self._sub_components.items():
      scope = decoder_name + '-summaries/'
      if name != self.name:
        summary_name = name + '_' + decoder_name
        if summary_name not in self._decoder_summaries:
          comp.build_secondary_decoding_summaries(scope, summary_name)
          self._decoder_summaries.append(summary_name)
      else:
        for decoded_name in self._sub_components:
          summary_name = decoded_name + '_' + decoder_name
          if summary_name not in self._decoder_summaries:
            comp.build_secondary_decoding_summaries(scope, summary_name)
            self._decoder_summaries.append(summary_name)
