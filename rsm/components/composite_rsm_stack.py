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

import tensorflow as tf

from pagi.utils.hparam_multi import HParamMulti
from pagi.components.composite_component import CompositeComponent

from rsm.components.sequence_memory_stack import SequenceMemoryStack
from rsm.components.sparse_conv_autoencoder_stack import SparseConvAutoencoderStack

class CompositeRSMStack(CompositeComponent):
  """A composite component consisting of a stack of k-Sparse convolutional autoencoders and a stack of RSM layers."""

  ae_name = 'ae_stack'
  rsm_name = 'rsm_stack'

  @staticmethod
  def default_hparams():
    """Builds an HParam object with default hyperparameters."""

    # create component level hparams (this will be a multi hparam, with hparams from sub components)
    batch_size = 64
    hparams = tf.contrib.training.HParams(
        batch_size=batch_size,
        build_rsm=True,
        build_ae=True
    )

    # create all possible sub component hparams
    ae_stack = SparseConvAutoencoderStack.default_hparams()
    rsm_stack = SequenceMemoryStack.default_hparams()

    subcomponents = [ae_stack, rsm_stack]   # all possible subcomponents

    def set_hparam_in_subcomponents(hparam_name, val):
      """Sets the common hparams to sub components."""
      for comp in subcomponents:
        comp.set_hparam(hparam_name, val)

    rsm_stack.set_hparam('sparsity', [25])
    rsm_stack.set_hparam('cols', [200])
    rsm_stack.set_hparam('cells_per_col', [4])

    rsm_stack.set_hparam('freq_min', 0.01)
    rsm_stack.set_hparam('inhibition_decay', [0.1])

    rsm_stack.set_hparam('feedback_norm', [True])
    rsm_stack.set_hparam('feedback_decay_rate', [0.0])
    rsm_stack.set_hparam('feedback_keep_rate', [1.0])

    rsm_stack.set_hparam('lifetime_sparsity_cols', False)
    rsm_stack.set_hparam('lifetime_sparsity_dends', False)

    # default hparams in individual component should be consistent with component level hparams
    set_hparam_in_subcomponents('batch_size', batch_size)

    # add sub components to the composite hparams
    HParamMulti.add(source=ae_stack, multi=hparams, component=CompositeRSMStack.ae_name)
    HParamMulti.add(source=rsm_stack, multi=hparams, component=CompositeRSMStack.rsm_name)

    return hparams

  @property
  def name(self):
    return self._name

  def get_loss(self):
    return self.get_sub_component('output').get_loss()

  def get_dual(self, name=None):  # pylint: disable=arguments-differ
    if name is None:
      return self._dual
    return self.get_sub_component(name).get_dual()

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
    self._name = name
    self._hparams = hparams

    with tf.variable_scope(self._name, reuse=tf.AUTO_REUSE):
      input_values_next = input_values
      input_shape_next = input_shape

      # Build the AE Stack
      if self._hparams.build_ae:
        input_values_next, input_shape_next = self._build_ae_stack(input_values_next, input_shape_next)

      # Build the RSM Stack
      if self._hparams.build_rsm:
        self._build_rsm_stack(input_values_next, input_shape_next, label_values, label_shape, decoder)

        self._layers = self.get_sub_component(self.rsm_name).get_layers()

  def _build_ae_stack(self, input_values, input_shape):
    """Builds a stack of k-Sparse convolutional autoencoders."""
    ae_stack = SparseConvAutoencoderStack()
    ae_stack_hparams = SparseConvAutoencoderStack.default_hparams()

    ae_stack_hparams = HParamMulti.override(multi=self._hparams, target=ae_stack_hparams, component=self.ae_name)
    ae_stack.build(input_values, input_shape, ae_stack_hparams, self.ae_name)
    self._add_sub_component(ae_stack, self.ae_name)

    input_values_next = ae_stack.get_sub_component('output').get_encoding_op()
    input_shape_next = input_values_next.get_shape().as_list()

    return input_values_next, input_shape_next

  def _build_rsm_stack(self, input_values, input_shape, label_values=None, label_shape=None, decoder=None):
    """Builds a stack of RSM layers."""
    rsm_stack = SequenceMemoryStack()
    rsm_stack_hparams = SequenceMemoryStack.default_hparams()

    rsm_stack_hparams = HParamMulti.override(multi=self._hparams, target=rsm_stack_hparams, component=self.rsm_name)
    rsm_stack.build(input_values, input_shape, label_values=label_values, label_shape=label_shape,
                    decoder=decoder, hparams=rsm_stack_hparams, name=self.rsm_name)
    self._add_sub_component(rsm_stack, self.rsm_name)

  def build_secondary_decoding_summaries(self, name, component):
    """Builds secondary decoding summaries."""
    scope = component + '-summaries/'
    self.get_sub_component(component).build_secondary_decoding_summaries(scope, name)

  # --------------------------------------------------------------------------
  # RSM-specific Methods
  # --------------------------------------------------------------------------

  def update_recurrent_and_feedback(self):
    return self.get_sub_component(self.rsm_name).update_recurrent_and_feedback()

  def update_statistics(self, session):
    return self.get_sub_component(self.rsm_name).update_statistics(session)

  def update_history(self, session, history_mask):
    return self.get_sub_component(self.rsm_name).update_history(session, history_mask)
