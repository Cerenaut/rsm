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

"""CompositeWorkflow class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from pagi.workflows.workflow import Workflow


class GANWorkflow(Workflow):
  """A simple workflow for GANs."""

  def _setup_component(self):
    """Setup the component"""

    # Create the encoder component
    # -------------------------------------------------------------------------
    self._component = self._component_type()

    self.real_input_shape = [self._hparams.batch_size] + self._dataset.shape[1:]
    # self.gen_input_shape = self.real_input_shape
    self.gen_input_shape = [self._hparams.batch_size, 1, 1, 100]
    self.condition_shape = [self._hparams.batch_size, self._dataset.num_classes]

    self._component.build(self.gen_input_shape, self.real_input_shape, self.condition_shape, self._hparams)

    if self._summarize:
      batch_types = ['training', 'encoding']
      if self._freeze_training:
        batch_types.remove('training')
      self._component.build_summaries(batch_types)  # Ask the component to unpack for you


  def training_step(self, dataset_handle, global_step, phase_change=False):  # pylint: disable=arguments-differ
    """The training procedure within the batch loop"""
    del phase_change

    labels_onehot_op = tf.one_hot(self._labels, depth=self._dataset.num_classes)

    inputs, labels_onehot = self._session.run([self._inputs, labels_onehot_op], feed_dict={
        self._placeholders['dataset_handle']: dataset_handle
    })

    self._gen_inputs = np.random.normal(size=self.gen_input_shape).astype(np.float32)
    # self._gen_inputs = self._real_inputs
    self._real_inputs = inputs
    self._condition = labels_onehot

    data_subset = 'train'

    batch_type = 'discriminator_training'
    self._do_batch(dataset_handle, batch_type, data_subset, global_step)

    batch_type = 'generator_training'
    self._do_batch(dataset_handle, batch_type, data_subset, global_step)

  def _do_batch(self, dataset_handle, batch_type, data_subset, global_step):
    """The training procedure within the batch loop"""
    del data_subset, dataset_handle

    feed_dict = {
        self._component.get_dual().get_pl('gen_inputs'): self._gen_inputs,
        self._component.get_dual().get_pl('real_inputs'): self._real_inputs,
        self._component.get_dual().get_pl('condition'): self._condition
    }

    # self._component.update_feed_dict(feed_dict, batch_type)
    fetches = {}
    self._component.add_fetches(fetches, batch_type)

    fetched = self.session_run(fetches, feed_dict=feed_dict)
    self._component.set_fetches(fetched, batch_type)
    self._component.write_summaries(global_step, self._writer, batch_type=batch_type)

    return feed_dict
