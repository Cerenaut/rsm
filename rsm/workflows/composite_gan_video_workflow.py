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

"""CompositeGANVideoWorkflow class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from rsm.workflows.composite_video_workflow import CompositeVideoWorkflow
from rsm.workflows.composite_gan_workflow import CompositeGANWorkflow

class CompositeGANVideoWorkflow(CompositeGANWorkflow, CompositeVideoWorkflow):
  """A composite variant of the video workflow."""

  @staticmethod
  def default_opts():
    """Builds an HParam object with default workflow options."""
    opts = tf.contrib.training.HParams()
    cgw_opts = CompositeGANWorkflow.default_opts()
    cvw_opts = CompositeVideoWorkflow.default_opts()

    for key, value in cgw_opts.values().items():
      opts.add_hparam(key, value)

    for key, value in cvw_opts.values().items():
      if key not in opts:
        opts.add_hparam(key, value)

    return opts

  def _build_prior_fetches(self):
    return {'inputs': self._inputs, 'states': self._states}

  def training(self, dataset_handle, global_step):  # pylint: disable=arguments-differ
    batch_type, fetched, feed_dict, data_subset = super().training(dataset_handle, global_step)
    self._do_batch_after_hook(global_step, batch_type, fetched, feed_dict, data_subset)

  def testing(self, dataset_handle, global_step):
    batch_type, fetched, feed_dict, data_subset = super().testing(dataset_handle, global_step)
    self._do_batch_after_hook(global_step, batch_type, fetched, feed_dict, data_subset)

  def _do_batch(self, fetches, feed_dict, batch_type, data_subset, global_step):
    """The training procedure within the batch loop"""
    fetches, feed_dict, fetched = super()._do_batch(fetches, feed_dict, batch_type, data_subset, global_step)

    if 'states' in fetched:
      self._states_vals = fetched['states']
    if 'inputs' in fetched:
      self._inputs_vals = fetched['inputs']

    return fetches, feed_dict, fetched

  def get_decoded_frame(self):
    return self._component.get_sub_component('gan').get_output()
