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

"""CompositeVideoWorkflow class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pagi.workflows.composite_workflow import CompositeWorkflow

from rsm.workflows.video_workflow import VideoWorkflow
from rsm.components.sequence_memory_layer import SequenceMemoryLayer
from rsm.components.composite_rsm_stack import CompositeRSMStack

class CompositeVideoWorkflow(CompositeWorkflow, VideoWorkflow):
  """A composite variant of the video workflow."""

  def _do_batch_after_hook(self, global_step, batch_type, fetched, feed_dict, data_subset):
    if CompositeRSMStack.ae_name in self._component.get_sub_components().keys():
      sub_components = self._component.get_sub_component(CompositeRSMStack.ae_name).get_sub_components()

      for i, (name, sub_component) in enumerate(sub_components.items()):
        # Skip the first layer; we already have its reconstruction
        if i == 0:
          continue

        self._decoder(global_step, name, CompositeRSMStack.ae_name, sub_component.get_encoding(), feed_dict)

    if CompositeRSMStack.rsm_name in self._component.get_sub_components().keys():
      super()._do_batch_after_hook(global_step, batch_type, fetched, feed_dict, data_subset)

      if CompositeRSMStack.ae_name in self._component.get_sub_components().keys():
        rsm_output = self.get_decoded_frame()
        self._decoder(global_step, CompositeRSMStack.rsm_name, CompositeRSMStack.ae_name,
                      rsm_output, feed_dict)

  def set_previous_frame(self, previous):
    self._component.get_sub_component('rsm_stack').get_layer(0).get_dual().set_values('previous', previous)

  def get_decoded_frame(self):
    return self._component.get_sub_component('rsm_stack').get_layer(0).get_values(SequenceMemoryLayer.decoding)

  def get_previous_frame(self):
    return self._component.get_sub_component('rsm_stack').get_layer(0).get_values(SequenceMemoryLayer.previous)

  def _get_status(self):
    """Return some string proxy for the losses or errors being optimized"""
    if CompositeRSMStack.rsm_name in self._component.get_sub_components().keys():
      return self._component.get_sub_component('rsm_stack').get_loss()
    return 0.0
