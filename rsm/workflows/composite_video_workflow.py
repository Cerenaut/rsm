# Copyright (C) 2018 Project AGI
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

"""VideoWorkflow class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
from PIL import Image

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from pagi.utils import tf_utils
from pagi.workflows.composite_workflow import CompositeWorkflow

from rsm.workflows.video_workflow import VideoWorkflow
from rsm.components.sequence_memory_stack import SequenceMemoryStack
from rsm.components.sequence_memory_layer import SequenceMemoryLayer
from rsm.components.composite_rsm_stack import CompositeRSMStack

class CompositeVideoWorkflow(CompositeWorkflow, VideoWorkflow):

  def _init_test_decodes(self):
    if CompositeRSMStack.ae_name in self._component.get_sub_components().keys():
      self._add_composite_decodes('ae_stack', 'ae_stack')

    if CompositeRSMStack.rsm_name in self._component.get_sub_components().keys():
      self._add_composite_decodes('rsm_stack', 'ae_stack')

    self._num_repeats = len(self.TEST_DECODES)

    return self.TEST_DECODES

  def _do_batch_after_hook(self, global_step, batch_type, fetched, feed_dict):
    if CompositeRSMStack.ae_name in self._component.get_sub_components().keys():
      # Get the decoding from the final layer in the AE Stack
      ae_output = self._component.get_sub_component(CompositeRSMStack.ae_name).get_output()

      self._decoder(global_step, CompositeRSMStack.ae_name, CompositeRSMStack.ae_name, ae_output, feed_dict)

    if CompositeRSMStack.rsm_name in self._component.get_sub_components().keys():
      super()._do_batch_after_hook(global_step, batch_type, fetched, feed_dict)

      rsm_output = self.get_decoded_frame()
      self._decoder(global_step, CompositeRSMStack.rsm_name, CompositeRSMStack.ae_name,
                    rsm_output, feed_dict)

  def set_previous_frame(self, previous):
    self._component.get_sub_component('output').get_layer(0)._dual.set_values('previous', previous)

  def get_decoded_frame(self):
    return self._component.get_sub_component('output').get_layer(0).get_values(SequenceMemoryLayer.decoding)

  def get_previous_frame(self):
    return self._component.get_sub_component('output').get_layer(0).get_values(SequenceMemoryLayer.previous)

  def _get_status(self):
    """Return some string proxy for the losses or errors being optimized"""
    return self._component.get_sub_component('output').get_loss()
