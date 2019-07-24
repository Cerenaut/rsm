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

"""MovingMNISTWorkflow class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from rsm.workflows.video_workflow import VideoWorkflow
from rsm.components.sequence_memory_layer import SequenceMemoryLayer

class MovingMNISTWorkflow(VideoWorkflow):
  """Workflow for dealing with Moving MNIST dataset."""

  def _sigmoid(self, x):
    return 1. / (1. + np.exp(-x))

  def _cross_entropy_loss(self, z, y, eps=1e-9):
    loss = z * np.log(y + eps) + (1 - z) * np.log((1 - y) + eps)
    loss = -np.sum(loss) / self._hparams.batch_size
    return loss

  def _do_batch_after_hook(self, global_step, batch_type, fetched, feed_dict):
    super()._do_batch_after_hook(global_step, batch_type, fetched, feed_dict)

    inputs = fetched['inputs']

    # Get decoding to calculate the loss
    decoding = self._component.get_layer(0).get_values(SequenceMemoryLayer.decoding)
    decoding = np.clip(decoding, 0.0, 1.0)

    # Compute cross entropy loss
    bce_loss = self._cross_entropy_loss(inputs, decoding)

    # Write summaries
    summary = tf.Summary()
    summary.value.add(tag=self._component.name + '/summaries/' + batch_type + '/bce_loss',
                      simple_value=bce_loss)
    self._writer.add_summary(summary, global_step)
    self._writer.flush()
