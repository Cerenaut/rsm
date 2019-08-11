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

import logging

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

    self.gen_input_shape = [self._hparams.batch_size, 1, 1, 100]
    self.real_input_shape = [self._hparams.batch_size] + self._dataset.shape[1:]

    self._component.build(self.gen_input_shape, self.real_input_shape, self._hparams)

    if self._summarize:
      batch_types = ['training', 'encoding']
      if self._freeze_training:
        batch_types.remove('training')
      self._component.build_summaries(batch_types)  # Ask the component to unpack for you


  def training(self, dataset_handle, global_step):  # pylint: disable=arguments-differ
    """The training procedure within the batch loop"""

    self._real_inputs = self._session.run(self._inputs, feed_dict={
        self._placeholders['dataset_handle']: dataset_handle
    })

    self._gen_inputs = np.random.normal(size=self.gen_input_shape).astype(np.float32)

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
        self._component.get_dual().get_pl('real_inputs'): self._real_inputs
    }

    # self._component.update_feed_dict(feed_dict, batch_type)
    fetches = {}
    self._component.add_fetches(fetches, batch_type)

    fetched = self.session_run(fetches, feed_dict=feed_dict)
    self._component.set_fetches(fetched, batch_type)
    self._component.write_summaries(global_step, self._writer, batch_type=batch_type)

    return feed_dict

  def run(self, num_batches, evaluate, train=True):
    """Run Experiment"""

    self._setup_profiler()

    if train:
      training_handle = self._session.run(self._dataset_iterators['training'].string_handle())
      self._session.run(self._dataset_iterators['training'].initializer)

      self._on_before_training_batches()

      for batch in range(self._last_step, num_batches):
        training_step = self._session.run(tf.train.get_global_step(self._session.graph))
        training_epoch = self._dataset.get_training_epoch(self._hparams.batch_size, training_step)

        # Perform the training, and retrieve feed_dict for evaluation phase
        self.training(training_handle, batch)

        self._on_after_training_batch(batch, training_step, training_epoch)

        # Export any experiment-related data
        # -------------------------------------------------------------------------
        # if self._export_opts['export_filters']:
        #   if batch == (num_batches - 1) or (batch + 1) % self._export_opts['interval_batches'] == 0:
        #     self.export(self._session, feed_dict)

        # if self._export_opts['export_checkpoint']:
        #   if batch == (num_batches - 1) or (batch + 1) % self._export_opts['interval_batches'] == 0:
        #     self._saver.save(self._session, os.path.join(self._summary_dir, 'model.ckpt'), global_step=batch + 1)

        # # evaluation: every N steps, test the encoding model
        # if evaluate:
        #   if (batch + 1) % self._eval_opts['interval_batches'] == 0:  # defaults to once per batches
        #     self.helper_evaluate(batch)

      logging.info('Training & optional evaluation complete.')

      self._run_profiler()

    elif evaluate:  # train is False
      self.helper_evaluate(0)
    else:
      logging.warning("Both 'train' and 'evaluate' flag are False, so nothing to run.")
