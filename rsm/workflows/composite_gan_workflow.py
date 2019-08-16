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

"""CompositeGANWorkflow class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from pagi.workflows.workflow import Workflow

from rsm.components.composite_rsm_stack import CompositeRSMStack


class CompositeGANWorkflow(Workflow):
  """A simple workflow for GANs."""

  @staticmethod
  def default_opts():
    """Builds an HParam object with default workflow options."""
    opts = Workflow.default_opts()
    opts.add_hparam('pretrain_steps', 2000)
    return opts

  def _train_prior(self, fetches, feed_dict, global_step):
    batch_type = 'training'
    if global_step > self._opts['pretrain_steps']:
      batch_type = 'encoding'

    return self._do_batch(fetches, feed_dict, batch_type, global_step)

  def _train_gan(self, fetched, global_step):
    """Train a GAN by doing a discriminator step, followed by generator step."""
    if self._hparams.build_gan and global_step > self._opts['pretrain_steps']:
      gan_step = int(global_step % (self._opts['pretrain_steps'] + 1e-8))
      disc_input_noise = self._disc_input_noise[gan_step]

      def build_feed_dict(gen_inputs, real_inputs):
        gan_dual = self._component.get_sub_component(CompositeRSMStack.gan_name).get_dual()
        fetches = {}
        feed_dict = {
            gan_dual.get_pl('gen_inputs'): gen_inputs,
            gan_dual.get_pl('real_inputs'): real_inputs,
            gan_dual.get_pl('noise_param'): disc_input_noise
        }
        return fetches, feed_dict

      real_inputs = fetched['inputs']
      real_inputs = 2 * real_inputs - 1  # Normalize to [-1, 1]
      gen_inputs = self._component.get_gan_inputs()

      fetches, feed_dict = build_feed_dict(gen_inputs, real_inputs)
      batch_type = self._component.gan_name + '-discriminator_training'
      self._do_batch(fetches, feed_dict, batch_type, global_step)

      fetches, feed_dict = build_feed_dict(gen_inputs, real_inputs)
      batch_type = self._component.gan_name + '-generator_training'
      self._do_batch(fetches, feed_dict, batch_type, global_step)

  def training(self, dataset_handle, global_step):  # pylint: disable=arguments-differ
    """The training procedure within the batch loop"""

    fetches = {'inputs': self._inputs}

    feed_dict = {
        self._placeholders['dataset_handle']: dataset_handle
    }

    _, _, fetched = self._train_prior(fetches, feed_dict, global_step)
    self._train_gan(fetched, global_step)

  def _do_batch(self, fetches, feed_dict, batch_type, global_step):
    """The training procedure within the batch loop"""
    self._component.update_feed_dict(feed_dict, batch_type)
    self._component.add_fetches(fetches, batch_type)

    fetched = self.session_run(fetches, feed_dict=feed_dict)
    self._component.set_fetches(fetched, batch_type)
    self._component.write_summaries(global_step, self._writer, batch_type=batch_type)

    return fetches, feed_dict, fetched

  def run(self, num_batches, evaluate, train=True):
    if self._hparams.build_gan:
      gan_batches = (num_batches - 1) - self._opts['pretrain_steps']
      self._disc_input_noise = np.linspace(0.1, 0.0, num=gan_batches)

    super().run(num_batches, evaluate, train)
