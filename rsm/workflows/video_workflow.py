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

"""VideoWorkflow class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation

from rsm.workflows.image_sequence_workflow import ImageSequenceWorkflow

from rsm.components.sequence_memory_stack import SequenceMemoryStack
from rsm.components.sequence_memory_layer import SequenceMemoryLayer

class VideoWorkflow(ImageSequenceWorkflow):
  """Workflow for dealing with video data i.e. a sequence of image frames."""

  def __init__(self, session, dataset_type, dataset_location, component_type, hparams_override, eval_opts, export_opts,
               opts=None, summarize=True, seed=None, summary_dir=None, checkpoint_opts=None):

    self._batch_ts_losses = []

    self._output_frames = []
    self._groundtruth_frames = []

    super().__init__(session, dataset_type, dataset_location, component_type, hparams_override, eval_opts, export_opts,
                     opts, summarize, seed, summary_dir, checkpoint_opts)

  @staticmethod
  def default_opts():
    """Builds an HParam object with default workflow options."""
    return tf.contrib.training.HParams(
        summarize=True,
        evaluate=True,
        train=True,
        training_progress_interval=0,
        prime=None,
        prime_num_frames=None,
        num_testing_batches=None,
        testing_progress_interval=1,
        clear_before_test=True,
        frame_padding_size=0,
        frame_padding_value=0.0,
        profile_file=None
    )

  def _setup_dataset(self):
    """Setup the dataset and retrieve inputs, labels and initializers"""
    with tf.variable_scope('dataset'):
      self._dataset = self._dataset_type(self._dataset_location)
      self._dataset.set_batch_size(self._hparams.batch_size)

      # Dataset for training
      train_dataset = self._dataset.get_train(options=self._opts)
      train_dataset = train_dataset.batch(self._hparams.batch_size, drop_remainder=True)

      train_dataset = train_dataset.prefetch(1)
      train_dataset = train_dataset.repeat()  # repeats indefinitely

      # Dataset for testing
      test_dataset = self._dataset.get_test(options=self._opts)
      test_dataset = test_dataset.batch(self._hparams.batch_size, drop_remainder=True)
      test_dataset = test_dataset.prefetch(1)
      test_dataset = test_dataset.repeat()  # repeats indefinitely

      self._placeholders['dataset_handle'] = tf.placeholder(
          tf.string, shape=[], name='dataset_handle')

      # Setup dataset iterators
      with tf.variable_scope('dataset_iterators'):
        self._iterator = tf.data.Iterator.from_string_handle(self._placeholders['dataset_handle'],
                                                             train_dataset.output_types, train_dataset.output_shapes)
        self._init_iterators()

        self._dataset_iterators = {}

        with tf.variable_scope('train_dataset'):
          self._dataset_iterators['train'] = train_dataset.make_initializable_iterator()

        with tf.variable_scope('test_dataset'):
          self._dataset_iterators['test'] = test_dataset.make_initializable_iterator()

  def _init_iterators(self):
    self._inputs, self._labels, self._states, self._end_states = self._iterator.get_next()
    self._sequence_length = self._dataset.num_frames

  def _setup_component(self):
    """Setup the component"""

    labels_one_hot = tf.one_hot(self._labels, self._dataset.num_classes)
    labels_one_hot_shape = labels_one_hot.get_shape().as_list()

    # Create the encoder component
    # -------------------------------------------------------------------------
    self._component = self._component_type()
    self._component.build(self._inputs, self._dataset.shape, labels_one_hot, labels_one_hot_shape, self._hparams)

    if self._summarize:
      self._build_summaries()

  def _build_summaries(self):
    """Build TensorBoard summaries for multiple modes."""
    batch_types = ['training', 'encoding']
    if self._freeze_training:
      batch_types.remove('training')
    self._component.build_summaries(batch_types)  # Ask the component to unpack for you

  def _get_status(self):
    """Return some string proxy for the losses or errors being optimized"""
    loss = self._component.get_values(SequenceMemoryStack.prediction_loss)
    return loss

  def _compute_end_state_mask(self, states):
    """Detect end of sequence and clear the component's history."""

    history_mask = np.ones(self._hparams.batch_size)

    for b in range(self._hparams.batch_size):
      end_state = states[b]
      if end_state:  # Reached end of sequence
        history_mask[b] = 0.0

    clear_previous = True
    self._component.update_history(self._session, history_mask, clear_previous)

  def training_step(self, dataset_handle, global_step, phase_change=None):  # pylint: disable=arguments-differ
    """The training procedure within the batch loop"""
    del phase_change

    if self._freeze_training:
      batch_type = 'encoding'
    else:
      batch_type = 'training'

    data_subset = 'train'

    self._do_batch(dataset_handle, batch_type, data_subset, global_step)

  def testing(self, dataset_handle, global_step):
    """The testing procedure within the batch loop"""

    batch_type = 'encoding'
    data_subset = 'test'

    self._do_batch(dataset_handle, batch_type, data_subset, global_step)

  def _compute_prime_end(self, states):
    """Start self-looping when model is primed."""
    for b in range(self._hparams.batch_size):
      state = states[b]

      decoding = self.get_decoded_frame()
      previous = self.get_previous_frame()

      # print('previous', np.min(previous), np.max(previous), np.mean(previous), np.std(previous))
      # print('decoding', np.min(decoding), np.max(decoding), np.mean(decoding), np.std(decoding))

      # Check if the model has been primed
      if state >= (self._opts['prime_num_frames'] - 1):
        # Replace groundtruth with output decoding for the next step
        previous[b] = decoding[b]
        self.set_previous_frame(previous)

  def set_previous_frame(self, previous):
    self._component.get_layer(0).get_dual().set_values('previous', previous)

  def get_decoded_frame(self):
    return self._component.get_layer(0).get_values(SequenceMemoryLayer.decoding)

  def get_previous_frame(self):
    return self._component.get_layer(0).get_values(SequenceMemoryLayer.previous)

  def _do_batch(self, dataset_handle, batch_type, data_subset, global_step):
    """The training procedure within the batch loop"""

    feed_dict = {
        self._placeholders['dataset_handle']: dataset_handle
    }

    self._component.update_feed_dict(feed_dict, batch_type)
    fetches = {'inputs': self._inputs, 'states': self._states, 'end_states': self._end_states}
    self._component.add_fetches(fetches, batch_type)

    if self.do_profile():
      logging.info('Running batch with profile')

    fetched = self.session_run(fetches, feed_dict=feed_dict)
    self._component.set_fetches(fetched, batch_type)
    self._component.write_summaries(global_step, self._writer, batch_type=batch_type)

    self._inputs_vals = fetched['inputs']
    self._states_vals = fetched['states']
    self._end_states_vals = fetched['end_states']

    self._do_batch_after_hook(global_step, batch_type, fetched, feed_dict, data_subset)

    return feed_dict

  all_mse_gan = []
  all_mse_rsm = []
  all_mse_prev = []

  all_bce_gan = []
  all_bce_rsm = []
  all_bce_prev = []

  previous_frame = None

  def _cross_entropy_loss(self, z, y, eps=1e-9):
    loss = z * np.log(y + eps) + (1 - z) * np.log((1 - y) + eps)
    loss = -np.sum(loss) / self._hparams.batch_size
    return loss

  def _do_batch_after_hook(self, global_step, batch_type, fetched, feed_dict, data_subset):
    """Things to do after a batch is completed."""
    del feed_dict, fetched

    # Test-time conditions
    if batch_type == 'encoding' and data_subset == 'test':
      decoding = self.get_decoded_frame()

      # Prime the model using the first N frames of a sequence
      if self._opts['prime']:
        self._compute_prime_end(self._states_vals)

      # Collect samples for video export
      self._output_frames.append(decoding)
      self._groundtruth_frames.append(self._inputs_vals)

      if self.previous_frame is None:
        self.previous_frame = np.zeros_like(self._inputs_vals)

      def denormalize(arr):
        eps = 1e-8
        return ((arr - arr.min()) * (1/(arr.max() - arr.min() + eps) * 255)).astype('uint8')

      def unpad(x, pad_width):
        slices = []
        for c in pad_width:
          e = None if c[1] == 0 else -c[1]
          slices.append(slice(c[0], e))
        return x[tuple(slices)]

      A = self._inputs_vals
      B = decoding
      C = self.previous_frame
      D = self._component.get_gan_inputs()

      # Reverse the padding
      padding_size = self._opts['frame_padding_size']
      if padding_size > 0:
        pad_h = [padding_size] * 2
        pad_w = [padding_size] * 2

        pad_width = [[0, 0], pad_h, pad_w, [0, 0]]

        A = unpad(A, pad_width)
        B = unpad(B, pad_width)
        C = unpad(C, pad_width)
        D = unpad(D, pad_width)

      num_features = np.prod(A.shape[1:])

      mse_gan = ((A - B)**2).mean(axis=None)
      mse_rsm = ((A - D)**2).mean(axis=None)
      mse_prev = ((A - C)**2).mean(axis=None)

      bce_gan = self._cross_entropy_loss(A, B)
      bce_rsm = self._cross_entropy_loss(A, D)
      bce_prev = self._cross_entropy_loss(A, C)

      self.all_mse_gan.append(mse_gan * num_features)
      self.all_mse_rsm.append(mse_rsm * num_features)
      self.all_mse_prev.append(mse_prev * num_features)

      self.all_bce_gan.append(bce_gan)
      self.all_bce_rsm.append(bce_rsm)
      self.all_bce_prev.append(bce_prev)

      avg_mse_gan = np.average(self.all_mse_gan)
      avg_mse_rsm = np.average(self.all_mse_rsm)
      avg_mse_prev = np.average(self.all_mse_prev)

      avg_bce_gan = np.average(self.all_bce_gan)
      avg_bce_rsm = np.average(self.all_bce_rsm)
      avg_bce_prev = np.average(self.all_bce_prev)

      # Write summaries
      summary = tf.Summary()
      summary.value.add(tag=self._component.name + '/summaries/' + batch_type + '/mse_gan',
                        simple_value=mse_gan)
      summary.value.add(tag=self._component.name + '/summaries/' + batch_type + '/mse_rsm',
                        simple_value=mse_rsm)
      summary.value.add(tag=self._component.name + '/summaries/' + batch_type + '/mse_prev',
                        simple_value=mse_prev)
      summary.value.add(tag=self._component.name + '/summaries/' + batch_type + '/avg_mse_gan',
                        simple_value=avg_mse_gan)
      summary.value.add(tag=self._component.name + '/summaries/' + batch_type + '/avg_mse_rsm',
                        simple_value=avg_mse_rsm)
      summary.value.add(tag=self._component.name + '/summaries/' + batch_type + '/avg_mse_prev',
                        simple_value=avg_mse_prev)

      summary.value.add(tag=self._component.name + '/summaries/' + batch_type + '/bce_gan',
                        simple_value=bce_gan)
      summary.value.add(tag=self._component.name + '/summaries/' + batch_type + '/bce_rsm',
                        simple_value=bce_rsm)
      summary.value.add(tag=self._component.name + '/summaries/' + batch_type + '/bce_prev',
                        simple_value=bce_prev)
      summary.value.add(tag=self._component.name + '/summaries/' + batch_type + '/avg_bce_gan',
                        simple_value=avg_bce_gan)
      summary.value.add(tag=self._component.name + '/summaries/' + batch_type + '/avg_bce_rsm',
                        simple_value=avg_bce_rsm)
      summary.value.add(tag=self._component.name + '/summaries/' + batch_type + '/avg_bce_prev',
                        simple_value=avg_bce_prev)

      self._writer.add_summary(summary, global_step)
      self._writer.flush()

      self.previous_frame = self._inputs_vals

    # Output as feedback for next step
    self._component.update_recurrent_and_feedback()
    self._component.update_statistics(batch_type, self._session)

    # Resets the history at end of a sequence
    #print('after batch #', global_step)
    self._compute_end_state_mask(self._end_states_vals)

  def frames_to_video(self, input_frames, filename=None):
    """Convert given frames to video format, and export it to disk."""
    plt.switch_backend('agg')

    if filename is None:
      filename = 'video'

    def chunks(l, n):
      """Yield successive n-sized chunks from l."""
      for i in range(0, len(l), n):
        yield l[i:i + n]

    sequence_chunks = list(chunks(input_frames, self._sequence_length))

    for i, sequence in enumerate(sequence_chunks):
      output_frames = []
      sequence_filename = filename + '.' + str(i)
      filepath = os.path.join(self._summary_dir, sequence_filename)

      for sample in sequence:
        cmap = None
        frame = sample[0]

        if frame.shape[2] == 1:
          frame = frame.reshape(frame.shape[0], frame.shape[1])
          cmap = 'gray'

        output_frames.append((frame, cmap))

      # Export video
      fig1 = plt.figure(1)
      video_frames = []

      for j, (frame, cmap) in enumerate(output_frames):
        title = 'Priming'
        if j >= self._opts['prime_num_frames']:
          title = 'Self-looping'

        ttl = plt.text(4, 1.2, title, horizontalalignment='center', verticalalignment='bottom')

        video_frames.append([plt.imshow(frame, cmap=cmap, animated=True), ttl])

      ani = animation.ArtistAnimation(fig1, video_frames, interval=50, blit=True,
                                      repeat_delay=1000)
      ani.save(filepath + '.mp4')
      plt.close(fig1)

      # Export frame by frame image
      num_frames = len(output_frames)
      fig2 = plt.figure(figsize=(num_frames + 1, 2))
      gs1 = gridspec.GridSpec(1, num_frames + 1)
      gs1.update(wspace=0.025, hspace=0.05) # set the spacing between axes.
      plt.tight_layout()

      frame_idx = 0
      for j in range(num_frames + 1):
        if j == self._opts['prime_num_frames']:
          frame, cmap = output_frames[0]
          frame = np.zeros_like(frame)
        else:
          frame, cmap = output_frames[frame_idx]
          frame_idx += 1

        ax = plt.subplot(gs1[j])
        ax.axis('off')
        ax.set_aspect('equal')
        ax.imshow(frame, cmap=cmap)
      fig2.savefig(filepath + '.png', bbox_inches='tight')
      plt.close(fig2)

  def run(self, num_batches, evaluate, train=True):
    super(ImageSequenceWorkflow, self).run(num_batches, evaluate, train)  # pylint: disable=bad-super-call

  def helper_evaluate(self, batch):
    """Evaluation method."""
    logging.info('Evaluate starting...')

    self._test_on_training_set = False
    if self._test_on_training_set is True:
      testing_handle = self._session.run(self._dataset_iterators['train'].string_handle())
    else:
      testing_handle = self._session.run(self._dataset_iterators['test'].string_handle())
      self._session.run(self._dataset_iterators['test'].initializer)

    # Clear the history of the component, because there'll be a discontinuity in the observed input
    # We just clear it ONCE before starting the new sequence
    clear_before_test = self._opts['clear_before_test']
    if clear_before_test:
      logging.info('Clearing memory history before testing set...')
      history_mask = np.zeros(self._hparams.batch_size) # Clear all
      self._component.update_history(self._session, history_mask)

    # Apply a fixed number of test batches. Might take time to warm up and dataset repeats.
    # We can pick an interval from the results later
    num_testing_batches = self._opts['num_testing_batches']

    for test_batch in range(0, num_testing_batches):
      # Perform the training, and retrieve feed_dict for evaluation phase
      do_print = True
      testing_progress_interval = self._opts['testing_progress_interval']

      if testing_progress_interval > 0:
        if (test_batch % testing_progress_interval) != 0:
          do_print = False

      if do_print:
        global_step = test_batch
        logging.info('Test batch %d of %d, global step: %d', global_step, num_testing_batches, batch)

      self.testing(testing_handle, test_batch)

    export_videos = True

    if export_videos:
    # Create videos from frames
      if self._output_frames:
        self.frames_to_video(self._output_frames, filename='output')

      if self._groundtruth_frames:
        self.frames_to_video(self._groundtruth_frames, filename='groundtruth')

      if  self._output_frames and self._groundtruth_frames:
        stacked_frames = np.concatenate(
            [self._output_frames, np.zeros_like(self._output_frames), self._groundtruth_frames], axis=4)
        self.frames_to_video(stacked_frames, filename='output_groundtruth')
