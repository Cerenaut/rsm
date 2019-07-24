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

"""ImageSequenceWorkflow class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import numpy as np
import tensorflow as tf

from pagi.utils.np_utils import np_pad_with
from pagi.utils.np_utils import np_accuracy
from pagi.utils.moving_average_summaries import MovingAverageSummaries
from pagi.workflows.workflow import Workflow

from rsm.components.sequence_memory_stack import SequenceMemoryStack


class ImageSequenceWorkflow(Workflow):
  """(Partially Observable) Image sequence workflow."""

  @staticmethod
  def default_opts():
    """Builds an HParam object with default workflow options."""
    return tf.contrib.training.HParams(
        summarize=True,
        evaluate=True,
        train=True,
        training_progress_interval=0,
        average_accuracy_interval=100,
        sequence='0123',  # Specify a filename, string or "random"
        sequence_length=8,  # Length of generated sequence if using "random" mode
        example_type='random',  # Specify: random, same, specific
        specific_examples=[1, 3, 5, 7, 2, 0, 13, 15, 17, 4]  # Used with "specific", indexed by label
    )

  def _setup_dataset(self):
    """Setup the dataset and retrieve inputs, labels and initializers"""
    with tf.variable_scope('dataset'):
      self._dataset = self._dataset_type(self._dataset_location)
      self._dataset.set_batch_size(self._hparams.batch_size)

      # Dataset for training
      train_dataset = self._dataset.get_train(options=self._opts)
      train_dataset = train_dataset.apply(tf.contrib.data.batch_and_drop_remainder(self._hparams.batch_size))
      train_dataset = train_dataset.prefetch(1)
      train_dataset = train_dataset.repeat()  # repeats indefinitely

      self._placeholders['dataset_handle'] = tf.placeholder(
          tf.string, shape=[], name='dataset_handle')

      # Setup dataset iterators
      with tf.variable_scope('dataset_iterators'):
        self._iterator = tf.data.Iterator.from_string_handle(self._placeholders['dataset_handle'],
                                                             train_dataset.output_types, train_dataset.output_shapes)
        self._init_iterators()

        self._dataset_iterators = {}

        with tf.variable_scope('train_dataset'):
          self._dataset_iterators['training'] = train_dataset.make_initializable_iterator()

  def _init_iterators(self):
    self._inputs, self._labels = self._iterator.get_next()

  def _setup_component(self):
    """Setup the component"""

    labels_one_hot = tf.one_hot(self._labels, self._dataset.num_classes)
    labels_one_hot_shape = labels_one_hot.get_shape().as_list()

    # Create the encoder component
    # -------------------------------------------------------------------------
    self._component = self._component_type()
    self._component.build(self._inputs, self._dataset.shape, labels_one_hot, labels_one_hot_shape, self._hparams)

    self._moving_average = MovingAverageSummaries()
    accuracy_interval = self._opts['average_accuracy_interval']
    self._moving_average.set_interval('accuracy', accuracy_interval)

    if self._summarize:
      self._build_summaries()

  def _get_status(self):
    """Return some string proxy for the losses or errors being optimized"""
    loss = self._component.get_values(SequenceMemoryStack.ensemble_loss_sum)
    return loss

  def training(self, training_handle, training_step, training_fetches=None):
    """The training procedure within the batch loop"""

    feed_dict = {
        self._placeholders['dataset_handle']: training_handle
    }

    batch_type = 'training'
    if self._freeze_training:
      batch_type = 'encoding'

    self._component.update_feed_dict(feed_dict, batch_type)
    training_fetches = {'labels': self._labels}
    self._component.add_fetches(training_fetches, batch_type)
    training_fetched = self._session.run(training_fetches, feed_dict=feed_dict)
    self._component.set_fetches(training_fetched, batch_type)
    self._component.write_summaries(training_step, self._writer, batch_type=batch_type)

    labels = training_fetched['labels']
    predicted_labels = self._component.get_values(SequenceMemoryStack.ensemble_top_1)

    if self._hparams.predictor_optimize == 'accuracy':
      accuracy = np_accuracy(predicted_labels, labels)
      self._moving_average.update('accuracy', accuracy, self._writer, batch_type, batch=training_step,
                                  prefix=self._component.name)

    # Output as feedback for next step
    self._component.update_recurrent()
    self._component.update_statistics(self._session) # only when training

    return feed_dict

  def run(self, num_batches, evaluate, train=True):
    """Run Experiment"""
    del evaluate

    if train:
      training_handle = self._session.run(self._dataset_iterators['training'].string_handle())
      self._session.run(self._dataset_iterators['training'].initializer)

      self._on_before_training_batches()

      for batch in range(self._last_step, num_batches):

        training_step = self._session.run(tf.train.get_global_step(self._session.graph))
        training_epoch = self._dataset.get_training_epoch(self._hparams.batch_size, training_step)

        # Perform the training, and retrieve feed_dict for evaluation phase
        feed_dict = self.training(training_handle, batch)

        self._on_after_training_batch(batch, training_step, training_epoch)

        # Export any experiment-related data
        # -------------------------------------------------------------------------
        if self._export_opts['export_filters']:
          if (batch + 1) % self._export_opts['interval_batches'] == 0:
            self.export(self._session, feed_dict)

        if self._export_opts['export_checkpoint']:
          if (batch + 1) % num_batches == 0:
            self._saver.save(self._session, os.path.join(self._summary_dir, 'model.ckpt'), global_step=batch + 1)
    else:
      logging.warning("Both 'train' and 'evaluate' flag are False, so nothing to run.")


  # Summaries
  # -------------------------------------------------------------------------
  def get_num_states(self):
    num_states = len(self._opts['sequence'])
    return num_states

  def _build_summaries(self):
    """Build TensorBoard summaries for multiple modes."""
    batch_types = ['training', 'encoding']
    if self._freeze_training:
      batch_types.remove('training')
    self._component.build_summaries(batch_types)  # Ask the component to unpack for you

    num_bins = self.get_num_states()
    self._build_sequence_error_summary(batch_types, num_bins=num_bins)

  def _compute_sequence_error(self, batch, predictions, labels):
    """Builds a sequence error bin for histogram summary."""
    sequence_len = len(self._opts['sequence'])
    correct_predictions = np.equal(labels, predictions)  # pylint: disable=assignment-from-no-return
    sequence_bins = dict.fromkeys(list(range(sequence_len)), 0)

    offset = batch

    for (k,), pred in np.ndenumerate(correct_predictions):
      sequence_idx = (k + offset) % sequence_len
      if pred == 0:
        sequence_bins[sequence_idx] += 1
      logging.debug('Batch: %s, Seq. idx: %s, Label: %s, Prediction: %s', batch, sequence_idx,
                    self._opts['sequence'][sequence_idx], pred)

    return np.array(list(sequence_bins.values()))

  def _build_sequence_error_summary(self, batch_types, num_bins):
    """Represents sequence errors as cells in a grid indicated by pixel intensity."""
    self._placeholders['sequence_error'] = tf.placeholder_with_default(
        input=tf.zeros(num_bins, tf.float32), shape=num_bins, name='sequence_error')
    self._sequence_error_summary = dict.fromkeys(batch_types)

    h, w, dim, pad = 20, 20, 1, 2
    grid_shape = [h + (2 * pad), (w + (2 * pad)) * num_bins, dim]

    for batch_type in batch_types:
      with tf.name_scope(self._component.name + '/summaries/' + batch_type):

        def image_to_grid(sequence_error):
          """Converts an image vector to a grid with borders."""
          grid = []
          for error in sequence_error:
            # Border colour
            color = 2

            error = np.tile(error, [h, w]).astype(np.float32)
            error_image = np.pad(error, pad, np_pad_with, padder=color)
            grid.append(error_image)

          return np.concatenate(grid[:len(sequence_error)], axis=1)

        sequence_error = self._placeholders['sequence_error']
        sequence_error_grid = tf.py_func(image_to_grid, [sequence_error], tf.float32)

        sequence_error_image = tf.reshape(sequence_error_grid,
                                          [1, grid_shape[0], grid_shape[1], grid_shape[2]])

        self._sequence_error_summary[batch_type] = tf.summary.image(
            'sequence_error_summary', sequence_error_image, max_outputs=1)

  def _write_sequence_error_summary(self, batch, sequence_error, batch_type):
    """Evaluate the sequence error summary, and write summary events to disk."""
    summary_values = self._session.run(self._sequence_error_summary[batch_type], feed_dict={
        self._placeholders['sequence_error']: sequence_error
    })
    self._writer.add_summary(summary_values, batch)
    self._writer.flush()
