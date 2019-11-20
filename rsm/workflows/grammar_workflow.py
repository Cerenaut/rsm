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

"""VOMGrammarWorkflow class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from pagi.utils import tf_utils
from pagi.utils.np_utils import np_accuracy

from rsm.workflows.image_sequence_workflow import ImageSequenceWorkflow

from rsm.components.sequence_memory_stack import SequenceMemoryStack
from rsm.components.sequence_memory_layer import SequenceMemoryLayer


class GrammarWorkflow(ImageSequenceWorkflow):
  """Sequence memory workflow for grammar-based problems. The grammar is used to generate sequences."""

  TEST_STATE_PREDICTIONS_COLLECTION = {
      'num_states': [],
      'num_correct': [],
      'loss': []
  }

  def __init__(self, session, dataset_type, dataset_location, component_type, hparams_override, eval_opts, export_opts,
               opts=None, summarize=True, seed=None, summary_dir=None, checkpoint_opts=None):
    super().__init__(session, dataset_type, dataset_location, component_type, hparams_override, eval_opts, export_opts,
                     opts, summarize, seed, summary_dir, checkpoint_opts)
    self._batch_ts_losses = []

  @staticmethod
  def default_opts():
    """Builds an HParam object with default workflow options."""
    return tf.contrib.training.HParams(
        summarize=True,
        evaluate=True,
        train=True,
        training_progress_interval=0,
        average_accuracy_interval=100,
        grammar_fixed_set=True,
        grammar_fixed_train_size=256,
        grammar_fixed_test_size=256,
        grammar_states={},
        grammar_transitions={},
        example_type='same',  # Specify: random, same, specific
        specific_examples=[1, 3, 5, 7, 2, 0, 13, 15, 17, 4]  # Used with "specific", indexed by label
    )

  def _init_iterators(self):
    self._inputs, self._labels, self._states = self._iterator.get_next()

  def get_num_states(self):
    num_states = len(self._opts['grammar_states'])
    return num_states

  def _compute_test_state_accuracy(self, predictions, labels, states, losses):
    """
    Calculate the accuracy and average accuracy over N batches.

    :param predictions predicted label of each sample in the batch (list)
    :param labels label of each sample in the batch (list)
    :param states current state for each sample in the batch (list) -> informs if 'test state' or not (and other stuff)
    :param losses for computing direct vom recon lost for 'test states'
    :return accuracy, accuracy_average
    """

    accuracy, accuracy_average, loss, loss_average = None, None, None, None

    # Compute correct predictions from test states only
    correct_predictions = []
    sum_loss = 0.0

    self._batch_ts_losses = []
    for pred, label, state, loss in zip(predictions, labels, states, losses):
      state = state.decode('utf-8')
      is_test_state = self._opts['grammar_states'][state]['test']

      if is_test_state:
        correct_prediction = np.equal(label, pred)  # pylint: disable=assignment-from-no-return
        correct_predictions.append(correct_prediction)
        sum_loss += loss    # sum of recon loss for all test states, for the batch
        self._batch_ts_losses.append(loss)

    # There may be no test states in a batch
    num_states = len(correct_predictions)  # number of TEST states only
    num_correct = np.sum(correct_predictions)

    if correct_predictions:
      correct_predictions = np.array(correct_predictions, dtype=np.float32)
      accuracy = np.average(correct_predictions)
      loss = sum_loss / float(num_states)   # recon loss per test state

    # print('num_states:', num_states, 'num_correct:', num_correct)
    self.TEST_STATE_PREDICTIONS_COLLECTION['num_states'].append(num_states)
    self.TEST_STATE_PREDICTIONS_COLLECTION['num_correct'].append(num_correct)
    self.TEST_STATE_PREDICTIONS_COLLECTION['loss'].append(sum_loss)

    # Track per-batch accuracy & compute average accuracy ever N batches
    if len(self.TEST_STATE_PREDICTIONS_COLLECTION['num_states']) == self._opts['average_accuracy_interval']:

      num_states_sum = np.sum(self.TEST_STATE_PREDICTIONS_COLLECTION['num_states'])
      num_correct_sum = np.sum(self.TEST_STATE_PREDICTIONS_COLLECTION['num_correct'])
      sum_loss_sum = np.sum(self.TEST_STATE_PREDICTIONS_COLLECTION['loss'])   # sum of 'batch sum of recon loss'

      accuracy_average = num_correct_sum / num_states_sum

      loss_average = sum_loss_sum / float(num_states_sum)

      # Reset the collection
      self.TEST_STATE_PREDICTIONS_COLLECTION = {  # pylint: disable=C0103
          'num_states': [],
          'num_correct': [],
          'loss': []
      }

    return accuracy, accuracy_average, loss, loss_average

  def _compute_sequence_error(self, batch, predictions, labels, states):  # pylint: disable=W0221
    """Builds a sequence error bin for histogram summary."""
    sequence_bins = dict.fromkeys(self._opts['grammar_states'].keys(), 0)

    for pred, label, state in zip(predictions, labels, states):
      if pred != label:
        state = state.decode('utf-8')  # bytes -> string
        sequence_bins[state] += 1

    return np.array(list(sequence_bins.values()))

  def _write_test_state_accuracy_summary(self, batch, accuracy, accuracy_average, loss, loss_average, batch_type):
    """Create off-graph TensorBoard value summaries, and write events to disk."""
    if accuracy is None and accuracy_average is None and loss is None and loss_average is None:
      return

    if self._batch_ts_losses:
      summary = tf_utils.histogram_summary(tag=self._component.name + '/summaries/' + batch_type + '/ts_batch_loss',
                                           values=self._batch_ts_losses)
    else:
      summary = tf.Summary()

    if accuracy is not None:
      summary.value.add(tag=self._component.name + '/summaries/' + batch_type + '/test_state_accuracy',
                        simple_value=accuracy)
    if loss is not None:
      summary.value.add(tag=self._component.name + '/summaries/' + batch_type + '/test_state_loss',
                        simple_value=loss)
    if accuracy_average is not None:
      summary.value.add(tag=self._component.name + '/summaries/' + batch_type + '/test_state_accuracy_average',
                        simple_value=accuracy_average)
    if loss_average is not None:
      summary.value.add(tag=self._component.name + '/summaries/' + batch_type + '/test_state_loss_average',
                        simple_value=loss_average)

    self._writer.add_summary(summary, batch)
    self._writer.flush()

  def _compute_end_state_mask(self, labels, states):
    """Detect end of sequence and clear the component's history."""
    del labels

    history_mask = np.ones(self._hparams.batch_size)

    for b in range(self._hparams.batch_size):
      state = states[b].decode('utf-8')  # bytes -> string
      is_end_state = self._opts['grammar_states'][state]['end']
      if is_end_state:
        history_mask[b] = 0.0

    self._component.update_history(self._session, history_mask)

  def training_step(self, training_handle, training_step, phase_change=False):  # pylint: disable=arguments-differ
    """The training procedure within the batch loop"""
    del phase_change

    feed_dict = {
        self._placeholders['dataset_handle']: training_handle
    }

    batch_type = 'training'
    if self._freeze_training:
      batch_type = 'encoding'

    self._component.update_feed_dict(feed_dict, batch_type)
    training_fetches = {'labels': self._labels, 'states': self._states}
    self._component.add_fetches(training_fetches, batch_type)
    training_fetched = self._session.run(training_fetches, feed_dict=feed_dict)
    self._component.set_fetches(training_fetched, batch_type)
    self._component.write_summaries(training_step, self._writer, batch_type=batch_type)

    states = training_fetched['states']
    labels = training_fetched['labels']

    predicted_labels = self._component.get_values(SequenceMemoryStack.ensemble_top_1)
    sample_loss = self._component.get_layer(0).get_values(SequenceMemoryLayer.sum_abs_error)

    self._compute_end_state_mask(labels, states)

    # Classification metrics
    if self._hparams.predictor_optimize == 'accuracy':
      accuracy = np_accuracy(predicted_labels, labels)
      self._moving_average.update('accuracy', accuracy, self._writer, batch_type,
                                  batch=training_step, prefix=self._component.name)

      # Check if grammar has any test state
      has_any_test_state = any([self._opts['grammar_states'][state]['test'] \
                                for state in self._opts['grammar_states'].keys()])

      # Compute test state accuracy only if grammar has any test states
      if has_any_test_state:
        ts_accuracy, ts_accuracy_average, ts_loss, ts_loss_average = self._compute_test_state_accuracy(
            predicted_labels, labels, states, sample_loss)

        self._write_test_state_accuracy_summary(training_step, ts_accuracy, ts_accuracy_average, ts_loss,
                                                ts_loss_average, batch_type)

      sequence_error = self._compute_sequence_error(training_step, predicted_labels, labels, states)
      self._write_sequence_error_summary(training_step, sequence_error, batch_type)

    # Output as feedback for next step
    self._component.update_recurrent_and_feedback()
    self._component.update_statistics(batch_type, self._session)

    return feed_dict
