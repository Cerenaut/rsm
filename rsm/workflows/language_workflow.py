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

"""LanguageWorkflow class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import math

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from pagi.utils.np_utils import np_softmax
from pagi.utils.np_utils import np_accuracy
from pagi.utils.moving_average_summaries import MovingAverageSummaries
from pagi.utils.embedding import Embedding
from pagi.workflows.workflow import Workflow

from rsm.components.sequence_memory_stack import SequenceMemoryStack
from rsm.components.token_embedding_decoder import TokenEmbeddingDecoder

class LanguageWorkflow(Workflow):
  """Next word prediction perplexity for natural language modelling."""

  @staticmethod
  def default_opts():
    """Builds an HParam object with default workflow options."""
    hparams = Workflow.default_opts()

    # Training features
    hparams.add_hparam('random_offsets', True)
    hparams.add_hparam('max_sequence_length', 0)
    hparams.add_hparam('stochastic_forgetting_probability', 0.0)

    # Testing & measurement
    hparams.add_hparam('num_testing_batches', 1)
    hparams.add_hparam('average_accuracy_interval', 100)
    hparams.add_hparam('perplexity_interval', 100)
    hparams.add_hparam('debug_start', -1)
    hparams.add_hparam('test_primer', None)
    hparams.add_hparam('test_distributions_filename', None)

    # Dataset & embedding
    hparams.add_hparam('corpus_train', '')
    hparams.add_hparam('corpus_test', '')

    hparams.add_hparam('embedding_file', '')
    hparams.add_hparam('token_file', '')
    hparams.add_hparam('token_delimiter', ',')

    return hparams

  def _setup_dataset(self):
    """Setup the dataset and retrieve inputs, labels and initializers"""
    with tf.variable_scope('dataset'):
      #import re # regexp

      self._generated = "" #[]

      self._freq_cells_training = None
      self._freq_cells_testing = None

      random_offsets = self._opts['random_offsets']
      max_sequence_length = self._opts['max_sequence_length']

      corpus_train_file = self._opts['corpus_train']
      corpus_test_file = self._opts['corpus_test']

      embedding_file = self._opts['embedding_file']
      token_file = self._opts['token_file']
      token_delimiter =  self._opts['token_delimiter']

      self._dataset = self._dataset_type(self._dataset_location)
      self._dataset.setup(int(self._hparams.batch_size), random_offsets, max_sequence_length, 
                          corpus_train_file, corpus_test_file,
                          token_file, embedding_file, token_delimiter)

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
        self._inputs, self._labels = self._iterator.get_next()

        self._dataset_iterators = {}

        with tf.variable_scope('train_dataset'):
          self._dataset_iterators['training'] = train_dataset.make_initializable_iterator()

        with tf.variable_scope('test_dataset'):
          self._dataset_iterators['testing'] = test_dataset.make_initializable_iterator()

  def _build_summaries(self):
    """Build TensorBoard summaries for multiple modes."""
    batch_types = ['training', 'encoding']
    if self._freeze_training:
      batch_types.remove('training')
    self._component.build_summaries(batch_types)  # Ask the component to unpack for you

  def _setup_component(self):
    """Setup the component"""

    self._loss_sum = 0.0
    self._loss_smoothed_sum = 0.0
    self._loss_samples = 0.0

    self._test_distributions = None  # Used to store prediction distributions loaded from file, e.g. another model.

    labels_one_hot = tf.one_hot(self._labels, self._dataset.num_classes)
    labels_one_hot_shape = labels_one_hot.get_shape().as_list()

    decoder_name = 'decoder'
    decoder = TokenEmbeddingDecoder(decoder_name, self._dataset)

    self._component = self._component_type()
    self._component.build(self._inputs, self._dataset.shape,
                          labels_one_hot, labels_one_hot_shape,
                          self._hparams, decoder)

    self._moving_average = MovingAverageSummaries()
    accuracy_interval = self._opts['average_accuracy_interval']
    self._moving_average.set_interval('accuracy', accuracy_interval)

    if self._summarize:
      self._build_summaries()

  def _get_status(self):
    """Return some string proxy for the losses or errors being optimized"""
    loss = self._component.get_values(SequenceMemoryStack.ensemble_loss_sum)
    return loss

  def testing(self, dataset_handle, global_step):
    """The testing procedure within the batch loop"""

    # Optionally load a distribution predicted by another external method (in this case, Kneser-Ney)
    test_distributions_filename = self._opts['test_distributions_filename']
    if test_distributions_filename is not None:
      if self._test_distributions is None:
        self._test_distributions = np.load(test_distributions_filename)

      sample = self._test_distributions[global_step]
      batch_sample = np.expand_dims(sample, 0)
      batch_samples = np.tile(batch_sample, [self._hparams.batch_size, 1])

      file_dual = self._component.get_dual().get(SequenceMemoryStack.file)
      file_dual.set_values(batch_samples)

    batch_type = 'encoding'
    data_subset = 'test'
    stochastic_forgetting_probability = 0.0  # Don't forget at test time
    self._do_batch(dataset_handle, batch_type, data_subset, global_step, stochastic_forgetting_probability)

  def training(self, dataset_handle, global_step):  # pylint: disable=arguments-differ
    """The training procedure within the batch loop"""

    batch_type = 'training'
    data_subset = 'train'
    stochastic_forgetting_probability = self._opts['stochastic_forgetting_probability']

    if self._freeze_training:
      batch_type = 'encoding'
      data_subset = 'test'

    self._do_batch(dataset_handle, batch_type, data_subset, global_step, stochastic_forgetting_probability)

  def _do_batch(self, dataset_handle, batch_type, data_subset, global_step, stochastic_forgetting_probability):
    """Execute a single batch"""

    # Option to clear history randomly, to (hopefully) create a more generalizable representation
    # No effect if P=0
    self._component.forget_history(self._session, stochastic_forgetting_probability)

    # Option to let dataset decide when to clear
    # History update with per-batch-sample flag for whether to clear
    max_sequence_length = self._opts['max_sequence_length']
    if max_sequence_length > 0:
      subset = self._dataset.get_subset(data_subset)
      history_mask = subset['mask']
      self._component.update_history(self._session, history_mask)

    # Optionally load a primer text, then self-generate the rest of the test corpus
    # i.e. generative mode.
    primer = self._opts['test_primer']
    if primer is not None:

      word_index = global_step

      # Work out how many tokens, split
      words = Embedding.tokenize_sentence(primer)

      # Initially, just loop the priming string
      if word_index < len(words):
        word = words[word_index]
      else:
        # Override the test token in previous with one from here.
        predictions = self._component.get_predicted_distribution()
        logits_max = np.argmax(predictions, axis=1)  # Top prediction
        logit_index = logits_max[0]  # index of max logit for batch = 0
        word = self._dataset.get_embedding().get_key(logit_index)

        # Place some token from the prediction
        self._generated = self._generated + word + ' '
        print('Generated: ', self._generated)

      # look up the embedding for that token
      embedding_values = self._dataset.get_embedding().get_values(word)
      embedding_values_reshape = np.reshape(embedding_values, [20, 20, 1])
      previous_values = np.expand_dims(embedding_values_reshape, 0)  # Insert batch dimension
      self._component.get_layer(0).get_dual().set_values('previous', previous_values)

    # Provide new data
    feed_dict = {
        self._placeholders['dataset_handle']: dataset_handle
    }

    self._component.update_feed_dict(feed_dict, batch_type)
    fetches = {'labels': self._labels}
    # Warning: double advance of the dataset on changing subset..? After here
    self._component.add_fetches(fetches, batch_type)
    fetched = self._session.run(fetches, feed_dict=feed_dict)
    self._component.set_fetches(fetched, batch_type)
    # Warning: double advance of the dataset on changing subset..? Before here
    self._component.write_summaries(global_step, self._writer, batch_type=batch_type)

    labels = fetched['labels']

    # Calculate stats about predictions
    if self._hparams.predictor_optimize == 'accuracy':
      predicted_labels = self._component.get_values(SequenceMemoryStack.ensemble_top_1)  #get_predicted_labels()

      # Accuracy
      accuracy = np_accuracy(predicted_labels, labels)
      self._moving_average.update('accuracy', accuracy, self._writer, batch_type, batch=global_step,
                                  prefix=self._component.name)

      perplexity_interval = self._opts['perplexity_interval']
      if perplexity_interval >= 0:

        # Calculate cumulative loss and accurate perplexity
        prediction_loss = self._component.get_values(SequenceMemoryStack.ensemble_loss_sum)
        self._loss_sum += prediction_loss
        self._loss_samples += self._hparams.batch_size
        self._loss_mean = self._loss_sum / self._loss_samples
        self._perplexity = math.exp(self._loss_mean)

        if perplexity_interval > 0:
          samples_batches = int(self._loss_samples / self._hparams.batch_size)
          if samples_batches == perplexity_interval:
            logging.info('Perplexity %f Loss mean %f  Samples %f', self._perplexity, self._loss_mean, self._loss_samples)
            self._write_perplexity_summary(batch_type, global_step, self._perplexity)
            self._loss_sum = 0.0
            self._loss_samples = 0
        else:
          self._write_perplexity_summary(batch_type, global_step, self._perplexity)

    debug_start = self._opts['debug_start']
    if debug_start >= 0 and global_step > debug_start:
      predictions = self._component.get_predicted_distribution()
      n = 120  # OR: 25, 120
      # max_perplexity = 5000.0
      perplexity = self._component.get_perplexity()
      # perplexity = min(max_perplexity, perplexity)  # Graphs are illegible otherwise
      # batch_size = predictions.shape[0]
      # num_classes = predictions.shape[1]
      b = 0
      logits_max = np.argmax(predictions, axis=1)
      label = labels[b]
      logit_max = logits_max[b]
      correct = 0
      if label == logit_max:
        correct = 1
      label_word = self._dataset.get_embedding().get_key(label)

      print('OK? ', correct, ' label:', label_word, '-> ', label, ' logit_max: ', logit_max)

      softmax = np_softmax(predictions[b])

      logits_sorted = predictions[b].argsort()[::-1]

      pl_values = np.zeros(n+1)
      pl_truth = np.zeros(n+1)
      pl_labels = []

      show_softmax = True
      if show_softmax:
        pl_values[0] = softmax[label]
      else:
        pl_values[0] = predictions[b][label]

      pl_labels.append(label_word + ' -')

      prediction_rank = np.where(logits_sorted == label)[0]
      print('Rank of true label: ', prediction_rank)

      for i in range(0, n):
        index = logits_sorted[i]
        if show_softmax:
          value = softmax[index]
        else:
          value = predictions[b][index]
        pl_values[i+1] = value

        label_word = self._dataset.get_embedding().get_key(index) + ' ' + str(i+1)
        if index == label:
          pl_truth[i+1] = pl_values[i+1]
        pl_labels.append(label_word)

      # PLOTTING
      # https://matplotlib.org/gallery/statistics/barchart_demo.html
      n_groups = n +1
      index = np.arange(n_groups)
      # bar_width = 0.8
      # opacity = 1.0
      # error_config = {'ecolor': '0.3'}

      fig, ax = plt.subplots()

      # rects1 = ax.bar(index, pl_values, bar_width,
      #                 alpha=opacity, color='b',
      #                 label='Logits')

      # rects2 = ax.bar(index, pl_truth, bar_width,
      #                 alpha=opacity, color='r',
      #                 label='True')

      # rects3 = ax.bar(index, pl_softmax, bar_width,
      #                 alpha=opacity, color='g',
      #                 label='Softmax')

      # n_groups = 5
      # rects1 = ax.bar(index, means_men, bar_width,
      #                 alpha=opacity, color='b',
      #                 error_kw=error_config,
      #                 label='Counts')

      # rects2 = ax.bar(index + bar_width, means_women, bar_width,
      #                 alpha=opacity, color='r',
      #                 yerr=std_women, error_kw=error_config,
      #                 label='Women')

      ax.set_xlabel('Rank')
      ax.set_ylabel('Logit value')
      ax.set_title('#' + str(global_step) + ' Rank ' +str(prediction_rank) + ' Perplexity: ' + str(perplexity))
      ax.set_xticks(index + 0)
      ax.set_xticklabels(pl_labels, rotation='vertical')
      ax.legend()

      fig.tight_layout()
      plt.show()

    # Output as feedback for next step
    self._component.update_recurrent_and_feedback()
    self._component.update_statistics(self._session) # only when training

    return feed_dict

  def _write_perplexity_summary(self, batch_type, batch, perplexity):
    """Create off-graph TensorBoard value summaries, and write events to disk."""
    summary = tf.Summary()
    summary.value.add(tag=self._component.name + '/summaries/' + batch_type + '/average_perplexity',
                      simple_value=perplexity)
    self._writer.add_summary(summary, batch)
    self._writer.flush()

  def helper_evaluate(self, batch):
    """Evaluation method."""

    logging.info('Evaluate starting...')
    self._loss_sum = 0.0
    self._loss_samples = 0.0
    self._loss_mean = 0.0
    self._perplexity = 0.0

    testing_handle = self._session.run(self._dataset_iterators['testing'].string_handle())
    self._session.run(self._dataset_iterators['testing'].initializer)

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

    if self._hparams.predictor_optimize == 'accuracy':
      logging.info('Perplexity %f Loss mean %f  Samples %f', self._perplexity, self._loss_mean, self._loss_samples)
