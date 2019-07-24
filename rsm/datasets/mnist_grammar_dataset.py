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

"""MNIST-based Grammar dataset using the tf.data module."""

import random

import numpy as np
import tensorflow as tf

from datasets.mnist_sequence_dataset import MNISTSequenceDataset


class MNISTGrammarDataset(MNISTSequenceDataset):  # pylint: disable=W0223
  """Grammaer Dataset using MNIST based on tf.data."""

  SYMBOLS = None
  SYMBOL_INDEX = None
  SEQUENCES = {'train': [], 'test': []}

  def _extract_symbols(self, states):
    """Extract state symbols from a State Transition Table."""

    # Create a set of unique symbol states
    state_symbols = set()
    for state in states:
      state_symbols.add(states[state]['symbol'])

    # Sort the state symbols for reproducibility
    state_symbols = sorted(state_symbols)

    # Get unique symbols from states
    self.SYMBOLS = list(state_symbols)  # pylint: disable=C0103
    self.SYMBOL_INDEX = dict((c, i) for i, c in enumerate(self.SYMBOLS))  # pylint: disable=C0103

    # Set number of classes = number of unique state symbols
    self._num_train_classes = len(self.SYMBOLS)
    self._num_test_classes = len(self.SYMBOLS)
    self._num_classes = len(self.SYMBOLS)

  def _get_sequence(self, split, transitions, fixed=False):
    """Generate a new sequence, or sample from a set."""
    if fixed:
      return random.choice(self.SEQUENCES[split])
    return self._generate_grammar(transitions)

  def _init_sequences(self, split, transitions, batch_size, fixed=False):
    """Initialise the first N (=batch_size) sequences with some offset."""
    sequences = []
    for i in range(batch_size):
      sequence = self._get_sequence(split, transitions, fixed)

      offset = i % len(sequence)
      sequence = sequence[offset:]
      sequences.append(sequence)
    return sequences

  def _dataset(self, split, images_file, labels_file, preprocess, options):  # pylint: disable=W0613, W0221
    """Generate a grammar sequence using MNIST."""

    # Extract symbols from states dictionary
    self._extract_symbols(options['grammar_states'])

    # Generate a fixed set of sequences
    if options['grammar_fixed_set']:
      #print( "fixed set <------------------------------------")
      num_sequences = options['grammar_fixed_' + split + '_size']
      setattr(self, '_' + split + '_size', num_sequences)

      self.SEQUENCES[split] = [self._generate_grammar(options['grammar_transitions']) for i in range(0, num_sequences)]

    # Batch size
    batch_size = self._batch_size

    # Get the dataset
    images, labels = self._get_images_and_labels(images_file, labels_file)

    # Initialise the sequence list with N (=batch_size) sequences
    sequences = self._init_sequences(split, options['grammar_transitions'], batch_size, options['grammar_fixed_set'])
    sequence_offsets = np.zeros(self._batch_size, dtype=np.int32)

    print('Sequence Sample:')
    print(self._states_to_symbols(options['grammar_states'], sequences[0]))

    def sequence_generator():
      """Sentence sequence generator."""

      # Loop indefinitely
      while True:
        for j in range(self._batch_size):

          i = sequence_offsets[j]

          # Try to get a sample from sequence
          try:
            sample_state = sequences[j][i]
          # Otherwise, generate a new sequence as this one has ended
          except IndexError:
            # Generate a new sequence, or randomly sample from bank
            sequence = self._get_sequence(split, options['grammar_transitions'], options['grammar_fixed_set'])

            # Append this sequence to the sequences list
            sequences[j] = sequence

            # Now try to get the sample again
            i = 0
            sample_state = sequences[j][i]

          sequence_offsets[j] = i + 1

          # Convert from state ID -> symbol -> symbol ID to match MNIST labels
          sample_symbol = options['grammar_states'][sample_state]['symbol']
          sample_idx = self.SYMBOL_INDEX[sample_symbol]

          idx, _ = self._pick_sample(labels, sample_idx, options['example_type'])
          yield (images[idx], labels[idx], sample_state)

    return tf.data.Dataset.from_generator(sequence_generator, output_types=(tf.float32, tf.int32, tf.string),
                                          output_shapes=(tf.TensorShape([self.IMAGE_DIM, self.IMAGE_DIM, 1]),
                                                         tf.TensorShape([]), tf.TensorShape([])))

  def _generate_grammar(self, state_transition_table, as_numpy=True):
    """Generates a grammar sequence (with state IDs) based on a given state transitions table."""
    out_states = []

    idx = 'START'
    while idx != -1:
      transition = state_transition_table[idx]
      path = random.choice(transition)
      idx = path['next']
      state = path['state']
      out_states.append(state)

    if as_numpy:
      return np.array(out_states)
    return out_states

  def _states_to_symbols(self, states, sequence):
    symbols = []
    for state in sequence:
      symbol = states[state]['symbol']
      symbols.append(symbol)
    return symbols

  def string_to_vector(self, s):
    """Converts a grammar sequence string to a NumPy array."""
    a = np.zeros(len(s))
    for i, c in enumerate(s):
      a[i] = self.SYMBOL_INDEX[c]
    return a

  def vector_to_string(self, xs):
    """Converts a grammar sequence NumPy array to a string."""
    string = []
    for _, value in np.ndenumerate(xs):
      char = self.SYMBOLS[int(value)]
      string.append(char)
    return ''.join(string)

  def list_to_string(self, l):
    symbols = []
    for i in l:
      symbols.append(self.SYMBOLS[i])
    return symbols
