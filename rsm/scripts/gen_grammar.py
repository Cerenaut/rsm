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

"""Generates a grammar for producing complex MNIST sequences."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ast
import json
import random

from itertools import tee, combinations
from collections import OrderedDict
from pprint import pprint

import numpy as np
import tensorflow as tf

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('subsequences', None, 'Pre-defined list of subsequences.')

tf.flags.DEFINE_integer('start_digits', 2, 'Number of start digits in each subsequence.')
tf.flags.DEFINE_integer('num_digits', 8, 'Number of remaining digits in the subsequence.')
tf.flags.DEFINE_integer('num_subsequences', 2, 'Final image size.')
tf.flags.DEFINE_integer('seed', 42, 'Seed used to control randomness for reproducability.')

tf.flags.DEFINE_boolean('export', True, 'Export the grammar structure to JSON file.')

MNIST_DIGITS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def main(_):
  random.seed(FLAGS.seed)
  np.random.seed(FLAGS.seed)
  tf.set_random_seed(FLAGS.seed)

  if FLAGS.subsequences:
    subsequences = ast.literal_eval(FLAGS.subsequences)
  else:
    print('\nGrammar Options\n-------------------')
    print('Number of subsequences:', FLAGS.num_subsequences)
    print('Number of digits in each subsequence:', FLAGS.num_digits + FLAGS.start_digits)
    print('Number of start digits:', FLAGS.start_digits)
    print('Number of remaining digits:', FLAGS.num_digits)
    print('\n')

    # Find unique combinations for the starting digits
    start_digits = list(combinations(MNIST_DIGITS, FLAGS.start_digits))
    random.shuffle(start_digits)

    if len(start_digits) < FLAGS.num_subsequences:
      raise ValueError('Number of subsequences cannot be greater than the number of unique start digit combinations.')

    start_digits = start_digits[:FLAGS.num_subsequences]

    print('Start Digits:\n-------------------')
    print(start_digits)
    print('\n')

    # Generate subsequences
    subsequences = []
    for i, start in enumerate(start_digits):

      while True:
        subsequence = [random.choice(MNIST_DIGITS) for i in range(FLAGS.num_digits)]
        subsequence_chunks = [tuple(subsequence[i:i + FLAGS.start_digits]) \
                              for i in range(len(subsequence) - (FLAGS.start_digits - 1))]

        if start not in subsequence_chunks:
          break

      subsequence = list(start) + subsequence
      subsequences.append(subsequence)

  # Generate grammar states
  states = {}
  state_symbols = set(x for l in subsequences for x in l)
  symbol_to_state = {}

  for i, symbol in enumerate(state_symbols):
    state_id = 'S' + str(i)
    states[state_id] = {
        'symbol': str(symbol),
        'test': False,
        'end': False
    }

    symbol_to_state[symbol] = state_id

  # Generate grammar state transitions
  transitions = OrderedDict()
  subsequences_transposed = list(zip(*subsequences))

  print('Subsequences:\n-------------------')
  pprint(subsequences)
  print('\n')

  if FLAGS.export:
    next_ids = ['START' for i in range(len(subsequences))]
    for i, sample in enumerate(subsequences_transposed):
      for j, symbol in enumerate(sample):
        if (i + 1) >= len(subsequences_transposed):
          next_id = -1  # Reached the end of the subsequence
        else:
          next_id = 'SS' + str(j) + '_' + str(i + 1)

        transition = {
          'state': symbol_to_state[symbol],
          'next': next_id
        }

        if next_ids[j] in transitions:
          transitions[next_ids[j]].append(transition)
        else:
          transitions[next_ids[j]] = [transition]

        next_ids[j] = transition['next']

    grammar = {
      'grammar_states': states,
      'grammar_transitions': transitions
    }

    filename = 'grammar.local.json'

    with open(filename, 'w') as fp:
      json.dump(grammar, fp, indent=2)

    print('Grammar saved to', filename)

if __name__ == '__main__':
  tf.app.run()
