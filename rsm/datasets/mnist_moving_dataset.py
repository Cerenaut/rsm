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

"""MNISTMovingDataset class."""

import os
import logging
import random

import numpy as np
import tensorflow as tf

from pagi.datasets.dataset import Dataset


class MNISTMovingDataset(Dataset):  # pylint: disable=W0223
  """A generator for the Moving MNIST Dataset."""

  IMAGE_DIM = 64

  def __init__(self, directory):
    super(MNISTMovingDataset, self).__init__(
        name='mnist_moving',
        directory=directory,
        dataset_shape=[-1, 64, 64, 1],
        train_size=60000,
        test_size=10000,
        num_train_classes=10,
        num_test_classes=10,
        num_classes=10)

    self._batch_size = None
    self.num_frames = None

  def get_train(self, preprocess=False, options=None):  # pylint: disable=W0221
    """tf.data.Dataset object for MNIST training data."""
    del options
    return self._dataset(self._directory, 'train-120k.npz', preprocess)

  def get_test(self, preprocess=False, options=None):  # pylint: disable=W0221
    """tf.data.Dataset object for MNIST test data."""
    del options
    return self._dataset(self._directory, 'test-10k.npz', preprocess)

  def set_batch_size(self, batch_size):
    self._batch_size = batch_size

  def _init_sequences(self, training, batch_size):
    """Initialise the first N (=batch_size) sequences with some offset."""
    sequences = []
    states = []

    for i in range(batch_size):
      sequence = self._get_sequence(training)
      sequence_states = list(range(sequence.shape[0]))

      offset = 0
      if training:
        offset = i % len(sequence)

      sequence = sequence[offset:]
      sequence_states = sequence_states[offset:]
      sequences.append(sequence)
      states.append(sequence_states)
    return sequences, states

  def _get_sequence(self, frames):
    idx = random.choice(range(0, frames.shape[0]))
    return frames[idx]

  def _dataset(self, directory, filename, preprocess):
    """Download and parse the dataset."""
    del preprocess

    dirpath = os.path.join(directory, self.name)
    filepath = os.path.join(dirpath, filename)

    with np.load(filepath) as data:
      frames = data['arr_0']

    batch_size = self._batch_size
    sequences, states = self._init_sequences(batch_size, frames)
    sequence_offsets = np.zeros(batch_size, dtype=np.int32)

    self.num_frames = frames.shape[1]

    def generator():
      """Generate frames from the dataset."""
      logging.debug('Batch size [generator]: %s', str(batch_size))

      # Loop indefinitely
      while True:
        for b in range(self._batch_size):

          i = sequence_offsets[b]
          label = 0

          # Try to get a sample from sequence
          try:
            frame = sequences[b][i]
            state = states[b][i]
          # Otherwise, generate a new sequence as this one has ended
          except IndexError:
            sequence = self._get_sequence(training)
            sequence_states = list(range(sequence.shape[0]))

            # Append this sequence to the sequences list
            sequences[b] = sequence
            states[b] = sequence_states

            # Now try to get the sample again
            i = 0
            frame = sequences[b][i]
            state = states[b][i]

          sequence_offsets[b] = i + 1

          # Normalize to [0, 1.0]
          frame = frame / 255.0

          end_state = False
          if state == (sequences[b].shape[0] - 1):
            end_state = True

          yield (frame, label, state, end_state)

    def preprocess(image, label, state, end_state):
      padding_size = options['frame_padding_size']

      if padding_size > 0:
        pad_h = [padding_size] * 2
        pad_w = [padding_size] * 2

        paddings = [pad_h, pad_w, [0, 0]]
        image = tf.pad(image, paddings,
                       constant_values=options['frame_padding_value'])

      return image, label, state, end_state

    dataset = tf.data.Dataset.from_generator(generator, output_types=(tf.float32, tf.int32, tf.int32, tf.bool),
                                             output_shapes=(
                                                 tf.TensorShape([self.IMAGE_DIM, self.IMAGE_DIM, 1]),
                                                 tf.TensorShape([]),
                                                 tf.TensorShape([]),
                                                 tf.TensorShape([])))
    dataset = dataset.map(preprocess)

    return dataset
