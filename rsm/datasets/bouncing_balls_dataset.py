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

"""BouncingBallsDataset class."""

import math
import logging

import numpy as np
import tensorflow as tf

from pagi.datasets.dataset import Dataset


class BouncingBallsDataset(Dataset):  # pylint: disable=W0223
  """
  A generator for the Bouncing Balls Dataset."

  'We used a dataset consisting of videos of 3 balls bouncing in abox. The videos are of length 100and of
  resolution 30×30. Each training example is synthetically generated, so notraining sequenceis seen twice by
  the model which means that overfitting is highly unlikely. The task is to learn togenerate videos at the pixel
  level. This problem is high-dimensional, having 900 dimensions perframe, and the RTRBM and the TRBM are given no
  prior knowledgeabout the nature of the task(e.g., by convolutional weight matrices).'

  Based on:
    The Recurrent Temporal Restricted Boltzmann Machine, by Sutskever et al. NIPS 2008
    https://papers.nips.cc/paper/3567-the-recurrent-temporal-restricted-boltzmann-machine
  """

  EPSILON = 0.5
  CONSTANT_R = 1.2
  CONSTANT_M = 1

  NUM_BALLS = 3
  IMAGE_DIM = 30
  NUM_FRAMES = 100
  BOUNDING_BOX_SIZE = 10

  def __init__(self, directory):
    super(BouncingBallsDataset, self).__init__(
        name='bouncing_balls',
        directory=directory,
        dataset_shape=[-1, self.IMAGE_DIM, self.IMAGE_DIM, 1],
        train_size=math.inf,
        test_size=2000,
        num_train_classes=10,
        num_test_classes=10,
        num_classes=10)

    self._batch_size = None
    self.num_frames = None

  def get_train(self, preprocess=False, options=None):  # pylint: disable=W0221
    """tf.data.Dataset object for training data."""
    del preprocess
    return self._dataset(training=True, options=options)

  def get_test(self, preprocess=False, options=None):  # pylint: disable=W0221
    """tf.data.Dataset object for test data."""
    del preprocess
    return self._dataset(training=False, options=options)

  def set_batch_size(self, batch_size):
    self._batch_size = batch_size

  def _compute_new_speeds(self, m1, m2, v1, v2):
    new_v2 = (2 * m1 * v1 + v2 * (m2 - m1)) / (m1 + m2)
    new_v1 = new_v2 + (v2 - v1)
    return new_v1, new_v2

  def _norm(self, x):
    return np.sqrt((x ** 2).sum())

  def _get_shape(self, input_array):
    if isinstance(input_array, np.ndarray):
      return np.shape(input_array)
    return input_array.shape()

  def _generate_bouncing(self, num_frames=128, num_balls=2, r=None, m=None):
    """Simulate the bouncing balls."""
    if r is None:
      r = np.array([self.CONSTANT_R] * num_balls)

    if m is None:
      m = np.array([self.CONSTANT_M] * num_balls)

    # r is to be rather small.
    generated = np.zeros((num_frames, num_balls, 2), dtype='float')
    v = np.random.randn(num_balls, 2)
    v = v / self._norm(v) * .5

    good_config = False

    while not good_config:
      x = 2 + np.random.rand(num_balls, 2) * 8
      good_config = True
      for i in range(num_balls):
        for j in range(2):
          if x[i][j] - r[i] < 0:
            good_config = False
          if x[i][j] + r[i] > self.BOUNDING_BOX_SIZE:
            good_config = False

      # that's the main part.
      for i in range(num_balls):
        for j in range(i):
          if self._norm(x[i] - x[j]) < r[i] + r[j]:
            good_config = False

    for i in range(num_frames):
      # for how long do we show small simulation
      for j in range(num_balls):
        generated[i, j] = x[j]

      for j in range(int(1 / self.EPSILON)):
        for k in range(num_balls):
          x[k] += self.EPSILON * v[k]

        for k in range(num_balls):
          for l in range(2):
            if x[k][l] - r[k] < 0:
              v[k][l] = abs(v[k][l])  # want positive

            if x[k][l] + r[k] > self.BOUNDING_BOX_SIZE:
              v[k][l] = -abs(v[k][l])  # want negative

        for k in range(num_balls):
          for l in range(k):
            if self._norm(x[k] - x[l]) < r[k] + r[l]:
              # the bouncing off part:
              w = x[k] - x[l]
              w = w / self._norm(w)

              v_i = np.dot(w.transpose(), v[k])
              v_j = np.dot(w.transpose(), v[l])

              new_v_i, new_v_j = self._compute_new_speeds(m[k], m[j], v_i, v_j)

              v[k] += w * (new_v_i - v_i)
              v[l] += w * (new_v_j - v_j)

    return generated

  def _arange(self, x, y, z):
    return z / 2 + np.arange(x, y, z, dtype='float')

  def _to_matrix(self, input_vector, image_dim, r=None):
    """Convert the generated vector into a matrix."""
    num_frames, num_balls = self._get_shape(input_vector)[0:2]

    if r is None:
      r = np.array([self.CONSTANT_R] * num_balls)

    output_matrix = np.zeros((num_frames, image_dim, image_dim), dtype='float')

    x, y = np.meshgrid(self._arange(0, 1, 1. / image_dim) * self.BOUNDING_BOX_SIZE,
                       self._arange(0, 1, 1. / image_dim) * self.BOUNDING_BOX_SIZE)

    for i in range(num_frames):
      for j in range(num_balls):
        expr = -(((x - input_vector[i, j, 0]) ** 2 + (y - input_vector[i, j, 1]) ** 2) / (r[j] ** 2)) ** 4
        output_matrix[i] += np.exp(expr)

      output_matrix[i][output_matrix[i] > 1] = 1

    return output_matrix

  def _generate_sequence(self, image_dim, num_balls=2, num_frames=128, r=None):
    """Generates a new sequence of bouncing balls."""
    if r is None:
      r = np.array([self.CONSTANT_R] * num_balls)

    generated = self._generate_bouncing(num_frames, num_balls, r)
    sequence = self._to_matrix(generated, image_dim, r)
    sequence = sequence.reshape(num_frames, image_dim, image_dim, 1)

    return sequence

  def _generate_test_set(self, test_size):
    data = []
    for _ in range(test_size):
      sequence = self._generate_sequence(self.IMAGE_DIM, self.NUM_BALLS, self.NUM_FRAMES)
      data.append(sequence)
    return np.array(data)

  def _init_sequences(self, training, batch_size):
    """Initialise the first N (=batch_size) sequences with some offset."""
    sequences = []
    states = []

    for i in range(batch_size):
      sequence = self._get_sequence(training)
      sequence_states = list(range(sequence.shape[0]))

      offset = i % len(sequence)
      sequence = sequence[offset:]
      sequence_states = sequence_states[offset:]
      sequences.append(sequence)
      states.append(sequence_states)
    return sequences, states

  def _get_sequence(self, training):
    """Sample a sequence from the training or test set."""
    if training:
      sequence = self._get_train_sequence()
    else:
      sequence = self._get_test_sequence()

    # Normalize
    sequence = 2 * sequence - 1
    sequence = sequence.astype(np.float32)

    return sequence

  def _get_train_sequence(self):
    """Generates a sequence with Moving MNIST digits."""
    return self._generate_sequence(self.IMAGE_DIM, self.NUM_BALLS, self.NUM_FRAMES)

  def _get_test_sequence(self):
    sequence = self._frames[self._next_idx]
    self._next_idx = (self._next_idx + 1) % self._frames.shape[0]
    return sequence

  def _dataset(self, training, options=None):
    """Download and parse the dataset."""

    self.num_frames = self.NUM_FRAMES

    if not training:
      self._frames = self._generate_test_set(200)
      self._next_idx = 0

    # Initialise sequences
    batch_size = self._batch_size
    sequences, states = self._init_sequences(training, batch_size)
    sequence_offsets = np.zeros(batch_size, dtype=np.int32)

    # Compute (potentially) padded image dimensions
    image_dim = self.IMAGE_DIM + (options['frame_padding_size'] * 2)
    self._dataset_shape = [-1, image_dim, image_dim, 1]

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
            # Generate a new sequence, or randomly sample from bank
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

          end_state = False
          if state == (sequences[b].shape[0] - 1):
            end_state = True

          yield (frame, label, end_state)

    def preprocess(image, label, state):
      padding_size = options['frame_padding_size']

      if padding_size > 0:
        pad_h = [padding_size] * 2
        pad_w = [padding_size] * 2

        paddings = [pad_h, pad_w, [0, 0]]
        image = tf.pad(image, paddings,
                       constant_values=options['frame_padding_value'])

      return image, label, state

    dataset = tf.data.Dataset.from_generator(generator, output_types=(tf.float32, tf.int32, tf.bool),
                                             output_shapes=(
                                                 tf.TensorShape([self.IMAGE_DIM, self.IMAGE_DIM, 1]),
                                                 tf.TensorShape([]),
                                                 tf.TensorShape([])))
    dataset = dataset.map(preprocess)

    return dataset
