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

"""RAVDESSDataset class."""

import os
import logging
import urllib

import h5py
import numpy as np
import tensorflow as tf

from pagi.datasets.dataset import Dataset
from pagi.utils.data_utils import generate_filenames


class RAVDESSDataset(Dataset):  # pylint: disable=W0223
  """A generator for the RAVDESSDataset Dataset."""

  IMAGE_DIM = 87

  NUM_FILES = 4

  def __init__(self, directory):
    super(RAVDESSDataset, self).__init__(
        name='ravdess',
        directory=directory,
        dataset_shape=[-1, 87, 87, 3],
        train_size=4904,
        test_size=4904,
        num_train_classes=10,
        num_test_classes=10,
        num_classes=10)

    self._batch_size = None
    self.num_frames = 100

  def get_train(self, preprocess=False, options=None):  # pylint: disable=W0221
    """tf.data.Dataset object for training data."""
    del options
    return self._dataset('train', self._directory, 'sharded_ravdess.h5', preprocess)

  def get_test(self, preprocess=False, options=None):  # pylint: disable=W0221
    """tf.data.Dataset object for test data."""
    del options
    return self._dataset('test', self._directory, 'sharded_ravdess.h5', preprocess)

  def set_batch_size(self, batch_size):
    self._batch_size = batch_size

  def _download(self, directory, filename):
    """Download the preprocessed RAVDESS dataset from Cloud Storage."""
    url = 'https://storage.googleapis.com/project-agi/datasets/ravdess/'

    dirpath = os.path.join(directory, self.name)


    if not tf.gfile.Exists(dirpath):
      tf.gfile.MakeDirs(dirpath)


    for i in range(self.NUM_FILES):
      temp_filename = filename + '-' + str(i)

      urlpath = url + temp_filename
      filepath = os.path.join(dirpath, temp_filename)

      if tf.gfile.Exists(filepath):
        continue

      logging.info('Downloading %s to %s', urlpath, filepath)
      urllib.request.urlretrieve(urlpath, filepath)

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
    if training:
      sequence = self._videos[self._next_train_idx][:]
      self._next_train_idx = (self._next_train_idx + 1) % len(self._videos)
      return sequence

    sequence = self._videos[self._next_test_idx][:]
    self._next_test_idx = (self._next_test_idx + 1) % len(self._videos)
    return sequence

  def _load_dataset(self, directory, filename):
    """Load and keep track of the RAVDESS videos."""
    self._download(directory, filename)

    videos = []
    identifiers = []
    filenames = generate_filenames(self.name, directory, filename)

    for filepath in filenames:
      hf = h5py.File(filepath, 'r')

      for dataset_name in hf['data']:
        dataset = hf['data'][dataset_name]

        if dataset_name.startswith('video'):
          videos.append(dataset)
        elif dataset_name.startswith('identifiers'):
          identifiers.append(dataset)

    return videos, identifiers

  def _dataset(self, split, directory, filename, preprocess):
    """Download and parse the dataset."""
    del preprocess

    training = True
    if split == 'test':
      training = False

    self._videos, _ = self._load_dataset(directory, filename)
    self._next_train_idx = 0
    self._next_test_idx = 0

    # Initialise sequences
    batch_size = self._batch_size
    sequences, states = self._init_sequences(training, batch_size)
    sequence_offsets = np.zeros(batch_size, dtype=np.int32)

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

    dataset = tf.data.Dataset.from_generator(generator, output_types=(tf.float32, tf.int32, tf.bool),
                                             output_shapes=(
                                                 tf.TensorShape([self.IMAGE_DIM, self.IMAGE_DIM, 3]),
                                                 tf.TensorShape([]),
                                                 tf.TensorShape([])))

    return dataset
