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

"""MNISTSequenceDataset class."""

import os
import re
import logging
import random
import struct

import numpy as np
import tensorflow as tf

from pagi.datasets.mnist_dataset import MNISTDataset


class MNISTSequenceDataset(MNISTDataset):  # pylint: disable=W0223
  """Sequence generator for the MNIST Dataset."""

  def set_batch_size(self, batch_size):
    self._batch_size = batch_size

  def get_train(self, preprocess=False, options=None):  # pylint: disable=W0221
    """tf.data.Dataset object for MNIST training data."""
    return self._dataset('train', 'train-images-idx3-ubyte', 'train-labels-idx1-ubyte', preprocess, options)

  def get_test(self, preprocess=False, options=None):  # pylint: disable=W0221
    """tf.data.Dataset object for MNIST test data."""
    return self._dataset('test', 't10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte', preprocess, options)

  def generate_random_sequence(self, num, k):
    return [random.choice(range(num)) for i in range(k)]

  def generate_sequence_from_text(self, text):
    text = text.replace(' ', '')
    return [int(d) for d in str(text)]

  def generate_sequence_from_file(self, filename):
    with open(filename, 'r') as f:
      # Filter non-numeric characters
      filtered_content = re.sub('[^0-9]', '', f.read())
      return self.generate_sequence_from_text(filtered_content)

  def _get_images_and_labels(self, images_file, labels_file):
    """Download and parse MNIST dataset."""
    # Download the dataset
    images_file = self._download(self._directory, images_file)
    labels_file = self._download(self._directory, labels_file)

    # Verify headers
    self._check_image_file_header(images_file)
    self._check_labels_file_header(labels_file)

    # Read and process the images
    images = self._read_file(images_file)
    images = images / 255.0
    images = np.reshape(images, [-1, self.IMAGE_DIM, self.IMAGE_DIM, 1])
    images = images.astype(np.float32)

    # Read and process the labels
    labels = self._read_file(labels_file)
    labels = labels.astype(np.int32)

    return images, labels

  def _pick_sample(self, labels, label, example_type='random'):
    """Picks a sample for specified label."""
    indices = np.where(labels == label)[0]

    # Pick same index everytime
    if example_type == 'same':
      idx = indices[0]

    # Pick specified index for this label
    elif example_type == 'specific':
      try:
        idx = indices[options['specific_examples'][label]]
      except e:
        print(e)
        logging.error('Failed to find an example associated with the label.')

    # Fallback to random index
    else:
      idx = np.random.choice(indices)

    return idx, indices

  def _get_sequence(self, sequence=None):
    if sequence == 'random' or None:
      return self.generate_random_sequence(self.num_classes, options['sequence_length'])
    return sequence

  def _init_sequences(self, batch_size, input_sequence=None):
    """Initialise the first N (=batch_size) sequences with some offset."""
    sequences = []
    for i in range(batch_size):
      sequence = self._get_sequence(input_sequence)

      offset = i % len(sequence)
      sequence = sequence[offset:]
      sequences.append(sequence)
    return sequences

  def _dataset(self, split, images_file, labels_file, preprocess, options):  # pylint: disable=W0613, W0221
    """Download and parse MNIST dataset."""

    # Batch size
    batch_size = self._batch_size
    logging.info('Batch size: %s', str(batch_size))

    # Capture sequence from a file
    input_sequence = None
    if 'sequence' in options:
      if os.path.isfile(options['sequence']):
        input_sequence = self.generate_sequence_from_file(options['sequence'])
      # Capture sequence from a given string
      elif isinstance(options['sequence'], str) and options['sequence'] != 'random':
        input_sequence = self.generate_sequence_from_text(options['sequence'])
      # Generate a random sequence
      elif options['sequence'] == 'random' and 'sequence_length' in options:
        input_sequence = 'random'

    logging.info('Sequence used: %s', str(input_sequence))

    # Get the dataset
    images, labels = self._get_images_and_labels(images_file, labels_file)

    # Initialise the sequence list with N (=batch_size) sequences
    sequences = self._init_sequences(batch_size, input_sequence)

    sequence_offsets = np.zeros(self._batch_size, dtype=np.int32)

    def sequence_generator():
      """Generates image and label pairs based on a given sequence of labels."""
      logging.debug('Batch size [generator]: %s', str(batch_size))

      # Loop indefinitely
      while True:
        for b in range(self._batch_size):

          i = sequence_offsets[b]

          # Try to get a sample from sequence
          try:
            sample_idx = sequences[b][i]
          # Otherwise, generate a new sequence as this one has ended
          except IndexError:
            # Generate a new sequence, or randomly sample from bank
            sequence = self._get_sequence(input_sequence)

            # Append this sequence to the sequences list
            sequences[b] = sequence

            # Now try to get the sample again
            i = 0
            sample_idx = sequences[b][i]

          #print('Batch:', b, ' Seq-idx: ', i, 'Sample: ', sample_idx)

          sequence_offsets[b] = i + 1

          idx, _ = self._pick_sample(labels, sample_idx, options['example_type'])
          yield (images[idx], labels[idx])

    return tf.data.Dataset.from_generator(sequence_generator, output_types=(tf.float32, tf.int32),
                                          output_shapes=(tf.TensorShape([self.IMAGE_DIM, self.IMAGE_DIM, 1]),
                                                         tf.TensorShape([])))

  def _read_file(self, filename):
    with open(filename, 'rb') as f:
      _, _, dims = struct.unpack('>HBB', f.read(4))
      shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
      return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)
