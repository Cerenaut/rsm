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

"""MNISTMovingInfiniteDataset class."""

import os
import math
import gzip
import logging

from urllib.request import urlretrieve
from PIL import Image

import numpy as np
import tensorflow as tf

from rsm.datasets.mnist_moving_dataset import MNISTMovingDataset


class MNISTMovingInfiniteDataset(MNISTMovingDataset):  # pylint: disable=W0223
  """A generator for the Moving MNIST Dataset."""

  MNIST_DIM = 28

  # Big
  # DIGIT_DIM = 28
  # IMAGE_DIM = 64
  # SPEED_RNG = 5
  # SPEED_MIN = 2

  # Small
  DIGIT_DIM = 14
  IMAGE_DIM = 32
  SPEED_RNG = 0
  SPEED_MIN = 2

  def get_train(self, preprocess=False, options=None):  # pylint: disable=W0221
    """tf.data.Dataset object for MNIST training data."""
    del preprocess
    return self._dataset(training=True, options=options)

  def get_test(self, preprocess=False, options=None):  # pylint: disable=W0221
    """tf.data.Dataset object for MNIST test data."""
    del preprocess
    return self._dataset(training=False, options=options)

  def set_batch_size(self, batch_size):
    self._batch_size = batch_size

  def _arr_from_img(self, im, mean=0, std=1):
    """
    Args:
        im: Image
        shift: Mean to subtract
        std: Standard Deviation to subtract
    Returns:
        Image in np.float32 format, in width height channel format. With values in range 0,1
        Shift means subtract by certain value. Could be used for mean subtraction.
    """
    width, height = im.size
    arr = im.getdata()
    c = int(np.product(arr.size) / (width * height))

    return (np.asarray(arr, dtype=np.float32).reshape((width, height, c)) / 255. - mean) / std

  def _get_image_from_array(self, image, index, mean=0, std=1):
    """
    Args:
        X: Dataset of shape N x C x W x H
        index: Index of image we want to fetch
        mean: Mean to add
        std: Standard Deviation to add
    Returns:
        Image with dimensions H x W x C or H x W if it's a single channel image
    """
    w, h, ch = image.shape[1], image.shape[2], image.shape[3]
    ret = (((image[index] + mean) * 255.) * std).reshape(w, h, ch).clip(0, 255).astype(np.uint8)
    if ch == 1:
      ret = ret.reshape(h, w)
    return ret

  def _load_mnist_dataset(self, training=True, options=None):
    """Download the MNIST dataset and load it into a NumPy array."""
    if training:
      filename = 'train-images-idx3-ubyte.gz'
      labels_filename = 'train-labels-idx1-ubyte.gz'
    else:
      filename = 't10k-images-idx3-ubyte.gz'
      labels_filename = 't10k-labels-idx1-ubyte.gz'

    dirpath = os.path.join(self._directory, self._name)
    filepath = os.path.join(dirpath, filename)
    labels_filepath = os.path.join(dirpath, labels_filename)
    source = 'http://yann.lecun.com/exdb/mnist/'

    if not tf.gfile.Exists(dirpath):
      tf.gfile.MakeDirs(dirpath)

    if not os.path.exists(filepath):
      urlretrieve(source + filename, filepath)

    if not os.path.exists(labels_filepath):
      urlretrieve(source + labels_filename, labels_filepath)

    with gzip.open(filepath, 'rb') as f:
      data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, self.MNIST_DIM, self.MNIST_DIM, 1)

    with gzip.open(labels_filepath, 'rb') as f:
      labels = np.frombuffer(f.read(), np.uint8, offset=8)

    if options is not None and options['allowed_classes']:
      print('Allowed Classes =', options['allowed_classes'])
      allowed_idxs = np.isin(labels, options['allowed_classes'])

      data = data[allowed_idxs]
      labels = labels[allowed_idxs]

    assert data.shape[0] == labels.shape[0]

    return data / np.float32(255)

  def _load_moving_mnist_dataset(self):
    """Load the Moving MNIST test set."""
    filename = 'mnist_test_seq.npy'
    dirpath = os.path.join(self._directory, self._name)
    filepath = os.path.join(dirpath, filename)
    source = 'http://www.cs.toronto.edu/~nitish/unsupervised_video/'

    if not tf.gfile.Exists(dirpath):
      tf.gfile.MakeDirs(dirpath)

    if not os.path.exists(filepath):
      urlretrieve(source + filename, filepath)

    data = np.load(filepath)
    data = np.transpose(data, (1, 0, 2, 3))
    data = np.expand_dims(data, axis=4)

    return data

  def _init_sequences(self, training, batch_size):  # pylint: disable=arguments-differ
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

  def _get_sequence(self, training):  # pylint: disable=arguments-differ
    """Sample a sequence from the training or test set."""
    if training or self.IMAGE_DIM != 64:
      sequence = self._get_train_sequence()
    else:
      sequence = self._get_test_sequence()

    # Normalize
    sequence = sequence / 255
    sequence = sequence.astype(np.float32)

    return sequence

  def _get_train_sequence(self):
    """Generates a sequence with Moving MNIST digits."""
    num_frames = self.num_frames
    original_size = self.DIGIT_DIM
    nums_per_image = self.digits_per_image
    width, height = (self.IMAGE_DIM, self.IMAGE_DIM)

    # Get how many pixels can we move around a single image
    lims = (x_lim, y_lim) = width - original_size, height - original_size

    # Create a sequence of shape (num_frames, new_width, new_height)
    # Example: (20, 64, 64, 1)
    sequence = np.empty((num_frames, width, height, 1), dtype=np.uint8)

    # Randomly generate direction, speed and velocity for both images
    direcs = np.pi * (np.random.rand(nums_per_image) * 2 - 1)
    if self.SPEED_RNG == 0:
      speeds = np.ones(nums_per_image) * self.SPEED_MIN
    else:
      speeds = np.random.randint(self.SPEED_RNG, size=nums_per_image) + self.SPEED_MIN
    veloc = np.asarray([(speed * math.cos(direc), speed * math.sin(direc)) for direc, speed in zip(direcs, speeds)])

    # Get a list containing two PIL images randomly sampled from the database
    def get_image(idx):
      image = Image.fromarray(self._get_image_from_array(self._images, idx, mean=0))
      image = image.resize((original_size, original_size), Image.ANTIALIAS)
      return image

    mnist_images = [get_image(r) for r in np.random.randint(0, self._images.shape[0], nums_per_image)]

    # Generate tuples of (x,y) i.e initial positions for nums_per_image (default : 2)
    positions = np.asarray([(np.random.rand() * x_lim, np.random.rand() * y_lim) for _ in range(nums_per_image)])

    # Generate new frames for the entire num_framesgth
    for frame_idx in range(num_frames):
      canvases = [Image.new('L', (width, height)) for _ in range(nums_per_image)]
      canvas = np.zeros((width, height, 1), dtype=np.float32)

      # In canv (i.e Image object) place the image at the respective positions
      # Super impose both images on the canvas (i.e empty np array)
      for i, canv in enumerate(canvases):
        canv.paste(mnist_images[i], tuple(positions[i].astype(int)))
        canvas += self._arr_from_img(canv, mean=0)

      # Get the next position by adding velocity
      next_pos = positions + veloc

      # Iterate over velocity and see if we hit the wall
      # If we do then change the  (change direction)
      for i, pos in enumerate(next_pos):
        for j, coord in enumerate(pos):
          if coord < -2 or coord > lims[j] + 2:
            veloc[i] = list(list(veloc[i][:j]) + [-1 * veloc[i][j]] + list(veloc[i][j + 1:]))

      # Make the permanent change to position by adding updated velocity
      positions = positions + veloc

      # Add the canvas to the dataset array
      sequence[frame_idx] = (canvas * 255).clip(0, 255).astype(np.uint8)

    return sequence

  def _get_test_sequence(self):
    sequence = self._frames[self._next_idx]
    self._next_idx = (self._next_idx + 1) % self._frames.shape[0]
    return sequence

  def _dataset(self, training, options=None): # pylint: disable=arguments-differ
    """Download and parse the dataset."""

    self.num_frames = 20
    self.digits_per_image = 2

    # Load MNIST images into memory
    if training:
      self._images = self._load_mnist_dataset(training, options)
    else:
      if self.IMAGE_DIM == 64:
        self._frames = self._load_moving_mnist_dataset()
        self._next_idx = 0
      else:
        self._images = self._load_mnist_dataset(training, options)

    # Initialise sequences
    batch_size = self._batch_size
    sequences, states = self._init_sequences(training, batch_size)
    sequence_offsets = np.zeros(batch_size, dtype=np.int32)

    # Compute (potentially) padded image dimensions
    image_dim = image_dim = self.IMAGE_DIM
    if options['frame_padding_size'] > 0:
      image_dim += (options['frame_padding_size'] * 2)
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

          seq_length = sequences[b].shape[0]
          end_state = False
          if i == (seq_length -1):
            end_state = True
          # end_state = False
          # if new_sequence is True:
          #   end_state = True
          # if state == (sequences[b].shape[0] - 1):
          #   end_state = True
          # if state == 0:
          #   end_state = True

          yield (frame, label, state, end_state)

    def preprocess(image, label, state, end_state):
      padding_size = options['frame_padding_size']

      image_shape = image.get_shape().as_list()

      # Given padding_size = -1
      if padding_size < 0:
        padding_size = abs(padding_size)  # padding_size = 1
        inset = int(padding_size * 2)  # inset = 2

        # If => image shape = (32, 32)
        # Then => cropped image shape = (32 - 2, 32 - 2 ) = (30, 30)
        image = tf.image.resize_image_with_crop_or_pad(image,
                                                       target_height=image_shape[0] - inset,
                                                       target_width=image_shape[1] - inset)

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
