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

"""TokenEmbeddingDataset class."""

import logging
import os.path

import numpy as np
import tensorflow as tf

from pagi.datasets.dataset import Dataset

from pagi.utils.embedding import Embedding
#from pagi.utils.embedding import SparseEmbedding, DenseEmbedding, SemanticEmbedding


class TokenEmbeddingDataset(Dataset):  # pylint: disable=W0223
  """TokenEmbeddingDataset based on tf.data."""

  def __init__(self, directory):
    super(TokenEmbeddingDataset, self).__init__(
        name='token-embedding',
        directory=directory,
        dataset_shape=[-1, 1, 1, 1],
        train_size=0,
        test_size=0,
        num_train_classes=0,
        num_test_classes=0,
        num_classes=0)

    self._embedding = None
    #self._max_length = 0
    self._subsets = {}

  def get_embedding(self):
    return self._embedding

  def get_subset(self, key):
    """Get a subset of the data."""
    if key in self._subsets:
      subset = self._subsets[key]
      return subset

    subset = {
        'size': 0,
        'corpus': None,
        'offsets': np.zeros(self._batch_size, dtype=np.int32),
        'lengths': np.zeros(self._batch_size, dtype=np.int32),
        'mask': np.zeros(self._batch_size, dtype=np.float32)  # Default zeros = clear all
    }
    self._subsets[key] = subset
    return subset

  def get_train(self, preprocess=False, options=None):  # pylint: disable=W0221
    """Returns tf.data.Dataset object """
    #init_offsets = 'striped'
    init_offsets = 'random'
    wrap_offsets = 'random'
    max_sequence_length = self._train_max_sequence_length
    return self._dataset(preprocess, options, self._embedding, 'train', init_offsets=init_offsets, wrap_offsets=wrap_offsets, max_sequence_length=max_sequence_length)

  def get_test(self, preprocess=False, options=None):  # pylint: disable=W0221
    """Returns tf.data.Dataset object """
    init_offsets = 'striped'
    wrap_offsets = 'zero'
    max_sequence_length = self._test_max_sequence_length
    return self._dataset(preprocess, options, self._embedding, 'test', init_offsets=init_offsets, wrap_offsets=wrap_offsets, max_sequence_length=max_sequence_length)

  # def is_test_state(self, subset):
  #   """Check if subset contains test state."""
  #   # A note about timing.
  #   # The current input x or label l is predicted withut using current x
  #   # Perplexity is measured every step
  #   # We are OK (for now) training every step
  #   # algo does:
  #   #   classification_loss = self._build_classification_loss(self._label_values, next_prediction)
  #   # algo predicts the CURRENT label, which means the *last* element of the sequence
  #   # if the NEXT value z=0.
  #   subset = self.get_subset(subset)
  #   sequence_lengths = subset['lengths']

  #   max_length = self._max_length
  #   if max_length == 0:
  #     return False  # All test states, so meaningless

  #   # Since theyre all synchronized, we only need to check one
  #   z = sequence_lengths[0]
  #   if z == 0:
  #     return True
  #   return False

  def get_tokens(self, text_file, token_delimiter, eos):
    """Get a list of words from the corpus."""
    return Embedding.tokenize_files([text_file], token_delimiter, eos)

  def create_embedding(self, token_file, tokens_values_file, token_delimiter):
    e = Embedding()
    e.read_tokens(token_file, token_delimiter)
    e.read_tokens_values(tokens_values_file)
    #e.check()
    return e

  def setup(self, batch_size,
            train_max_sequence_length, test_max_sequence_length,
            train_text_file, test_text_file, token_file, embedding_file,
            token_delimiter=',', eos='<end>'):
    """Setup the text embedding dataset."""

    #embedding_size = np.prod(embedding_shape[:])
    logging.info('Batch size: %s', str(batch_size))
    logging.info('Training corpus file: %s', train_text_file)
    logging.info('Testing corpus file: %s', test_text_file)
    logging.info('Embedding file: %s', embedding_file)
    logging.info('Token file: %s', token_file)
    logging.info('Token delimiter: %s', token_delimiter)
    logging.info('EOS token: %s', eos)
    logging.info('Training Max seq. len.: %s', str(train_max_sequence_length))
    logging.info('Testing Max seq. len.: %s', str(test_max_sequence_length))
    #logging.info('Random offsets: %s', str(random_offsets))

    self._eos = eos
    self._batch_size = int(batch_size)
    self._train_max_sequence_length = int(train_max_sequence_length)
    self._test_max_sequence_length = int(test_max_sequence_length)
    #self._random_offsets = random_offsets

    # Create the embedding
    self._embedding = self.create_embedding(token_file, embedding_file, token_delimiter)
    num_tokens = self._embedding.get_num_tokens()
    token_value_shape = self._embedding.get_token_value_shape()
    logging.info('Embedding has %s keys and %s values.', str(num_tokens), str(token_value_shape))

    # Override base dataset properties:
    self._num_classes = num_tokens
    self._dataset_shape = [-1, token_value_shape[0], token_value_shape[1], 1]  # [b,h,w,d]

    # Read corpus
    corpus_delimiter = ' '
    corpus_train = self.get_tokens(train_text_file, corpus_delimiter, self._eos)
    corpus_test = self.get_tokens(test_text_file, corpus_delimiter, self._eos)

    ok_train = self._embedding.has_tokens(corpus_train)
    ok_test = self._embedding.has_tokens(corpus_test)

    if ok_train and ok_test:
      logging.info('All tokens found in embedding.')
    else:
      logging.error('Some tokens missing from embedding.')

    train_size = len(corpus_train)
    test_size = len(corpus_test)

    train_subset = self.get_subset('train')
    test_subset = self.get_subset('test')

    train_subset['size'] = train_size
    test_subset['size'] = test_size

    train_subset['corpus'] = corpus_train
    test_subset['corpus'] = corpus_test

  def get_embedding_shape(self):
    token_value_shape = self._embedding.get_token_value_shape()
    embedding_shape = [token_value_shape[0], token_value_shape[1], 1]  # add depth dim
    return embedding_shape

  def _dataset(self, preprocess, options, embedding, subset_key, init_offsets, wrap_offsets, max_sequence_length=None):  # pylint: disable=W0613, W0221
    """Generate a dataset from the provided sentences & embedding."""
    #random_offsets = False
    subset = self.get_subset(subset_key)

    tokens = subset['corpus']
    sequence_offsets = subset['offsets']
    sequence_lengths = subset['lengths']
    reset_masks = subset['mask']
    num_words = subset['size']

    logging.info('Dataset subset %s has %d tokens.', subset_key, num_words)

    # Default max seq len is the corpus size; but can be made shorter
    # max_sequence_length = num_words  # Never hits this, because it wraps
    # if self._max_sequence_length > 0:
    #   max_sequence_length = self._max_sequence_length # Truncate sequences to this length
    if max_sequence_length <= 0:
      max_sequence_length = num_words  # Never hits this, because it wraps
    # else: is some value.

    # Controls for generating dataset offsets
    wrap_random_offsets = False
    init_random_offsets = False
    init_striped_offsets = False

    if wrap_offsets == 'random':
      wrap_random_offsets = True

    if init_offsets == 'random':
      init_random_offsets = True
    elif init_offsets == 'striped':
      init_striped_offsets = True

    # Initialise the sequence list with N (=batch_size) sequences
    embedding_shape = self.get_embedding_shape()

    def get_random_index(num_words):
      # Say we have max_length = 5 and num_words = 100
      # Valid indices are 0..99
      # 100-5 = 95
      # i  z1 z2
      # 95 0  1
      # 96 1  2
      # 97 2  3
      # 98 3  4
      # 99 4  5 >= 5 --> 0
      #i = np.random.randint(num_words - max_length)  # don't pick truncated seq
      i = np.random.randint(num_words)
      return i

    # init the batch of indices into the corpus
    for b in range(self._batch_size):
      i = 0  # default
      if init_random_offsets:
        i = get_random_index(num_words)
      elif init_striped_offsets:
        # Say I have 100 words and batch size 5
        # Then 100/5 = 20 ie each starts at fixed intervals
        # What if not divisible by batch size?
        # e.g. 82430 / 300 = 274
        # 299 * 274 = 81926 + 274 = 82200 i.e. 82430-82200=230 words would be untested
        stripe_size = num_words / self._batch_size
        i = b * stripe_size  # i.e. 0*274, 1*274, 2*274
      sequence_offsets[b] = i


    def sequence_generator():
      """Batch sequence generator."""

      # Loop indefinitely
      while True:
        # Generate samples for a batch
        for b in range(self._batch_size):

          i = sequence_offsets[b]  # Index (offset)
          z = sequence_lengths[b]  # Length

          token = tokens[i]  # CURRENT input

          z = z + 1  # Increment length
          i = i +1  # NEXT index.
          mask = 1.0  # keep history

          # Wrap @ fixed length smaller than the dataset
          # Or if we've watche a sequence of defined length
          if z >= max_sequence_length:
            #print('max len')
            # Get new offset
            if wrap_random_offsets:
              #print('random offsets')
              i = get_random_index(num_words)
            else:
              #print('zero offsets')
              i = 0
            z = 0  # Zero length
            mask = 0.0  # Clear history

          # Wrap @ end of corpus
          if i >= num_words:
            #print('wrap offsets')
            #z = 0 don't truncate the sequence counter
            i = 0  # Wrap to start
            mask = 0.0  # clear history

          # Useful debugging info, but very verbose:
          #logging.info('NEXT: Dataset subset: %s batch %d mask: %f offset: %s len: %d of %d max seq len %d', subset_key, b, mask, i, z, num_words, max_sequence_length)

          sequence_offsets[b] = i
          sequence_lengths[b] = z
          reset_masks[b] = mask  # need to produce a mask for NEXT sample.

          values = self._embedding.get_token_values(token)  # 2d

          # For btree embeddings:
          #values = np.minimum(values, 1.0)  # Force all 2s to 1s

          #print('Token:', token)
          #print('Embedding:', values)
          values_3d = np.reshape(values, embedding_shape)
          label = self._embedding.get_index(token)

          yield (values_3d, label)

    # Calculate size of embedding
    output_shapes = (tf.TensorShape(embedding_shape), tf.TensorShape([]))

    return tf.data.Dataset.from_generator(sequence_generator, output_types=(tf.float32, tf.int32),
                                          output_shapes=output_shapes)
