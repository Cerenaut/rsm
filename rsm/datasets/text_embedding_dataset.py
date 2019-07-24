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

"""TextEmbeddingDataset class."""

import logging
import os.path

import numpy as np
import tensorflow as tf

from pagi.datasets.dataset import Dataset

from pagi.utils.embedding import SparseEmbedding, DenseEmbedding, SemanticEmbedding


class TextEmbeddingDataset(Dataset):  # pylint: disable=W0223
  """TextEmbeddingDataset based on tf.data."""

  def __init__(self, directory):
    super(TextEmbeddingDataset, self).__init__(
        name='text-embedding',
        directory=directory,
        dataset_shape=[-1, 1, 1, 1],
        train_size=0,
        test_size=0,
        num_train_classes=0,
        num_test_classes=0,
        num_classes=0)

    self._use_sparse_embedding = True
    self._embedding = None
    self._embedding_shape = None

    self._max_length = 0
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
    return self._dataset(preprocess, options, self._embedding, 'train')

  def get_test(self, preprocess=False, options=None):  # pylint: disable=W0221
    """Returns tf.data.Dataset object """
    return self._dataset(preprocess, options, self._embedding, 'test', random_offsets=False)

  def get_words(self, embedding, text_file, eos):
    """Get a list of words from the corpus."""
    del eos

    sentences = embedding.read_corpus_files([text_file])

    corpus = []
    for sentence in sentences:
      for word in sentence:
        corpus.append(word)

    return corpus

  def is_test_state(self, subset):
    """Check if subset contains test state."""
    # A note about timing.
    # The current input x or label l is predicted withut using current x
    # Perplexity is measured every step
    # We are OK (for now) training every step
    # algo does:
    #   classification_loss = self._build_classification_loss(self._label_values, next_prediction)
    # algo predicts the CURRENT label, which means the *last* element of the sequence
    # if the NEXT value z=0.
    subset = self.get_subset(subset)
    sequence_lengths = subset['lengths']

    max_length = self._max_length
    if max_length == 0:
      return False  # All test states, so meaningless

    # Since theyre all synchronized, we only need to check one
    z = sequence_lengths[0]
    if z == 0:
      return True
    return False

  def _create_embedding(self, train_text_file, test_text_file, embedding_file, embedding_shape,
                        embedding_sparsity, eos):
    if self._embedding_type == 'sparse':
      return self._create_sparse_embedding(train_text_file, test_text_file, embedding_file, embedding_shape,
                                           embedding_sparsity, eos)
    if self._embedding_type == 'dense':
      return self._create_dense_embedding(train_text_file, test_text_file, embedding_file, embedding_shape,
                                          embedding_sparsity, eos)
    return self._create_semantic_embedding(train_text_file, test_text_file, embedding_file, embedding_shape,
                                           embedding_sparsity, eos)

  def _create_sparse_embedding(self, train_text_file, test_text_file, embedding_file, embedding_shape,
                               embedding_sparsity, eos):
    """Create a sparse embedding."""
    embedding = SparseEmbedding()
    if not os.path.isfile(embedding_file):
      logging.info('Creating sparse embedding...')
      embedding.create([train_text_file, test_text_file], embedding_file, embedding_shape, embedding_sparsity, eos)

    logging.info('Reading embedding...')
    embedding.read(embedding_file)

    # embedding.check()

    return embedding, embedding_shape

  def _create_dense_embedding(self, train_text_file, test_text_file, embedding_file, embedding_shape,
                              embedding_sparsity, eos):
    """Create a dense embedding."""
    embedding = DenseEmbedding()
    corpus_files = [train_text_file, test_text_file]
    if not os.path.isfile(embedding_file):
      logging.info('Creating dense embedding...')
      embedding_shape = embedding.create(corpus_files, embedding_file, embedding_shape, embedding_sparsity, eos)
    else:
      embedding_shape = embedding.create_shape(corpus_files, eos)

    logging.info('Reading embedding...')
    embedding.read(embedding_file)

    return embedding, embedding_shape

  def _create_semantic_embedding(self, train_text_file, test_text_file, embedding_file, embedding_shape,
                                 embedding_sparsity, eos):
    """Create a semantic embedding."""
    embedding = SemanticEmbedding()
    if not os.path.isfile(embedding_file):
      logging.info('Creating semantic embedding...')
      embedding.create([train_text_file, test_text_file], embedding_file, embedding_shape, embedding_sparsity, eos)

    logging.info('Reading embedding...')
    embedding.read(embedding_file)
    return embedding, embedding_shape

  def setup(self, batch_size, train_text_file, test_text_file, embedding_type, embedding_file, embedding_shape,
            embedding_sparsity, max_sequence_length, eos='<end>'):
    """Setup the text embedding dataset."""

    embedding_size = np.prod(embedding_shape[:])
    logging.info('Batch size: %s', str(batch_size))
    logging.info('Training corpus file: %s', train_text_file)
    logging.info('Testing corpus file: %s', test_text_file)
    logging.info('Embedding type: %s', embedding_type)
    logging.info('Embedding file: %s', embedding_file)
    logging.info('Embedding size: %s', str(embedding_size))
    logging.info('Max seq. len.: %s', str(max_sequence_length))

    self._eos = eos
    self._embedding_type = embedding_type
    self._batch_size = int(batch_size)
    self._max_length = int(max_sequence_length)

    self._embedding, self._embedding_shape = self._create_embedding(train_text_file, test_text_file, embedding_file,
                                                                    embedding_shape, embedding_sparsity, eos)

    emb_keys = self._embedding.get_num_keys()
    emb_vals = self._embedding.get_num_values()

    logging.info('Embedding has %s keys and %s values.', str(emb_keys), str(emb_vals))

    corpus_train = self.get_words(self._embedding, train_text_file, self._eos)
    corpus_test = self.get_words(self._embedding, test_text_file, self._eos)

    ok_train = self._embedding.has_keys(corpus_train)
    ok_test = self._embedding.has_keys(corpus_test)

    if ok_train and ok_test:
      logging.info('All tokens found in embedding.')
    else:
      logging.error('Some tokens missing from embedding.')

    # Override base dataset properties:
    self._dataset_shape = [-1, self._embedding_shape[0], self._embedding_shape[1], 1]
    self._num_classes = emb_keys

    train_size = len(corpus_train)
    test_size = len(corpus_test)

    train_subset = self.get_subset('train')
    test_subset = self.get_subset('test')

    train_subset['size'] = train_size
    test_subset['size'] = test_size

    train_subset['corpus'] = corpus_train
    test_subset['corpus'] = corpus_test

  def get_embedding_shape(self):
    return self._embedding_shape

  def _dataset(self, preprocess, options, embedding, subset_key, random_offsets=True):  # pylint: disable=W0613, W0221
    """Generate a dataset from the provided sentences & embedding."""

    max_length = self._max_length
    subset = self.get_subset(subset_key)

    words = subset['corpus']
    sequence_offsets = subset['offsets']
    sequence_lengths = subset['lengths']
    reset_masks = subset['mask']
    num_words = subset['size']

    logging.info('Dataset subset %s has %d tokens.', subset_key, num_words)

    # Default max seq len is the corpus size; but can be made shorter
    max_seq_length = num_words
    if max_length > 0:
      max_seq_length = max_length # Truncate sequences to this length

    # Initialise the sequence list with N (=batch_size) sequences
    embedding_shape = [self._embedding_shape[0], self._embedding_shape[1], 1]

    def get_random_index(num_words, max_length):
      # Say we have max_length = 5 and num_words = 100
      # Valid indices are 0..99
      # 100-5 = 95
      # i  z1 z2
      # 95 0  1
      # 96 1  2
      # 97 2  3
      # 98 3  4
      # 99 4  5 >= 5 --> 0
      i = np.random.randint(num_words - max_length)  # don't pick truncated seq
      return i

    # init the batch of indices into the corpus
    for b in range(self._batch_size):
      if random_offsets:
        i = get_random_index(num_words, max_length)
        sequence_offsets[b] = i

    def sequence_generator():
      """Batch sequence generator."""

      # Loop indefinitely
      while True:
        for b in range(self._batch_size):

          i = sequence_offsets[b]
          z = sequence_lengths[b]

          key = words[i]  # CURRENT input

          z = z + 1  # Increase length
          i = i +1  # NEXT index.
          mask = 1.0  # keep

          # Wrap @ fixed length smaller than the dataset
          # Or if we've watche a sequence of defined length
          if z >= max_seq_length:
            z = 0
            if random_offsets:
              i = get_random_index(num_words, max_length)
            else:
              i = 0
            mask = 0.0  # clear

          # Wrap @ end of corpus
          if i >= num_words:
            #z = 0 don't truncate the sequence counter
            i = 0  # Wrap to start
            mask = 0.0  # clear

          logging.debug('Dataset subset: %s batch %d mask: %f offset: %s len: %d of %d',
                        subset_key, b, mask, i, z, num_words)

          sequence_offsets[b] = i
          sequence_lengths[b] = z
          reset_masks[b] = mask  # need to produce a mask for NEXT sample.

          values = self._embedding.get_values(key)
          values_3d = np.reshape(values, embedding_shape)
          label = self._embedding.get_index(key)

          yield (values_3d, label)

    # Calculate size of embedding
    output_shapes = (tf.TensorShape(embedding_shape), tf.TensorShape([]))

    return tf.data.Dataset.from_generator(sequence_generator, output_types=(tf.float32, tf.int32),
                                          output_shapes=output_shapes)
