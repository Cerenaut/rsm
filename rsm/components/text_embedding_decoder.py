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

"""TextEmbeddingDecoder class."""

import numpy as np
import tensorflow as tf

from pagi.components.dual_component import DualComponent


class TextEmbeddingDecoder(DualComponent):
  """Predicts from a 'path' of choices as in hierarchical language model."""

  # Static names
  embedding = 'embedding'

  def __init__(self, name, dataset):
    super().__init__(name)

    self._dataset = dataset

  def build(self, prediction):
    """Generate the logits from a prediction per batch of the 'choices' in the a binary tree path to each leaf token."""
    prediction_4d_shape = prediction.get_shape().as_list()
    batch_size = prediction_4d_shape[0]
    embedding_2d_shape = self._dataset.get_embedding_shape()
    num_words = self._dataset.num_classes
    num_bits = embedding_2d_shape[0]
    num_draws = embedding_2d_shape[1]

    print("Prediction shape: ", prediction_4d_shape)
    print("# Batch: ", batch_size)
    print("Embedding shape: ", embedding_2d_shape)
    print("# Words: ", num_words)
    print("# Bits: ", num_bits)
    print("# Draws: ", num_draws)

    # embedding_shape is [bits,2*draws]
    # Where:
    # b = batch
    # t = token or word
    # h = embedding values (path choices)
    # w = bits * draws
    prediction_3d_shape = [batch_size, embedding_2d_shape[0], embedding_2d_shape[1]]
    prediction_3d = tf.reshape(prediction, shape=prediction_3d_shape)
    embedding_3d_shape = [num_words, embedding_2d_shape[0], embedding_2d_shape[1]]
    embedding_3d_pl = self._dual.add(self.embedding, shape=embedding_3d_shape, default_value=0.0).add_pl()
    prediction_4d = tf.expand_dims(prediction_3d, axis=1)  # [b,h,w] --> [b,1,h,w] insert token/word/class dimension
    embeddings_4d = tf.expand_dims(embedding_3d_pl, axis=0)  # [t,h,w] --> [1,t,h,w]

    # Load the embedding values once
    embedding = self._dataset.get_embedding()
    matrix = np.reshape(embedding.get_matrix(), embedding_3d_shape)
    self._dual.set_values(self.embedding, matrix)

    diff = tf.clip_by_value(prediction_4d, 0.0, 1.0) - embeddings_4d  # [b,h,w,d]
    inv_abs_diff = 1.0 - tf.abs(diff)  # implies the maximum diff is zero, i.e. the predictions must be unit.
    prediction_logits = tf.reduce_sum(inv_abs_diff, axis=[2, 3], keepdims=False)  # [b,h,w,d] --> [b,h] ie batch x classes
    return prediction_logits

  # COMPONENT INTERFACE --------------------------------------------------------
  def update_feed_dict(self, feed_dict, batch_type='training'):
    del batch_type

    names = [self.embedding]
    self._dual.update_feed_dict(feed_dict, names)
