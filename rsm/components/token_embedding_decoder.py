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

"""TokenEmbeddingDecoder class."""

import numpy as np
import tensorflow as tf

from pagi.components.dual_component import DualComponent


class TokenEmbeddingDecoder(DualComponent):
  """Predicts from binary-tree 'paths' of choices as in hierarchical language model."""

  # Static names
  embedding = 'embedding'

  def __init__(self, name, dataset):
    super().__init__(name)

    self._dataset = dataset

  def build(self, prediction):
    """Generate a prediction distribution over all tokens, for each batch sample."""

    # Load the embedding values once
    embedding = self._dataset.get_embedding()
    token_vectors = embedding.get_tokens_values()  # [t,h,w]
    token_vectors_shape = token_vectors.shape

    np.set_printoptions(threshold=np.nan)
    print('> token vectors shape: ', token_vectors.shape)
    print('> prediction shape: ', prediction.shape)
    print('> token vectors: ', token_vectors)
        
    # [t,h,w] where h = num trees and w = num decisions
    token_vectors_3d_pl = self._dual.add(self.embedding, shape=token_vectors_shape, default_value=0.0).add_pl()
    self._dual.set_values(self.embedding, token_vectors)
    token_vectors_4d = tf.expand_dims(token_vectors_3d_pl, 0)  # [1,t,h,w]

    m0 = 1.0 - token_vectors_4d
    m1 = token_vectors_4d
    m2 = tf.maximum(token_vectors_4d, 1.0) -1.0 # [1,t,h,w]

    # m2 mask: want a 1 where value is 2
    # x max(x,1) -1
    # 2 2         1
    # 1 1         0
    # 0 1         0

    prediction = tf.clip_by_value(prediction, 0.0, 1.0)
    prediction_vectors_3d = tf.reduce_sum(prediction, axis=3)  # [b,h,w,1] -> [b,h,w]
    prediction_vectors_4d = tf.expand_dims(prediction_vectors_3d, axis=1)  # [b,h,w] -> [b,1,h,w]

    p0 = 1.0 - prediction_vectors_4d
    p1 = prediction_vectors_4d

    # [b,t,h,w] = [300*10000*2*20] (approx 120m elems)
    # = M0[w] * P0[w]
    # + M1[w] * P1[w]

    tree_paths_probs = m0 * p0 + m1 * p1 # [b,t,h,w]
    tree_paths_probs = tf.maximum(tree_paths_probs, m2) # [b,t,h,w]
    tree_predictions = tf.reduce_prod(tree_paths_probs, axis=3)  # [b,t,h,w] -> [b,t,h]
    sum_predictions = tf.reduce_sum(tree_predictions, axis=2)  # [b,t,h] -> [b,t]
    sum_distributions = tf.reduce_sum(sum_predictions, axis=1, keepdims=True) # [b,1]
    prediction_distributions = tf.divide(sum_predictions, sum_distributions)  # [b,t] A dist per batch
    return prediction_distributions

  # COMPONENT INTERFACE --------------------------------------------------------
  def update_feed_dict(self, feed_dict, batch_type='training'):
    del batch_type

    names = [self.embedding]
    self._dual.update_feed_dict(feed_dict, names)
