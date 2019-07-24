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

"""
Kneser Ney smoothing test

Example usage:
----------------------
python kn.py /home/dave/jenkins/data/kn5_predictions_1k /home/dave/jenkins/data/ptb.train.txt \
  /home/dave/jenkins/data/ptb.test.txt /home/dave/jenkins/data/rod_embedding_dense_2kn.txt 5
python kn.py /home/dave/jenkins/data/kn3_predictions_1k /home/dave/jenkins/data/ptb.train.txt \
  /home/dave/jenkins/data/ptb.test.txt /home/dave/jenkins/data/rod_embedding_dense_2kn.txt 3
python kn.py /home/dave/jenkins/data/kn5D_predictions_1k /home/dave/jenkins/data/ptb.train.txt \
  /home/dave/jenkins/data/ptb.test.txt /home/dave/jenkins/data/rod_embedding_dense_2kn.txt 5
python kn.py ./kn5D_predictions_1k /home/dave/agi/penn-treebank/simple-examples/data/ptb.train.txt \
  /home/dave/agi/penn-treebank/simple-examples/data/ptb.test.txt ./inc_embedding.txt 5
python kn.py /home/dave/jenkins/data/kn5_predictions_82k /home/dave/jenkins/data/ptb.train.txt \
  /home/dave/jenkins/data/ptb.test.txt /home/dave/jenkins/data/rod_embedding_dense_2kn.txt 5

On a remote machine (e.g. incbox):
  cd kn5/memory
  nohup python kn.py ./kn5_predictions /home/dave/agi/penn-treebank/simple-examples/data/ptb.train.txt \
    /home/dave/agi/penn-treebank/simple-examples/data/ptb.test.txt ./ptb_dense.txt 5 &
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import numpy as np

from pagi.utils.ngram import KneserNeyNGram

from rsm.datasets.text_embedding_dataset import TextEmbeddingDataset


def main(test_dist_file, corpus_train_file, corpus_test_file, embedding_file, order, limit, discount):
  print('Prediction distributions file: ', test_dist_file)
  print('Training corpus: ', corpus_train_file)
  print('Testing corpus: ', corpus_test_file)
  print('Embedding file: ', embedding_file)
  print('Order: ', order)
  print('Limit: ', limit)
  print('Discount: ', discount)

  pad_start = '<end>'
  pad_end = pad_start
  eos = pad_end

  batch_size = 1
  max_sequence_length = 1000
  embedding_w = 20
  embedding_h = embedding_w
  embedding_sparsity = 30
  embedding_type = 'dense'
  embedding_shape = [int(embedding_h), int(embedding_w)]

  print('Create dataset')
  dataset = TextEmbeddingDataset('.')
  print('Setup dataset')
  dataset.setup(int(batch_size), corpus_train_file, corpus_test_file, embedding_type, embedding_file, embedding_shape,
                embedding_sparsity, max_sequence_length, eos=eos)

  # Sentences
  print('Get sentences')
  sentences_train = dataset.get_embedding().read_corpus_files([corpus_train_file])
  num_train = len(sentences_train)
  print('Have ', num_train, ' training sentences')

  sentences_test = dataset.get_embedding().read_corpus_files([corpus_test_file])
  num_test_sentences = len(sentences_test)
  print('Have ', num_test_sentences, ' testing sentences')

  # Tokens
  tokens_train = dataset.get_words(dataset.get_embedding(), corpus_train_file, eos)
  num_train = len(tokens_train)
  print('Have ', num_train, ' training tokens')

  tokens_test = dataset.get_words(dataset.get_embedding(), corpus_test_file, eos)
  num_test = len(tokens_test)
  print('Have ', num_test, ' testing tokens')

  num_tokens = dataset.get_embedding().get_num_keys()
  print('Have ', num_tokens, ' tokens in dictionary')
  dictionary = dataset.get_embedding().get_keys()

  print('Creating language model')
  lm = KneserNeyNGram(sents=sentences_train, words=tokens_train, n=order, discount=discount)

  # optimize the discount using validation or test set
  # lm.optimize_discount(sentences_test)

  # Hack - for debugging, truncate the test set and/or tokens
  if limit > 0:
    num_test = limit

  token_buffer = []
  for j in range(order):
    token_buffer.append(eos)

  test_dists = np.zeros([num_test, num_tokens])

  for i in range(num_test):
    print('Token ', i, ' of ', num_test, '. Token buffer: ', token_buffer)

    sum_p = 0.0
    token_buffer_tuple = tuple(token_buffer)
    for k in range(num_tokens):
      pred_token = dictionary[k]
      p = lm.cond_prob_fast(pred_token, prev_tokens=token_buffer_tuple)
      test_dists[i][k] = p
      sum_p += p

    del token_buffer_tuple

    token = tokens_test[i]

    # rotate the token buffer
    for j in range(order-1):
      token_buffer[j] = token_buffer[j+1]
    token_buffer[order-1] = token

  print('Writing prediction distributions to: ', test_dist_file)
  np.save(test_dist_file, test_dists)
  print('Program complete.')

if __name__ == '__main__':
  args = {
      'limit': -1,
      'discount': None
  }

  args['test_dist_file'] = sys.argv[1]
  args['corpus_train_file'] = sys.argv[2]
  args['corpus_test_file'] = sys.argv[3]
  args['embedding_file'] = sys.argv[4]
  args['order'] = int(sys.argv[5])

  if len(sys.argv) > 6:
    args['limit'] = int(sys.argv[6])
  if len(sys.argv) > 7:
    args['discount'] = float(sys.argv[7])

  main(**args)
