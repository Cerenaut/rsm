# Copyright (C) 2018 Project AGI
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

"""Kneser Ney smoothing test"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gc
import ast
import json
import sys

import math
import numpy as np

#import mlflow
#import tensorflow as tf


from datasets.text_embedding_dataset import TextEmbeddingDataset
from pagi.utils.sparse_embedding import SparseEmbedding

from pagi.utils import generic_utils as util

# from nltk.corpus import gutenberg
# from nltk.util import ngrams

from pagi.utils.ngram import NGram, AddOneNGram, InterpolatedNGram, BackOffNGram, KneserNeyNGram

def main(test_dist_file, corpus_train_file, corpus_test_file, embedding_file, order, limit, discount):

  print('Prediction distributions file: ', test_dist_file)
  print('Training corpus: ', corpus_train_file)
  print('Testing corpus: ', corpus_test_file)
  print('Embedding file: ', embedding_file)
  print('Order: ', order)
  print('Limit: ', limit)
  print('Discount: ', discount)

  #pad_start = '<s>'
  pad_start = '<end>'
  pad_end = pad_start
  eos = pad_end

  batch_size = 1
  max_sequence_length = 1000
  embedding_w = 20
  embedding_h = embedding_w
  embedding_sparsity = 30  
  #embedding_type = 'sparse'
  embedding_type = 'dense'
  embedding_shape = [int(embedding_h), int(embedding_w)]

  print('Create dataset')
  dataset = TextEmbeddingDataset('.')
  print('Setup dataset')
  dataset.setup(int(batch_size), corpus_train_file, corpus_test_file, embedding_type, embedding_file, embedding_shape, embedding_sparsity, max_sequence_length, eos=eos)

  # Sentences
  print('Get sentences')
  sentences_train = dataset._embedding.read_corpus_files([corpus_train_file])
  num_train = len(sentences_train)
  print('Have ', num_train, ' training sentences')

  sentences_test = dataset._embedding.read_corpus_files([corpus_test_file])
  num_test_sentences = len(sentences_test)
  print('Have ', num_test_sentences, ' testing sentences')

  # Tokens
  tokens_train = dataset.get_words(dataset._embedding, corpus_train_file, eos)
  num_train = len(tokens_train)
  print('Have ', num_train, ' training tokens')

  tokens_test = dataset.get_words(dataset._embedding, corpus_test_file, eos)
  num_test = len(tokens_test)
  print('Have ', num_test, ' testing tokens')

  num_tokens = dataset._embedding.get_num_keys()
  print('Have ', num_tokens, ' tokens in dictionary')
  dictionary = dataset._embedding.get_keys()
  
  print('Creating language model')
  lm = KneserNeyNGram(sents=sentences_train, words=tokens_train, n=order, discount=discount)

  # optimize the discount using validation or test set
  #lm.optimize_discount(sentences_test)

  # Hack - for debugging, truncate the test set and/or tokens
  if limit > 0:
    num_test = limit
  #num_test = 1000
  #num_test = 15
  #num_tokens = 8

  token_buffer = []
  for j in range(order):
    token_buffer.append(eos)

  test_dists = np.zeros([num_test,num_tokens])
  #print('test dists shape', test_dists.shape)

  for i in range(num_test):

    # if (i % 100) == 0:
    #   print('Garbage collection -----------------------------------')
    #   gc.collect()

    print('Token ', i, ' of ', num_test, '. Token buffer: ', token_buffer)

    sum_p = 0.0
    token_buffer_tuple = tuple(token_buffer)
    for k in range(num_tokens):
      pred_token = dictionary[k]
      p = lm.cond_prob_fast(pred_token, prev_tokens=token_buffer_tuple)
      #p = lm.cond_prob(pred_token, prev_tokens=token_buffer_tuple)
      #p = lm.cond_prob(pred_token, prev_tokens=token_buffer)
      #print('Evaluating token: ', pred_token, ' P=', p)
      test_dists[i][k] = p
      sum_p += p

    del token_buffer_tuple

    #print('Sum P:', sum_p)

    token = tokens_test[i]
    #print('New token:', token)

    # rotate the token buffer
    for j in range(order-1):
      token_buffer[j] = token_buffer[j+1]
    token_buffer[order-1] = token

  print('Writing prediction distributions to: ', test_dist_file)
  #test_dists.tofile(test_dist_file)
  np.save(test_dist_file, test_dists)
  print('Program complete.')

# Example usage:
# python kn.py /home/dave/jenkins/data/kn5_predictions_1k /home/dave/jenkins/data/ptb.train.txt /home/dave/jenkins/data/ptb.test.txt /home/dave/jenkins/data/rod_embedding_dense_2kn.txt 5
# python kn.py /home/dave/jenkins/data/kn3_predictions_1k /home/dave/jenkins/data/ptb.train.txt /home/dave/jenkins/data/ptb.test.txt /home/dave/jenkins/data/rod_embedding_dense_2kn.txt 3
# python kn.py /home/dave/jenkins/data/kn5D_predictions_1k /home/dave/jenkins/data/ptb.train.txt /home/dave/jenkins/data/ptb.test.txt /home/dave/jenkins/data/rod_embedding_dense_2kn.txt 5

# python kn.py ./kn5D_predictions_1k /home/dave/agi/penn-treebank/simple-examples/data/ptb.train.txt /home/dave/agi/penn-treebank/simple-examples/data/ptb.test.txt ./inc_embedding.txt 5

# python kn.py /home/dave/jenkins/data/kn5_predictions_82k /home/dave/jenkins/data/ptb.train.txt /home/dave/jenkins/data/ptb.test.txt /home/dave/jenkins/data/rod_embedding_dense_2kn.txt 5
# incbox:
# cd kn5/memory
# nohup python kn.py ./kn5_predictions /home/dave/agi/penn-treebank/simple-examples/data/ptb.train.txt /home/dave/agi/penn-treebank/simple-examples/data/ptb.test.txt ./ptb_dense.txt 5 &

test_dist_file = sys.argv[1]
corpus_train_file = sys.argv[2]
corpus_test_file = sys.argv[3]
embedding_file = sys.argv[4]
order = int(sys.argv[5])
limit = -1
discount = None
if len(sys.argv) > 6:
  limit = int(sys.argv[6])
if len(sys.argv) > 7:
  discount = float(sys.argv[7])

if __name__ == "__main__": 
  main(test_dist_file, corpus_train_file, corpus_test_file, embedding_file, order, limit, discount)
