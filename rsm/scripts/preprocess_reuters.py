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

"""Experiment framework for training and evaluating COMPONENTS."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

import tensorflow as tf

from bs4 import BeautifulSoup

from pagi.utils.embedding import Embedding


def main(args):

  print("Args:", args)

  # data_dir = '/home/dave/agi/ptb_err'
  # count_file = 'error_count.csv'
  # dist_file = 'error_hist.csv'
  embedding_file = './ptb_embedding.txt'
  #input_file = '/home/dave/agi/reuters_news/reuters21578/reut2-000.sgm'
  #output_file = 'reuters.txt'
  input_file = args[1]
  output_file = args[2]

  e = Embedding()
  e.clear()
  e.read(embedding_file)

  f = open(input_file, 'r')
  data = f.read()
  #print( 'data: ', data)

  # Must replace body with content tags, for reasons
  # See: https://stackoverflow.com/questions/15863751/extracting-body-tags-from-smg-file-beautiful-soup-and-python
  data_replaced = data.replace('<BODY>', '<content>')
  data_replaced = data_replaced.replace('</BODY>', '</content>')

  # Parse the modified content
  tag = 'content'
  unknown_token = '<unk>'
  number_token = 'N'
  num_footer_tags = 2  # Reuters always has a footer at the end

  # https://www.crummy.com/software/BeautifulSoup/bs4/doc/#contents-and-children
  soup = BeautifulSoup(data_replaced)
  articles = soup.findAll(tag) # find all body tags
  print('Have ', len(articles), ' articles.')  # print number of body tags in sgm file
  i = 0
  corpus = ''

  # Loop through each body tag and print its content
  for article in articles:  # pylint: disable=too-many-nested-blocks
    content = article.contents
    if i < 10:
      print('Article: ', content)
      print('| ')

    output = ''
    output_list = []

    tokens = content[0].split()  # on whitespace
    num_tokens = len(tokens)
    for j in range(num_tokens-num_footer_tags):
      input_token = tokens[j]
      token = input_token.strip()

      # force lowercase
      token = token.lower()

      # remove ALL commas (there are none in PTB)
      #token = token.replace('\n', ' ')

      # remove ALL commas (there are none in PTB)
      token = token.replace(',', '')

      # replace dlrs with $
      token = token.replace('dlrs', '$')
      token = token.replace('dlr', '$')

      # replace mln
      token = token.replace('mln', 'million')
      token = token.replace('bln', 'billion')
      token = token.replace('trn', 'trillion')

      # replace tonnes
      token = token.replace('tonnes', 'tons')

      # replace pct with percent
      token = token.replace('pct', 'percent')

      # remove trailing periods
      end_of_sentence = False
      if token.endswith('.'):
        end_of_sentence = True
        token = token[:-1]

      # replace the angle brackets around proper nouns
      token = token.replace('<', '')
      token = token.replace('>', '')

      # replace numbers with N
      try:
        float(token)
        token = number_token
      except ValueError:
        pass

      # https://stackoverflow.com/questions/5917082/regular-expression-to-match-numbers-with-or-without-commas-and-decimals-in-text
      is_number = re.search('(?<!\S)(?=.)(0|([1-9](\d*|\d{0,2}(,\d{3})*)))?(\.\d*[1-9])?(?!\S)', token)  # pylint: disable=anomalous-backslash-in-string
      if is_number:
        token = number_token

      # space before 's and 're etc.
      # was n't did n't etc.
      #if token == 'didn\'t':
      suffix = None
      recognized = False
      if token.endswith('n\'t'):
        suffix = ' n\'t'  # split into 2 tokens
        token = token.replace('n\'t', '')
      elif token.endswith('\'s'):
        suffix = ' \'s'  # split into 2 tokens
        token = token.replace('\'s', '')

      elif token.endswith('\'re'):
        suffix = ' \'re'  # split into 2 tokens
        token = token.replace('\'re', '')

      # replace unknown tokens with UNK
      if not recognized:
        has_key = e.has_key(token)
        if not has_key:
          token = unknown_token

      #if i<10:
      #  print('Original: ', input_token, ' TOKEN: |', token, '| In dict?: ', has_key, ' EOS?: ', end_of_sentence)
      output_list.append(token)
      if suffix is not None:
        output_list.append(suffix)
      #output = output + token + suffix
      #output = output + ' '

      if end_of_sentence:

        # Reorder some common tokens where the style is peculiar to a particular outlet
        # Reuters style:   N million $      N $     N million $
        # PTB (WSJ):     $ N million      $ N     $ N billion
        output_length = len(output_list)
        for k in range(output_length):
          if k > 0:
            output_token_1 = output_list[k-1]
            output_token_2 = output_list[k]

            # N $ --> $ N
            if (output_token_1 == 'N') and (output_token_2 == '$'):
              output_list[k-1] = '$'
              output_list[k] = 'N'
            elif k > 1:
              output_token_0 = output_list[k-2]

              if output_token_0 == 'N' and output_token_1 in ['million', 'billion', 'trillion'] and (
                  output_token_2 == '$'):
                output_list[k-2] = '$'
                output_list[k-1] = 'N'
                output_list[k] = output_token_1

        # Copy the final list to the output buffer
        for k in range(output_length):
          output_token = output_list[k]
          output = output + output_token + ' '

        # Add EOS marker
        output = output + '\n'

        # Clear the token list
        output_list = []  # reset list

    if i < 10:
      print('ArticTx: ', output)
      print('--------------\n\n')

    # assemble the final corpus line, add newline at end
    corpus = corpus + output
    i = i + 1

  print('Articles: ', i)

  with open(output_file, 'a') as text_file:
    text_file.write(corpus)

if __name__ == '__main__':
  tf.app.run()
