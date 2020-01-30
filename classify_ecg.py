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

"""ECG Data Classifier."""

import os
import glob
import zipfile
import itertools

import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import spectrogram, find_peaks
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from tensorflow import keras
from tensorflow.keras import layers, regularizers, models

import ecg_utils

# 18/12/2019, Abdelrahman Ahmed

INPUT_PATH = 'data/ecg'

LEADS = ['I', 'II', 'III', 'aVL', 'aVF', 'aVR', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']

SEED = 42


def preprocess(signal, labels, window_size, timestep, peak_distance, peak_len=None, mode='spectrogram', lead='I'):
  """Perform signal processing using spectrogram. Mode supports spectrogram (default) or 'peaks' for peak deltas. """

  # Signal Processing
  # ----------------------------------------------------------------------------
  f, t, sg = spectrogram(signal, fs=1/timestep, window='hamming', nperseg=window_size, scaling='spectrum', axis=-1,
                         mode='magnitude')

  # print('Frequency (shape) =', f.shape, 'Time (shape) =', t.shape, 'Spectrogram (shape) =', sg.shape, '\n')

  # Get lead index
  lead_idx = LEADS.index(lead)

  # Data Preprocessing
  # ----------------------------------------------------------------------------
  if mode == 'peaks':
    spectrogram_max = np.amax(sg, axis=2)
    peaks, avg_len = ecg_utils.find_peaks_2d(spectrogram_max, distance=peak_distance)

    if peak_len is None:
      peak_len = avg_len

    _, peak_values_delta = ecg_utils.get_peak_values(spectrogram_max, peaks, peak_len=peak_len)

    input_data = peak_values_delta[:, lead_idx]  # pylint: disable=invalid-sequence-index

    nonzero_idxs = ~np.all(input_data == 0, axis=1)
    input_data = input_data[nonzero_idxs]
    input_labels = labels[nonzero_idxs]
  elif mode == 'spectrogram':
    input_data = sg[:, lead_idx]
    input_labels = labels

  # Flatten the inputs
  input_data = np.reshape(input_data, [input_data.shape[0], np.prod(input_data.shape[1:])])

  # print('Input data, flattened (shape)', input_data.shape)
  # print('Input labels (shape)', input_labels.shape, '\n')

  return input_data, input_labels


def main():
  # Data Loader
  # ---------------------------------------------------------------------------
  train_data_files = glob.glob(os.path.join(INPUT_PATH, 'train_data-*.npz'))
  train_labels_files = glob.glob(os.path.join(INPUT_PATH, 'train_labels-*.npz'))
  train_data_files, train_labels_files = sorted(train_data_files), sorted(train_labels_files)

  test_data_files = glob.glob(os.path.join(INPUT_PATH, 'test_data-*.npz'))
  test_labels_files = glob.glob(os.path.join(INPUT_PATH, 'test_labels-*.npz'))
  test_data_files, test_labels_files = sorted(test_data_files), sorted(test_labels_files)

  # Retrieve shapes without loading data into memory
  num_train_samples = 0
  for i in train_data_files:
    _, signal_shape, _ = next(ecg_utils.npz_headers(i))
    num_train_samples += signal_shape[0]

  num_train_labels = 0
  for i in train_labels_files:
    _, labels_shape, _ = next(ecg_utils.npz_headers(i))
    num_train_labels += labels_shape[0]

  num_test_samples = 0
  for i in test_data_files:
    _, signal_shape, _ = next(ecg_utils.npz_headers(i))
    num_test_samples += signal_shape[0]

  num_test_labels = 0
  for i in test_labels_files:
    _, labels_shape, _ = next(ecg_utils.npz_headers(i))
    num_test_labels += labels_shape[0]

  print('Number of training samples =', num_train_samples)
  print('Number of training labels =', num_train_labels)
  print('Number of test samples =', num_test_samples)
  print('Number of test labels =', num_test_labels)

  # Preprocessing Parameters
  lead = 'I'
  timestep = 0.001    # it looks like milliseconds from the plot
  window_size = 100
  peak_distance = 1
  peak_len = 17
  mode = 'spectrogram'

  # NN Params
  batch_size = 128
  num_epochs = 20
  num_units = [256]
  penalty_l2 = 0.05

  print('==============================================================')
  print('Using LEAD:', lead, '\n')

  def data_generator(data_files, labels_files):
    """Reads data from disk using a generator fn."""
    while 1:
      for data_file, labels_file in zip(data_files, labels_files):
        try:
          signal = np.load(data_file)['signal']
          labels = np.load(labels_file)['labels']

          signal, labels = preprocess(signal, labels, timestep=timestep, window_size=window_size,
                                      peak_distance=peak_distance, peak_len=peak_len, mode=mode, lead=lead)

          batches = int(np.ceil(len(labels) / batch_size))

          for i in range(0, batches):
            end = min(len(signal), i * batch_size + batch_size)
            yield signal[i * batch_size:end], labels[i * batch_size:end]
        except EOFError:
          print('Error processing files: ' + data_filepath)

  train_gen = data_generator(train_data_files, train_labels_files)
  test_gen = data_generator(test_data_files, test_labels_files)

  sample_signal, _ = next(train_gen)

  num_train_batches = int(np.ceil(num_train_samples / batch_size))
  num_test_batches = int(np.ceil(num_test_samples / batch_size))

  # Classification
  # ---------------------------------------------------------------------------
  model = 'nn'  # logistic, nn

  if model == 'nn':
    verbosity_level = 1

    inputs = keras.Input(shape=(sample_signal.shape[1],), name='inputs')
    layer_output = layers.Dense(num_units[0], activation='relu', name='dense_1',
                                kernel_regularizer=regularizers.l2(penalty_l2))(inputs)
    outputs = layers.Dense(1, activation='sigmoid', name='predictions')(layer_output)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    tensorboard_cb = keras.callbacks.TensorBoard(
        log_dir='./run/cardioscan', histogram_freq=0, write_graph=True, write_images=True)

    model.fit_generator(train_gen,
                        steps_per_epoch=num_train_batches,
                        epochs=num_epochs,
                        verbose=verbosity_level,
                        callbacks=[tensorboard_cb])
    model.save('./run/cardioscan/model.h5')

    train_results = model.evaluate_generator(train_gen, steps=num_train_batches, verbose=verbosity_level)
    test_results = model.evaluate_generator(test_gen, steps=num_test_batches, verbose=verbosity_level)

    _, train_acc = train_results
    _, test_acc = test_results

    print('Training Accuracy =', train_results)
    print('Test Accuracy =', test_acc)

    # binary_threshold = 0.5
    # train_probs = model.predict_generator(train_gen, steps=num_train_batches)
    # train_preds = train_probs > binary_threshold

    # test_probs = model.predict_generator(test_gen, steps=num_test_batches)
    # test_preds = test_probs > binary_threshold

    # train_labels = []
    # for i, (_, labels) in enumerate(train_gen):
    #   train_labels.extend(labels)

    #   if len(train_labels) == num_train_samples:
    #     break

    # test_labels = []
    # for i, (_, labels) in enumerate(test_gen):
    #   test_labels.extend(labels)

    #   if len(test_labels) == num_test_samples:
    #     break

  else:
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
    clf = clf.fit(train_data, train_labels)

    train_preds = clf.predict(train_data)
    test_preds = clf.predict(test_data)

    train_acc = clf.score(train_data, train_labels)
    test_acc = clf.score(test_data, test_labels)

    train_f1 = metrics.classification_report(train_labels, train_preds)
    train_cm = metrics.confusion_matrix(train_labels, train_preds)

    test_f1 = metrics.classification_report(test_labels, test_preds)
    test_cm = metrics.confusion_matrix(test_labels, test_preds)

    def print_results(acc, f1, cm):
      tn, fp, fn, tp = cm.ravel()

      print('Accuracy =', acc)
      print('\n')
      print('True Negatives =', tn)
      print('False Positives =', fp)
      print('False Negatives =', fn)
      print('True Positives =', tp)
      print('\n')
      print(f1)
      print('\n')

    print('================ Training Results ================')
    print_results(train_acc, train_f1, train_cm)

    print('================ Test Results     ================')
    print_results(test_acc, test_f1, test_cm)

if __name__ == '__main__':
  main()
