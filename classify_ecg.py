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
import itertools

import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import spectrogram, find_peaks
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from tensorflow import keras
from tensorflow.keras import layers, regularizers


# 18/12/2019, Abdelrahman Ahmed

INPUT_PATH = 'data/ecg'

LEADS = ['I', 'II', 'III', 'aVL', 'aVF', 'aVR', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']

SEED = 42


def find_peaks_2d(array, distance):
  """Extends scipy.signal.find_peaks to 2D-arrays."""
  peaks = []

  lens = []
  for i, _ in enumerate(array):
    peaks_1d = []
    for j, _ in enumerate(array[i]):
      peaks_2d, _ = find_peaks(array[i][j], distance=distance)
      peaks_2d = list(peaks_2d)
      lens.append(len(peaks_2d))
      peaks_1d.append(peaks_2d)
    peaks.append(peaks_1d)

  avg_len = int(np.mean(lens))

  return peaks, avg_len


def get_peak_values(array, peaks, peak_len=None):
  peaks_shape = [len(peaks), 0, 0]

  for sample in peaks:
    sample_max = len(sample)
    if sample_max > peaks_shape[1]:
      peaks_shape[1] = sample_max

    if peak_len is None:
      for lead in sample:
        lead_max = len(lead)
        if lead_max > peaks_shape[2]:
          peaks_shape[2] = lead_max
    else:
      peaks_shape[2] = peak_len

  peak_values = np.zeros(peaks_shape)
  peak_values_delta = np.zeros([peaks_shape[0], peaks_shape[1], peaks_shape[2] - 1])
  # print('Peak values (shape) =', peak_values.shape)

  for i, _ in enumerate(peaks):
    for j, v in enumerate(peaks[i]):
      skip_peak = False

      if not v:
        skip_peak = True

      if peak_len and len(v) < peak_len:
        skip_peak = True

      if not skip_peak:
        if peak_len:
          start = len(v) // 2 - peak_len // 2
          v = v[start:start + peak_len]
        value = array[i][j][v]
        delta = np.diff(array[i][j][v])
        peak_values[i][j][0:len(v)] = value
        peak_values_delta[i][j][0:len(v)-1] = delta

  return peak_values, peak_values_delta


def main():
  # Data Loader
  # ---------------------------------------------------------------------------
  with np.load(os.path.join(INPUT_PATH, 'data.npz')) as f:
    full_frequency = f['full_frequency']

  with np.load(os.path.join(INPUT_PATH, 'labels.npz')) as f:
    labels = f['labels']

  signal = full_frequency
  # signal = avg_frequency

  print('Labels (shape) =', labels.shape)
  print('Signal (shape) =', signal.shape)

  # Signal Processing
  # ----------------------------------------------------------------------------
  timestep = 0.001    # it looks like milliseconds from the plot
  window_size = 20

  f, t, sg = spectrogram(signal, fs=1/timestep, window='hamming', nperseg=window_size, scaling='spectrum', axis=-1,
                         mode='magnitude')

  print('Frequency (shape) =', f.shape, 'Time (shape) =', t.shape, 'Spectrogram (shape) =', sg.shape, '\n')

  mode = 'peaks'  # spectrogram, peaks
  lead_idx = LEADS.index('I')

  print('==============================================================')
  print('Using LEAD:', LEADS[lead_idx], '\n')

  # Data Preprocessing
  # ----------------------------------------------------------------------------
  if mode == 'peaks':
    peak_distance = 5

    spectrogram_max = np.amax(sg, axis=2)
    peaks, avg_len = find_peaks_2d(spectrogram_max, distance=peak_distance)

    peak_values, peak_values_delta = get_peak_values(spectrogram_max, peaks, peak_len=avg_len)

    # DEBUG: Show peaks in matplotlib, overlayed on plot as 'x'
    # sidx = 1
    # plt.plot(spectrogram_max[sidx][lead_idx])
    # plt.plot(peaks[sidx][lead_idx], np.trim_zeros(peak_values[sidx][lead_idx]), 'x')
    # plt.show()

    input_data = peak_values_delta[:, lead_idx]  # pylint: disable=invalid-sequence-index

    nonzero_idxs = ~np.all(input_data == 0, axis=1)
    input_data = input_data[nonzero_idxs]
    input_labels = labels[nonzero_idxs]
  else:
    input_data = sg[:, lead_idx]
    input_labels = labels

  # Flatten the inputs
  input_data = np.reshape(input_data, [input_data.shape[0], np.prod(input_data.shape[1:])])

  # Classification
  # ---------------------------------------------------------------------------
  model = 'nn'  # logistic, nn

  print('Input data, flattened (shape)', input_data.shape)
  print('Input labels (shape)', input_labels.shape, '\n')

  train_data, test_data, train_labels, test_labels = train_test_split(
      input_data, input_labels, stratify=input_labels, test_size=0.20, random_state=SEED)

  # train_data, val_data, train_labels, val_labels = train_test_split(
  #     train_data, train_labels, stratify=train_labels, test_size=0.10, random_state=SEED)

  print('Training data (shape) =', train_data.shape, train_labels.shape)
  # print('Validation data (shape) =', val_data.shape, val_labels.shape)
  print('Test data (shape) =', test_data.shape, test_labels.shape, '\n')

  if model == 'nn':
    batch_size = 128
    num_epochs = 100
    num_units = [256]
    penalty_l2 = 0.1

    inputs = keras.Input(shape=(train_data.shape[1],), name='inputs')
    layer_output = layers.Dense(num_units[0], activation='relu', name='dense_1',
                                kernel_regularizer=regularizers.l2(penalty_l2))(inputs)
    outputs = layers.Dense(1, activation='sigmoid', name='predictions')(layer_output)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_data, train_labels,
              batch_size=batch_size,
              epochs=num_epochs,
              verbose=0)

    train_results = model.evaluate(train_data, train_labels, batch_size=batch_size, verbose=0)
    test_results = model.evaluate(test_data, test_labels, batch_size=batch_size, verbose=0)

    _, train_acc = train_results
    _, test_acc = test_results

    binary_threshold = 0.5
    train_probs = model.predict(train_data)
    train_preds = train_probs > binary_threshold

    test_probs = model.predict(test_data)
    test_preds = test_probs > binary_threshold

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
