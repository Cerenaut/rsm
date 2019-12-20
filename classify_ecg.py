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
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import spectrogram

# 18/12/2019, Abdelrahman Ahmed

INPUT_PATH = 'data/ecg'

LEADS = ['I', 'II', 'III', 'aVL', 'aVF', 'aVR', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']

def plot_fourier(signal, labels, freq, fourier, lead='I', idxs=None):
  fig, axs = plt.subplots(2, 2)

  lead_idx = LEADS.index(lead)

  idx = idxs[0]

  axs[0, 0].plot(signal[idx][lead_idx])
  axs[0, 0].set_title('Signal (label = %s)' % labels[idx])
  axs[0, 1].plot(freq, fourier[idx][lead_idx], 'tab:orange')
  axs[0, 1].set_title('Fourier (label = %s)' % labels[idx])

  idx = idxs[1]
  axs[1, 0].plot(signal[idx][lead_idx], 'tab:green')
  axs[1, 0].set_title('Signal (label = %s)' % labels[idx])
  axs[1, 1].plot(freq, fourier[idx][lead_idx], 'tab:red')
  axs[1, 1].set_title('Fourier (label = %s)' % labels[idx])

  # for ax in axs.flat:
  #   ax.set(xlabel='x-label', ylabel='y-label')

  # Hide x labels and tick labels for top plots and y ticks for right plots.
  for ax in axs.flat:
    ax.label_outer()

  plt.show()

def plot_leads(ecg_input, ecg_label=None, show_plot=True, save_plot=False):
  fig, axs = plt.subplots(2, 6)

  j = 0
  for i in range(0, 6):
    axs[0, j].plot(ecg_input[i])
    axs[0, j].set_title('Lead = %s (label = %s)' % (LEADS[i], ecg_label))
    j += 1

  j = 0
  for i in range(6, 12):
    axs[1, j].plot(ecg_input[i])
    axs[1, j].set_title('Lead = %s (label = %s)' % (LEADS[i], ecg_label))
    j += 1

  # for ax in axs.flat:
  #   ax.set(xlabel='x-label', ylabel='y-label')

  # Hide x labels and tick labels for top plots and y ticks for right plots.
  for ax in axs.flat:
    ax.label_outer()

  if show_plot:
    plt.show()

  if save_plot:
    fig.fig.savefig('ecg_leads.png', dpi=fig.dpi)

def main():
  with np.load(os.path.join(INPUT_PATH, 'data.npz')) as f:
    full_frequency = f['full_frequency']

  with np.load(os.path.join(INPUT_PATH, 'labels.npz')) as f:
    labels = f['labels']

  signal = full_frequency
  # signal = avg_frequency

  print('Labels (shape) =', labels.shape)
  print('Signal (shape) =', signal.shape)

  # n = signal.shape[-1]
  n = 50

  # fourier = np.fft.fftn(signal, [n], axes=[2])
  # print('Fourier (shape) =', fourier.shape)

  # freq = np.fft.fftfreq(n)
  # print('Frequency (shape) =', freq.shape)

  f, t, Sxx = spectrogram(signal, fs=500, window='hamming', nperseg=50, scaling='spectrum', axis=-1, mode='magnitude')
  print(f.shape, t.shape, Sxx.shape)

  plt.imshow(full[0][2])
  plt.show()

  # plt.pcolormesh(t, f, Sxx)
  # plt.ylabel('Frequency [Hz]')
  # plt.xlabel('Time [sec]')
  # plt.show()

  # Sliding window
  # signal_windowed = []
  # fourier_windowed = []


  # window = 50
  # stride = 1
  # size = signal.shape[-1]

  # for i in range(0, size, stride):
  #   if (i + window) >= size:
  #     # Skip the windows that would extend beyond the end of the data
  #     continue

  #   signal_tmp = signal[:, :, i:i + window]
  #   signal_windowed.append(signal_tmp)
  #   # print(i, data_tmp.shape)

  #   # data_tmp -= np.mean(data_tmp)
  #   # data_tmp = np.multiply(data_tmp, np.hanning(len(data_tmp)))

  #   fourier = np.fft.fft(signal_tmp)
  #   # fft_data_tmp = abs(fft_data_tmp[:int(len(fft_data_tmp)/2)])**2
  #   fourier_windowed.append(fourier)

  # fourier_windowed = np.array(fourier_windowed)
  # fourier_windowed = np.transpose(fourier_windowed, axes=[1, 2, 0, 3])

  # signal_windowed = np.array(signal_windowed)
  # signal_windowed = np.transpose(signal_windowed, axes=[1, 2, 0, 3])

  # print('Signal (windowed) =', signal_windowed.shape)
  # print('Fourier (windowed) =', fourier_windowed.shape)

  # freq = np.fft.fftfreq(window)
  # print('Frequency (shape) =', freq.shape)

  # idx = 10
  # fig, axs = plt.subplots(2)
  # fig.suptitle('Window = %s, signal (top) and fourier (bottom)' % window)
  # axs[0].plot(signal_windowed[0, 0, idx])
  # axs[1].plot(freq, fourier_windowed[0, 0, idx])
  # plt.show()

  # plot_fourier(signal, labels, freq, fourier, lead='aVF', idxs=[0, 6])
  # plot_leads(full_frequency[0], labels[0])


if __name__ == '__main__':
  main()
