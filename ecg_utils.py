import zipfile

import numpy as np

from scipy.signal import find_peaks

def npz_headers(npz):
  """Takes a path to an .npz file, which is a Zip archive of .npy files.
  Generates a sequence of (name, shape, np.dtype).
  """
  with zipfile.ZipFile(npz) as archive:
    for name in archive.namelist():
      if not name.endswith('.npy'):
        continue

      npy = archive.open(name)
      version = np.lib.format.read_magic(npy)
      shape, fortran, dtype = np.lib.format._read_array_header(npy, version)
      yield name[:-4], shape, dtype

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
  """Using indices of peaks, retrieve peak values and deltas between values."""
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
