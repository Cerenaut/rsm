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

"""ECG Data Parser."""

import os
import pathlib
import xml.etree.ElementTree as ET

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 25/09/2019, Shuai Sun
# 18/12/2019, Abdelrahman Ahmed

INPUT_PATH = 'data/ecg_xml'
OUTPUT_PATH = 'data/ecg'

SEED = 42
FREQUENCY_LEN = 5000
NUM_SAMPLES_PER_FILE = 5000

CLASSES = {
    'normal': 0,
    'abnormal': 1
}

def parse_features(group):
  features = []
  for item in group:
    item_value = item.text
    x = item_value.split(',')
    y = [float(p) for p in x]
    features.append(y)
  return features


def parse_xml(root):
  for child in root:
    print(child.tag, ":", child.attrib, ':', child.text)

    for children in child:
      print(children.tag, ':', children.attrib, ':', children.text)

      for gradchild in children:
        print(gradchild.tag, ':', gradchild.attrib, gradchild.text)


def chunks(lst, n):
  """Yield successive n-sized chunks from lst."""
  for i in range(0, len(lst), n):
    yield lst[i:i + n]


def main():
  filepath = os.path.join(INPUT_PATH, '*.xml')

  filenames = []
  for root, _, files in os.walk(INPUT_PATH):
    for file in files:
      if file.endswith('.xml'):
        filepath = os.path.join(root, file)
        filenames.append(filepath)

  np.random.shuffle(filenames)

  print('Parsing XML Data...\n')

  skipped_samples = {
      'normal': {
          'count': 0,
          'files': []
      },
      'abnormal': {
          'count': 0,
          'files': []
      },
  }

  filenames_chunks = list(chunks(filenames, NUM_SAMPLES_PER_FILE))

  for i, filenames_chunk in enumerate(filenames_chunks):
    labels = []
    full_frequency = []

    for j, filepath in enumerate(filenames_chunk):
      parent_dir = filepath.split('/')[-2]  # Get immediate parent of this file
      label_key = parent_dir.split('.')[-1]

      if label_key not in CLASSES:
        continue

      try:
        tree = ET.parse(filepath)
        root = tree.getroot()
      except Exception:  # pylint: disable=broad-except
        skipped_samples[label_key]['count'] += 1
        skipped_samples[label_key]['files'].append(filepath)
        continue

      groups = root[-1]

      # Full Frequency
      features = parse_features(groups[0])

      if len(features[0]) != FREQUENCY_LEN:
        skipped_samples[label_key]['count'] += 1
        skipped_samples[label_key]['files'].append(filepath)
        continue

      full_frequency.append(features)

      label = CLASSES[label_key]
      labels.append(label)

    labels = np.array(labels)
    signal = np.array(full_frequency)

    classes, classes_freq = np.unique(labels, return_counts=True)

    print('Classes =', classes)
    print('Classes Frequency =', classes_freq)

    print('Labels (shape) =', labels.shape)
    print('Signal (shape) =', signal.shape)
    print('Number of skipped files:', skipped_samples)

    pathlib.Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)

    train_data, test_data, train_labels, test_labels = train_test_split(
        signal, labels, stratify=labels, test_size=0.20, random_state=SEED)

    filename_suffix = '-' + str(i)

    np.savez_compressed(os.path.join(OUTPUT_PATH, 'train_data' + filename_suffix), signal=train_data)
    np.savez_compressed(os.path.join(OUTPUT_PATH, 'train_labels' + filename_suffix), labels=train_labels)

    np.savez_compressed(os.path.join(OUTPUT_PATH, 'test_data' + filename_suffix), signal=test_data)
    np.savez_compressed(os.path.join(OUTPUT_PATH, 'test_labels' + filename_suffix), labels=test_labels)

if __name__ == '__main__':
  main()
