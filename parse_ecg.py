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

# 25/09/2019, Shuai Sun
# 18/12/2019, Abdelrahman Ahmed

INPUT_PATH = 'data/ecg_xml'
OUTPUT_PATH = 'data/ecg'

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


def main():
  filepath = os.path.join(INPUT_PATH, '*.xml')

  filenames = []
  for root, dirs, files in os.walk(INPUT_PATH):
    for file in files:
      if file.endswith('.xml'):
        filepath = os.path.join(root, file)
        filenames.append(filepath)

  print('filenames =', filenames)

  labels = []
  full_frequency = []
  avg_frequency = []

  for file in filenames:
    parent_dir = file.split('/')[-2]  # Get immediate parent of this file
    label_key = parent_dir.split('.')[-1]
    label = CLASSES[label_key]
    labels.append(label)

    tree = ET.parse(file)
    root = tree.getroot()

    groups = root[-1]

    # Full frequency = 500>
    features = parse_features(groups[0])
    full_frequency.append(features)

    # AVG frequency = 500>
    features = parse_features(groups[1])
    avg_frequency.append(features)

  labels = np.array(labels)
  full_frequency = np.array(full_frequency)
  avg_frequency = np.array(avg_frequency)

  print('Labels (shape) =', labels.shape)
  print('Full Frequency (shape) =', full_frequency.shape)
  print('AVG Frequency (shape) =', avg_frequency.shape)

  # fig = plt.figure()
  # plt.plot(full_frequency[0, 0, :])
  # fig.savefig('ecg.png', dpi=fig.dpi)

  pathlib.Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)

  np.savez(os.path.join(OUTPUT_PATH, 'data'), full_frequency=full_frequency, avg_frequency=avg_frequency)
  np.savez(os.path.join(OUTPUT_PATH, 'labels'), labels=labels)

if __name__ == '__main__':
  main()
