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
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt

# 25/09/2019, Shuai Sun

INPUT_PATH = 'data/ecg_xml'
OUTPUT_PATH = 'data/ecg'

filename = [os.path.join(INPUT_PATH, 'x2.xml')]

data_temp1 = np.zeros((12, 5000))
data_temp2 = np.zeros((12, 600))

for file in filename:
  tree = ET.parse(file)
  root = tree.getroot()

  for child in root:
    print(child.tag, ":", child.attrib, ':', child.text)
    for children in child:
      print(children.tag, ':', children.attrib, ':', children.text)
      for gradchild in children:
        print(gradchild.tag, ':', gradchild.attrib, gradchild.text)

  data_group = root[-1]

  # Full frequency = 500>
  data_list = data_group[0]
  counter = 0
  for j in range(12):
    item = data_list[j].text
    x = item.split(',')
    y = [float(p) for p in x]
    data_temp1[counter, :] = y
    counter += 1

  # AVG frequency = 500>
  data_list = data_group[1]
  counter = 0
  for j in range(12):
    item = data_list[j].text
    x = item.split(',')
    y = [float(p) for p in x]
    data_temp2[counter, :] = y
    counter += 1

data = np.append(data_temp1.transpose(), data_temp2.transpose(), 0)

print(data.shape)
train_data = np.zeros((5600, 12, 1))
train_data[:, :, 0] = data

label_data = np.zeros(5600)
label_data[5000:-1] = 1

fig = plt.figure()
plt.plot(train_data[:, 0, 0])
# plt.show()
fig.savefig('ecg.png', dpi=fig.dpi)

# np.save(os.path.join(OUTPUT_PATH, 'training_data'), train_data)
# np.save(os.path.join(OUTPUT_PATH, 'label_data'), label_data)
