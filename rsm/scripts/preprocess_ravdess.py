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

"""Modules for preparing RAVDESS as input dataset.

It reads from RAVDESS video files and finally writes
the frames and label information as a tf.Example in a tfrecords file.

  Sample usage:
    python preprocess_ravdess.py --data_dir=PATH_TO_AFFNIST_DIRECTORY
      --crop_size=700 --resize=87
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
from os.path import isfile, join
from collections import OrderedDict

import cv2
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.transform import resize

from tqdm import tqdm
import numpy as np
import scipy.io as spio
import tensorflow as tf

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('data_dir', './data/ravdess', 'Directory for storing input data')
tf.flags.DEFINE_string('file_type', 'h5', 'Filetype: tfrecords, h5, npz.')
tf.flags.DEFINE_integer('crop_size', 700, 'Size of the center crop.')
tf.flags.DEFINE_integer('image_size', 87, 'Final image size.')
tf.flags.DEFINE_integer('max_shard', 0,
                        'Maximum number of examples in each file.')


def get_files(dirname):
  dir_list = os.listdir(dirname)
  all_files = list()

  for entry in dir_list:
    full_path = os.path.join(dirname, entry)

    if os.path.isdir(full_path):
      all_files = all_files + get_files(full_path)
    else:
      all_files.append(full_path)

  return all_files

def video_to_array(filepath):
  cap = cv2.VideoCapture(filepath)
  num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  channel = 3

  frame_buffer = np.empty((num_frames, height, width, channel), dtype=np.float32)

  frame_num = 0
  returned = True

  while (frame_num < num_frames  and returned):
    returned, frame = cap.read()
    if frame is not None:
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      frame = frame.astype(np.float32)
      frame = frame / 255.0

      if np.sum(frame) > 0.0:
        frame_buffer[frame_num] = frame
    frame_num += 1

  cap.release()

  return frame_buffer

def crop_center(img,cropx,cropy):
  y, x, c = img.shape
  startx = x // 2 - (cropx // 2)
  starty = y // 2 - (cropy // 2)
  return img[starty:starty + cropy, startx:startx + cropx]


def int64_feature(value):
  """Casts value to a TensorFlow int64 feature list."""
  if FLAGS.file_type == 'tfrecords':
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
  return value


def bytes_feature(value):
  """Casts value to a TensorFlow bytes feature list."""
  if FLAGS.file_type == 'tfrecords':
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def write_sharded_dataset(data, filename, max_shard, crop_size, image_size, file_type):
    num_videos = len(data)
    num_shards = int(np.ceil(num_videos / FLAGS.max_shard))

    sharded_data = dict.fromkeys(range(num_shards))

    for i in range(num_shards):
      start = i * max_shard
      end = (i + 1) * max_shard

      sharded_data[i] = data[start:end]
      sharded_filename = (filename + '-{0}').format(i)

      if not os.path.exists(sharded_filename):
        write_dataset(sharded_data[i], sharded_filename, crop_size, image_size, file_type)


def write_dataset(data, filename, crop_size, image_size, file_type):
  writer = None
  output_data = []
  progress = tqdm(total=len(data))

  if file_type == 'tfrecords':
    writer = tf.python_io.TFRecordWriter(filename)

  for record in data:

    # Convert video into an array of image frames
    video = video_to_array(record['filepath'])

    feature = {}
    num_frames = video.shape[0]
    depth = 3

    # Define feature
    # feature = {
    #     'num_frames': int64_feature(num_frames),
    #     'height': int64_feature(image_size),
    #     'width': int64_feature(image_size),
    #     'depth': int64_feature(depth)
    # }

    # Attach identifiers to the feature dict
    feature['identifiers'] = record['identifiers']

    # Preprocess individal frames in the video
    processed_video = np.empty((num_frames, image_size, image_size, depth))

    for frame_idx in range(num_frames):
      cropped_frame = crop_center(video[frame_idx], crop_size, crop_size)
      resized_frame = resize(cropped_frame, (image_size, image_size), anti_aliasing=True)
      processed_video[frame_idx] = resized_frame

      if file_type == 'tfrecords':
        path = 'blob/' + str(frame_idx)
        frame_raw = resized_frame.tostring()
        feature[path] = bytes_feature(frame_raw)

    if file_type != 'tfrecords':
      feature['video'] = processed_video
      output_data.append(feature)

    # Write sequence to TFRecord
    if file_type == 'tfrecords':
      example = tf.train.Example(features=tf.train.Features(feature=feature))
      writer.write(example.SerializeToString())

    progress.update(1)

  progress.close()

  if output_data:
    with h5py.File(filename, 'w') as hf:
      group = hf.create_group('data')

      dt = h5py.special_dtype(vlen=np.dtype('float32'))
      features = hf.create_dataset('features', (len(output_data),), dtype=dt, compression='lzf')
      labels = hf.create_dataset('labels', (len(output_data),), dtype=dt, compression='lzf')

      for i, feature in enumerate(output_data):
        group.create_dataset(name='video_' + str(i),
                             data=feature['video'],
                             compression='lzf')

        group.create_dataset(name='identifiers_' + str(i),
                             data=list(feature['identifiers'].values()),
                             compression='lzf')

  if writer is not None:
    writer.close()


def read_data(filepaths):
  data = []

  for file in filepaths:
    if 'mp4' in file:
      filename = os.path.basename(file)
      identifiers = filename[:-4].split('-')

      identifiers_dict = OrderedDict()
      identifiers_dict['modality'] = int(identifiers[0])
      identifiers_dict['vocal_channel'] = int(identifiers[1])
      identifiers_dict['emotion'] = int(identifiers[2])
      identifiers_dict['emotional_intensity'] = int(identifiers[3])
      identifiers_dict['statement'] = int(identifiers[4])
      identifiers_dict['repetition'] = int(identifiers[5])
      identifiers_dict['actor'] = int(identifiers[6])

      record = {
          'filepath': file,
          'identifiers': identifiers_dict
      }

      data.append(record)

  random.shuffle(data)

  return data

def main(_):
  filepaths = get_files(FLAGS.data_dir)

  data = read_data(filepaths)

  filename = 'ravdess.' + FLAGS.file_type

  if FLAGS.max_shard > 0:
      filename = 'sharded_' + filename
  filepath = os.path.join(FLAGS.data_dir, filename)

  if FLAGS.max_shard > 0:
    write_sharded_dataset(data, filepath, FLAGS.max_shard, FLAGS.crop_size, FLAGS.image_size, FLAGS.file_type)
  else:
    write_dataset(data, filepath, FLAGS.crop_size, FLAGS.image_size, FLAGS.file_type)


if __name__ == '__main__':
  tf.app.run()
