"""Downloads the RAVDESS Video Dataset."""

import os
import sys
import zipfile
import argparse
import urllib.request


NUM_ACTORS = 24
DATA_TYPES = ['song', 'speech']
BASE_DOWNLOAD_URL = 'https://zenodo.org/record/1188976/files/{0}?download=1'


def main(dest):
  for datatype in DATA_TYPES:
    base_filename = 'Video_' + datatype.capitalize() + '_Actor_{0}.zip'

    for i in range(1, NUM_ACTORS + 1):
      actor_id = str(i).zfill(2)  # Add leading zeros
      filename = base_filename.format(actor_id)
      download_url = BASE_DOWNLOAD_URL.format(filename)
      filepath = os.path.join(dest, filename)

      if not os.path.exists(filepath):
        try:
          print('Downloading ', download_url, 'to', filepath)
          urllib.request.urlretrieve(download_url, filepath)
        except:
          print('Failed to download. Skipping ', download_url)

      extract_dirpath = os.path.join(dest, datatype)
      extract_filepath = os.path.join(extract_dirpath, 'Actor_' + actor_id)

      if not os.path.exists(extract_dirpath):
        os.makedirs(extract_dirpath)

      if not os.path.exists(extract_filepath):
        try:
          print('Extracting ', filepath, 'to', extract_filepath)
          zip_ref = zipfile.ZipFile(filepath, 'r')
          zip_ref.extractall(extract_dirpath)
          zip_ref.close()
        except:
          print('Failed to extract. Skipping ', filepath)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Command line options')
  parser.add_argument('--dest', type=str, dest='dest', default='../data')
  args = parser.parse_args(sys.argv[1:])

  main(**{k: v for (k, v) in vars(args).items() if v is not None})
