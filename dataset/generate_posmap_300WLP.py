"""Generate uv position map of 300W_LP dataset.

Reference: https://github.com/YadiraF/face3d/blob/master/examples/8_generate_posmap_300WLP.py
"""
import os

import numpy as np

import face3d
from face3d.morphable_model import MorphabelModel


def process_uv(uv_coords, uv_h=256, uv_w=256):
  #TODO(minoring): Findout what it does!. Make clear shape of uv coords
  uv_coords[:, 0] = uv_coords[:, 0] * (uv_w - 1)
  uv_coords[:, 1] = uv_coords[:, 1] * (uv_h - 1)
  uv_coords[:, 1] = uv_h - uv_coords[:, 1] - 1
  uv_coords = np.hstack((uv_coords, np.zeros((uv_coords[0], 1)))) # Add z
  return uv_coords


if __name__ == '__main__':
  save_folder = 'results/posmap_300WLP'
  if not os.path.exists(save_folder):
    os.mkdir(save_folder)
  # Set parameters
  uv_h = 256
  uv_w = 256
  image_h = 256
  image_w = 256

  # Load uv coords
  uv_coords = face3d.morphable_model.load.load_uv_coords(
      'Data/BFM/Out/BFM_UV.mat')
  uc_coords = process_uv(uv_coords, uv_h, uv_w)

  # Load BFM
  bfm = MorphabelModel('Data/BFM/Out/BFM.mat')
