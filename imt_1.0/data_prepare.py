import argparse
from distutils.command.install_lib import PYTHON_SOURCE_EXTENSION
import os
import yaml
import numpy as np
from collections import deque
import shutil
from numpy.linalg import inv
import struct
import time
import math

def parse_calibration(filename):
  """ read calibration file with given filename

      Returns
      -------
      dict
          Calibration matrices as 4x4 numpy arrays.
  """
  calib = {}

  calib_file = open(filename)
  for line in calib_file:
    key, content = line.strip().split(":")
    values = [float(v) for v in content.strip().split()]

    pose = np.zeros((4, 4))
    pose[0, 0:4] = values[0:4]
    pose[1, 0:4] = values[4:8]
    pose[2, 0:4] = values[8:12]
    pose[3, 3] = 1.0

    calib[key] = pose

  calib_file.close()

  return calib

def parse_poses(filename, calibration):
  """ read poses file with per-scan poses from given filename

      Returns
      -------
      list
          list of poses as 4x4 numpy arrays.
  """
  file = open(filename)

  poses = []

  Tr = calibration["Tr"]
  Tr_inv = inv(Tr)

  for line in file:
    values = [float(v) for v in line.strip().split()]

    pose = np.zeros((4, 4))
    pose[0, 0:4] = values[0:4]
    pose[1, 0:4] = values[4:8]
    pose[2, 0:4] = values[8:12]
    pose[3, 3] = 1.0

    poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))

  return poses

def rot_matrix2axis_angle(rot_matrix):
    '''
    transform rot_matrix to axis and angle + translation
    input matrix(4x4)
    output axis + theta + trans (1x7)
    '''
    rot = rot_matrix
    a = np.float32((rot[0, 0] + rot[1, 1] + rot[2, 2]-1) / 2.0)
    if a > 1:
      a = 1
    theta = math.acos(a)
    arr2 = np.array([rot[2, 1] - rot[1, 2], rot[0, 2] - rot[2, 0], rot[1, 0] - rot[0, 1]], dtype=np.float32)
    axis = (1/(2 * math.sin(theta) + 0.0001)) * arr2
    theta = np.array(theta, dtype=np.float32)
    axis = np.array(axis, dtype=np.float32)
    translation = rot[:3, 3].T
    transform = np.hstack((axis, theta, translation))
    return transform

def R2rnt(R):
  R1 = R[:3, :]
  r = R1[:, :3].reshape(-1)
  t = R1[:, 3].reshape(-1)
  pose = np.hstack((r, t))
  return pose
  
def process(index_s, index_t, pose, point_file, pose_aa, insp_s, insp_t, point_path):
  Point_s = np.fromfile(point_path + pt_files[index_s], dtype=np.float32).reshape(-1,7)
  Point_t = np.fromfile(point_path + pt_files[index_t], dtype=np.float32).reshape(-1,7)
  Pose_s = pose[index_s]
  Pose_t = pose[index_t]
  # print(Pose_t)
  pose_gt = np.matmul(inv(Pose_t), Pose_s)
  # print(pose_gt)
  # get pose_aa
  # transform = rot_matrix2axis_angle(pose_gt)
  # get pose 
  pose_rt = R2rnt(pose_gt)
  pose_aa.append(pose_rt)
  insp_s.append(Point_s)
  insp_t.append(Point_t)

if __name__ == '__main__':
    input_folder = '/media/puzek/sda/dataset/semantic_kitti/dataset/sequences/00/'
    point_path = input_folder + 'instance/'
    calibration = parse_calibration(os.path.join(input_folder, "calib.txt"))
    poses = parse_poses(os.path.join(input_folder, "poses.txt"), calibration)

    src_path = input_folder + 'source_rt/'
    if not os.path.exists(src_path):
        os.makedirs(src_path)
    tgt_path = input_folder + 'target_rt/'
    if not os.path.exists(tgt_path):
        os.makedirs(tgt_path)

    pt_files = os.listdir(point_path)
    pt_files.sort(key=lambda x: int(x[:6]))

    insp_s = []
    insp_t = []
    pose_aa = []

    for i in range(len(pt_files) - 35):
        index_T = np.array([i+3, i+6, i+9, i+12, i+15, i+18, i+21, i+24, i+27, i+30])
        for index in index_T:
          process(i, index, poses, pt_files, pose_aa, insp_s, insp_t, point_path)
    
    for i in range(len(pose_aa)):
        insp_s[i] = np.array(insp_s[i], dtype=np.float32)
        insp_t[i] = np.array(insp_t[i], dtype=np.float32)
        insp_s[i].tofile(src_path + '%06d.bin' % i)
        insp_t[i].tofile(tgt_path + '%06d.bin' % i)
        print(i , '___done')
      
    pose_aa = np.array(pose_aa, dtype=np.float32)
    print(pose_aa.shape)
    np.savetxt(input_folder + 'pose_rt.txt', pose_aa)
    
