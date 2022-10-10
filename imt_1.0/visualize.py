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
import open3d as o3d

def R2rnt(R):
  R1 = R[:3, :]
  r = R1[:, :3].reshape(-1)
  t = R1[:, 3].reshape(-1)
  pose = np.hstack((r, t))
  return pose

def parse_poses(filename):
  """ read poses file with per-scan poses from given filename

      Returns
      -------
      list
          list of poses as 4x4 numpy arrays.
  """
  file = open(filename)

  poses = []

  for line in file:
    values = [float(v) for v in line.strip().split()]
    pose = np.zeros((4, 4))
    pose[0, 0:3] = values[0:3]
    pose[1, 0:3] = values[3:6]
    pose[2, 0:3] = values[6:9]
    pose[:3, 3] = values[9:12]
    pose[3, 3] = 1.0
    poses.append(pose)

  return poses

if __name__ == '__main__':
    pred_path = './result/result.txt'
    gt_path = '/media/puzek/sda/dataset/imt_data/dataset/sequences/Area_0/pose_rt.txt'
    point_folder = '/media/puzek/sda/dataset/semantic_kitti/dataset/sequences/07/velodyne/'
    gt = parse_poses(gt_path)
    preds = parse_poses(pred_path)

    pt_files = os.listdir(point_folder)
    pt_files.sort(key=lambda x: int(x[:6]))

    pt = pt_files[1]

    pts = np.fromfile((point_folder + pt), dtype=np.float32).reshape(-1, 4)

    pose_pred = preds[16]
    print(pose_pred)
    pose_gt = gt[16]
    print(pose_gt)

    points = np.ones((pts.shape))
    points[:, 0:3] = pts[:, 0:3]

    tpoints_gt = np.matmul(pose_gt, points.T).T
    tpoints_pred = np.matmul(pose_pred, points.T).T
    
    points_ori = np.zeros((pts.shape[0],3))
    points_ori = pts[:, :3]
    points_gt = np.zeros((pts.shape[0],3))
    points_gt = tpoints_gt[:, :3]
    points_pred = np.zeros((pts.shape[0],3))
    points_pred = tpoints_pred[:, :3]

    pc_ori = o3d.geometry.PointCloud()
    pc_ori.points = o3d.utility.Vector3dVector(points_ori)

    pc_gt = o3d.geometry.PointCloud()
    pc_gt.points = o3d.utility.Vector3dVector(points_gt)

    pc_pred = o3d.geometry.PointCloud()
    pc_pred.points = o3d.utility.Vector3dVector(points_pred)

    o3d.visualization.draw_geometries([pc_ori, pc_pred, pc_gt])


    
