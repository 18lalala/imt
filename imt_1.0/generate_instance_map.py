import numpy as np
import open3d as o3d
import os
from sklearn.cluster import DBSCAN
from copy import deepcopy

def normalize_pc(pc):
    centroid = np.mean(pc, axis=0)
    print(centroid)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    return centroid, m


def get_box(label, points, preds):
    pcd = o3d.geometry.PointCloud()
    pts = []
    for j in range(preds.shape[0]):
        if preds[j] == label:
            pts.append(points[j])
    pts = np.asarray(pts, dtype=np.float32)
    pcd.points = o3d.utility.Vector3dVector(pts)
    bb = pcd.get_axis_aligned_bounding_box()
    return bb

def process_map(label, points, preds, centroid, max):
    a = []
    fea = []
    for i in range(points.shape[0]):
        if preds[i] == label: # traffic-sign
            a.append(points[i])
    a = np.array(a)
    print(a.shape)
    if a.size > 0:
        if label == 81:
            dbscan = DBSCAN(eps=3, min_samples=1).fit_predict(a)
        elif label ==50:
            dbscan = DBSCAN(eps=2.3, min_samples=80).fit_predict(a)
        elif label ==80:
            dbscan = DBSCAN(eps=2, min_samples=1).fit_predict(a)
        preds = dbscan
        points = a
        points_b = []
        preds_b = []
        for i in range(points.shape[0]):
            if preds[i] != -1:
                points_b.append(points[i])
                preds_b.append(preds[i])
        points_b = np.array(points_b)
        preds_b = np.array(preds_b)
        num_labels = np.max(preds_b)+1
        for i in range(num_labels):
            label_db = i
            box_pcd = get_box(label_db, points_b, preds_b)
            if label == 50:
                print(box_pcd.get_extent().T)
                print(max)
            feature = np.hstack(((box_pcd.get_center().T - centroid) / max,box_pcd.get_extent().T / max,[label])) # 0:traffic-sign
            fea.append(feature)
        fea = np.array(fea, dtype=np.float32)
        return fea

if __name__ == '__main__':
    point_path = '/media/puzek/sda/dataset/semantic_kitti/dataset/sequences/07/velodyne/'
    label_path = '/media/puzek/sda/dataset/semantic_kitti/dataset/sequences/07/labels/'
    save_path = '/media/puzek/sda/dataset/semantic_kitti/dataset/sequences/07/instance/'
    # point_path = 'map/velodyne/'
    pt_files = os.listdir(point_path)
    pt_files.sort(key=lambda x: int(x[:6]))
    label_files = os.listdir(label_path)
    label_files.sort(key=lambda x: int(x[:6]))

    for i in range(len(pt_files)):
        print('processing  <<<<<<<<<<<' + pt_files[i])
        pt = np.fromfile((point_path+pt_files[i]), dtype=np.float32).reshape(-1, 4)
        preds = np.fromfile((label_path+label_files[i]), dtype=np.uint32)
        points = np.zeros((pt.shape[0], 3))
        points = pt[:, :3]
        cent, max = normalize_pc(points)

        fea1 = process_map(81, points, preds, cent, max)
        fea2 = process_map(50, points, preds, cent, max)
        fea3 = process_map(80, points, preds, cent, max)
        if fea1 is not None and fea3 is not None:
            feature = np.vstack((fea1, fea2, fea3))
        else:
            feature = fea2
        feature = np.array(feature, dtype=np.float32)
        print(feature.shape)
        feature.tofile(save_path+pt_files[i])
        print('done...')

    



    