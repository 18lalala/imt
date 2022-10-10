from sklearn.cluster import DBSCAN
import numpy as np
from bin2rosbag import talker
import open3d as o3d
import numpy as np
from copy import deepcopy
# dbscan = DBSCAN(eps=0.4, min_samples=50).fit_predict(a) # car multiscan
# dbscan = DBSCAN(eps=1.5, min_samples=15).fit_predict(a) # tree
# dbscan = DBSCAN(eps=2.5, min_samples=100).fit_predict(a) # building frame
# dbscan = DBSCAN(eps=0.8, min_samples=20).fit_predict(a) # car
 

def get_box(label, points, preds):
    pcd = o3d.geometry.PointCloud()
    pts = []
    for j in range(preds_b.shape[0]):
        if preds_b[j] == i:
            pts.append(points_b[j])
    pts = np.asarray(pts, dtype=np.float32)
    pcd.points = o3d.utility.Vector3dVector(pts)
    bb = pcd.get_axis_aligned_bounding_box()
    return bb


if __name__ == '__main__':
    preds = np.fromfile('/media/puzek/sda/dataset/semantic_kitti/dataset/sequences/08/labels/000018.label', dtype=np.uint32)
    pt = np.fromfile('/media/puzek/sda/dataset/semantic_kitti/dataset/sequences/08/velodyne/000018.bin', dtype=np.float32).reshape(-1, 4)
    # print('labels:', preds.shape, 'points:', points.shape)
    points = np.zeros((pt.shape[0], 3))
    points = pt[:, :3]
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>')
    a = []
    for i in range(points.shape[0]):
        if preds[i] == 81:
            a.append(points[i])

    a = np.array(a)

    # dbscan = DBSCAN(eps=0.3, min_samples=50).fit_predict(a) # 81
    # dbscan = DBSCAN(eps=2.6, min_samples=140).fit_predict(a) # building 50
    # dbscan = DBSCAN(eps=2.3, min_samples=80).fit_predict(a) # building 50 trainset
    dbscan = DBSCAN(eps=3, min_samples=5).fit_predict(a)


    preds = dbscan
    points = a
    print(preds.shape, points.shape)
    # print(set(preds))
    
    np.random.seed(10)
    colors_0 = np.random.randint(255, size=(200, 3))/255.
 
    pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points[:, :3])

    
    #显示预测结果
    pcd1 = deepcopy(pcd)
    # pcd1.translate((0, 5, 0)) #整体进行y轴方向平移5
    #为各个预测标签指定颜色
    # colors = colors_0[(preds+1).astype(np.uint8)]

    points_b = []
    preds_b = []
    for i in range(points.shape[0]):
        if preds[i] != -1:
            points_b.append(points[i])
            preds_b.append(preds[i])

    points_b = np.array(points_b)
    preds_b = np.array(preds_b)
    print(points_b.shape, preds_b.shape)
    pcd1.points = o3d.utility.Vector3dVector(points_b[:, :3])

    # colors = colors_0[(preds+1).astype(np.uint8)]
    colors = colors_0[preds_b.astype(np.uint8)]
    pcd1.colors = o3d.utility.Vector3dVector(colors[:, :3])

 
    # #显示预测结果和真实结果对比
    # pcd2 = deepcopy(pcd)
    # pcd2.translate((0, -5, 0)) #整体进行y轴方向平移-5
    # preds = preds.astype(np.uint8) == points[:, -1].astype(np.uint8)
    # #为各个预测标签指定颜色
    # colors = colors_0[preds.astype(np.uint8)]
    # pcd2.colors = o3d.utility.Vector3dVector(colors[:, :3])
 
    # 点云显示
    # o3d.visualization.draw_geometries([pcd1])

    #vis visualize

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    num_labels = np.max(preds_b)+1
    for i in range(num_labels):
        label = i
        box_pcd = get_box(label, points_b, preds_b)
        print(box_pcd.get_extent())
        vis.add_geometry(box_pcd)
    
    vis.add_geometry(pcd1)
    vis.run()
