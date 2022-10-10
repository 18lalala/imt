import rospy
from std_msgs.msg import String
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointField
import numpy as np
import time
from std_msgs.msg import Header
from sensor_msgs import point_cloud2

import numpy as np
from sensor_msgs.msg import PointField
from sensor_msgs.msg import PointCloud2

import os

def talker(points):
    print(points.shape)
    pub = rospy.Publisher('puzek', PointCloud2, queue_size=1)
    # pub1 = rospy.Publisher('recons', PointCloud2, queue_size=1)
    rospy.init_node('node', anonymous=True)
    rate = rospy.Rate(10)

    while not rospy.is_shutdown():
        msg = PointCloud2()
        msg.header.stamp = rospy.Time().now()
        msg.header.frame_id = "test"

        if len(points.shape) == 3:
            msg.height = points.shape[1]
            msg.width = points.shape[0]
        else:
            msg.height = 1
            msg.width = len(points)

        msg.fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = msg.point_step * points.shape[0]
        msg.is_dense = False
        msg.data = np.asarray(points, dtype=np.float32).tostring()
        pub.publish(msg)


        # msg1 = PointCloud2()
        # msg1.header.stamp = rospy.Time().now()
        # msg1.header.frame_id = "test"

        # if len(recons.shape) == 3:
        #     msg1.height = recons.shape[1]
        #     msg1.width = recons.shape[0]
        # else:
        #     msg1.height = 1
        #     msg1.width = len(recons)

        # msg1.fields = [
        #     PointField('x', 0, PointField.FLOAT32, 1),
        #     PointField('y', 4, PointField.FLOAT32, 1),
        #     PointField('z', 8, PointField.FLOAT32, 1),]
        # msg1.is_bigendian = False
        # msg1.point_step = 12
        # msg1.row_step = msg1.point_step * recons.shape[0]
        # msg1.is_dense = False
        # msg1.data = np.asarray(recons, np.float32).tostring()

        # pub1.publish(msg1)

        rate.sleep()

    
if __name__ == '__main__':     
    pc_file = '/media/puzek/sda/dataset/semantic_kitti/dataset/sequences/01/velodyne/000010.bin'
    pc = np.fromfile(pc_file, dtype=np.float32).reshape(-1, 4)
    pt = np.ones((pc.shape[0], 3))
    pt = pc[:, :3]
    talker(pt)
