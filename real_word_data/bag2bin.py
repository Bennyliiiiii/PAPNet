import sys
import os
import rosbag
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import numpy as np

def read_and_save_rosbag(bag_file, topic_name, output_dir):
    bag = rosbag.Bag(bag_file)
    frame_count = 0
    for topic, msg, t in bag.read_messages(topics=[topic_name]):
        if msg._type == 'sensor_msgs/PointCloud2':
            points = []
            for point in pc2.read_points(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True):
                x, y, z, intensity = point
                points.append([x, y, z, intensity])
            bin_file = os.path.join(output_dir, f"{frame_count:06d}.bin")
            save_to_bin(points, bin_file)
            print(f"Saved {len(points)} points to {bin_file}")
            frame_count += 1
    bag.close()

def save_to_bin(points, bin_file):
    points_array = np.array(points, dtype=np.float32)
    points_array.tofile(bin_file)

if __name__ == "__main__":
    bag_file = '/home/auto/lrt/rosbag/lrtbag.bag'  # 修改为你的 ROS bag 文件路径
    topic_name = '/ouster/points'  # 修改为你的点云话题名称
    output_dir = '/home/auto/lrt'  # 输出 BIN 文件的目录

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    read_and_save_rosbag(bag_file, topic_name, output_dir)

