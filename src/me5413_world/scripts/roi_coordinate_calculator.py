#!/usr/bin/env python
import rospy
import tf
from sensor_msgs.msg import CameraInfo, Image
from vision_msgs.msg import Detection2D
from geometry_msgs.msg import PoseStamped, PointStamped
from std_msgs.msg import String
from cv_bridge import CvBridge
import numpy as np

import rospy
import tf
from sensor_msgs.msg import CameraInfo, Image
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from vision_msgs.msg import Detection2D

class ROICoordinateCalculator:
    def __init__(self):
        rospy.init_node('roi_coordinate_calculator')

        self.listener = tf.TransformListener()

        self.camera_info_sub = rospy.Subscriber('/front/rgb/camera_info', CameraInfo, self.camera_info_callback)
        self.depth_image_sub = rospy.Subscriber('/front/depth/image_raw', Image, self.depth_callback)
        self.detection_sub = rospy.Subscriber('/me5413/detected', Detection2D, self.detection_callback)
        
        self.goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)
        self.goal_name_pub = rospy.Publisher('/rviz_panel/goal_name', String, queue_size=10)

        self.bridge = CvBridge()
        self.camera_info = None
        self.cv_depth_image = None

    def camera_info_callback(self, data):
        self.camera_info = data

    def depth_callback(self, data):
        self.cv_depth_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")

    def detection_callback(self, detection_msg):
        if self.camera_info is None or self.cv_depth_image is None:
            rospy.loginfo("等待相机信息和深度图像...")
            return

        bbox = detection_msg.bbox
        center = bbox.center

        # 获取相机内参矩阵
        fx = self.camera_info.K[0]
        fy = self.camera_info.K[4]
        cx = self.camera_info.K[2]
        cy = self.camera_info.K[5]

        # 获取ROI的像素坐标
        u = center.x
        v = center.y

        # 获取深度值
        depth_value = self.cv_depth_image[int(v)][int(u)]

        # 计算物体的三维坐标
        X = (u - cx) * depth_value / fx
        Y = (v - cy) * depth_value / fy

        # 发布物体坐标
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = "map"
        goal_pose.pose.position.x = X
        goal_pose.pose.position.y = Y
        goal_pose.pose.position.z = depth_value  # 使用深度值作为高度
        self.goal_pub.publish(goal_pose)

if __name__ == "__main__":
    roi_calculator = ROICoordinateCalculator()
    rospy.spin()
