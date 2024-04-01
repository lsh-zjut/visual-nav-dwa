#!/usr/bin/env python
import rospy
from sensor_msgs.msg import CameraInfo, Image, RegionOfInterest
from cv_bridge import CvBridge
import tf
import numpy as np

class ROICoordinateCalculator:
    def __init__(self):
        rospy.init_node('roi_coordinate_calculator')

        # 订阅相机信息和ROI
        self.camera_info_sub = rospy.Subscriber('/front/camera_info', CameraInfo, self.camera_info_callback)
        self.roi_sub = rospy.Subscriber('/me5413/detected', RegionOfInterest, self.roi_callback)
        self.depth_sub = rospy.Subscriber('/front/depth/image_raw', Image, self.depth_callback)

        # 发布ROI坐标
        self.coord_pub = rospy.Publisher('/roi/coordinates', Image, queue_size=10)

        self.camera_info = None
        self.bridge = CvBridge()
        self.listener = tf.TransformListener()

    def camera_info_callback(self, data):
        self.camera_info = data

    def roi_callback(self, roi):
        self.roi = roi

    def depth_callback(self, data):
        if self.camera_info is None or self.roi is None:
            return

        try:
            # 从深度图像中获取深度信息
            cv_image = self.bridge.imgmsg_to_cv2(data, "32FC1")
            depth_array = np.array(cv_image, dtype=np.float32)

            # 计算ROI中心的深度
            center_x = self.roi.x_offset + self.roi.width / 2
            center_y = self.roi.y_offset + self.roi.height / 2
            depth = depth_array[int(center_y)][int(center_x)]

            # 确保深度值有效
            if np.isnan(depth) or np.isinf(depth):
                rospy.loginfo("Depth value is not valid.")
                return

            # 使用相机内参计算三维坐标
            fx = self.camera_info.K[0]
            fy = self.camera_info.K[4]
            cx = self.camera_info.K[2]
            cy = self.camera_info.K[5]

            X = (center_x - cx) * depth / fx
            Y = (center_y - cy) * depth / fy
            Z = depth

            # 打印结果
            rospy.loginfo(f"ROI Coordinates: X={X}, Y={Y}, Z={Z}")

            # 这里你可以添加将X, Y, Z发布出去的代码

        except Exception as e:
            rospy.logerr(e)

if __name__ == '__main__':
    roi_coord_calculator = ROICoordinateCalculator()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
