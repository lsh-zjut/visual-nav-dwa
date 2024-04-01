#!/usr/bin/env python
import math
import rospy
import tf
from sensor_msgs.msg import CameraInfo, Image
from vision_msgs.msg import Detection2D
from geometry_msgs.msg import PoseStamped, PointStamped
from std_msgs.msg import String
from cv_bridge import CvBridge
import numpy as np

class ROICoordinateCalculator:
    def __init__(self):
        rospy.init_node('roi_coordinate_calculator')

        self.listener = tf.TransformListener()

        self.camera_info_sub = rospy.Subscriber('/front/rgb/camera_info', CameraInfo, self.camera_info_callback)
        self.depth_image_sub = rospy.Subscriber('/front/depth/image_raw', Image, self.depth_callback)
        self.detection_sub = rospy.Subscriber('/me5413/detected', Detection2D, self.detection_callback)
        
        self.goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)
        self.goal_name_pub = rospy.Publisher('/rviz_panel/goal_name', String, queue_size=10)
        self.done_pub = rospy.Publisher('/me5413/done', String, queue_size=10)

        self.bridge = CvBridge()
        self.camera_info = None
        self.cv_depth_image = None

    def camera_info_callback(self, data):
        self.camera_info = data

    def depth_callback(self, data):
        # 使用"32FC1"编码转换深度图像，直接得到以米为单位的浮点数格式
        self.cv_depth_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="32FC1")

    def detection_callback(self, detection_msg):
        if self.camera_info is None or self.cv_depth_image is None:
            rospy.loginfo("Waiting for camera info and depth image...")
            return
        
        if detection_msg.bbox.size_x == 0 or detection_msg.bbox.size_y == 0:
            return

        rospy.loginfo("Detection found.")
        bbox = detection_msg.bbox
        center = bbox.center
        
        # 直接获取以米为单位的深度值
        depth = self.cv_depth_image[int(center.y), int(center.x)]
        
        if np.isnan(depth) or np.isinf(depth):
            rospy.loginfo("Invalid depth value.")
            return
        
        fx = self.camera_info.K[0]
        fy = self.camera_info.K[4]
        cx = self.camera_info.K[2]
        cy = self.camera_info.K[5]
        
        X = (center.x - cx) * depth / fx
        Y = (center.y - cy) * depth / fy
        Z = depth - 0.2  
        # TODO: 请根据实际情况调整Z值，使得机器人能够在目标前停下
        # The specific method is depth subtract 0.3
        # Then, check the point availability
        # If the point is available, publish the goal pose
        # If not, subtract 0.05 from the depth and check again
        # Repeat until the point is available

        # Transform the point to the map frame before cheking and publishing
        # Subscribe the /move_base/global_costmap/costmap so that you can get the availability of the point

        point_stamped = PointStamped()
        current_time = rospy.Time.now()
        point_stamped.header.stamp = current_time
        point_stamped.header.frame_id = "front_frame_optical"
        point_stamped.point.x = X
        point_stamped.point.y = Y
        point_stamped.point.z = Z

        # 尝试从相机坐标系转换到地图坐标系
        try:
            self.listener.waitForTransform("map", point_stamped.header.frame_id, current_time, rospy.Duration(4.0))
            map_point = self.listener.transformPoint("map", point_stamped)

            goal_pose = PoseStamped()
            goal_pose.header.stamp = current_time
            goal_pose.header.frame_id = "map"
            goal_pose.pose.position.x = map_point.point.x
            goal_pose.pose.position.y = map_point.point.y
            goal_pose.pose.position.z = map_point.point.z
            goal_pose.pose.orientation.w = 1.0

            rospy.loginfo("Target position successfully calculated: x={}, y={}, z={}".format(goal_pose.pose.position.x, goal_pose.pose.position.y, goal_pose.pose.position.z))

            self.goal_pub.publish(goal_pose)
            self.done_pub.publish(String("true"))

            return
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logerr("TF error when converting point: %s", e)

if __name__ == '__main__':
    node = ROICoordinateCalculator()
    rospy.spin()
