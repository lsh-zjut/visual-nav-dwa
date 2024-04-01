#!/usr/bin/env python
import rospy
import tf
from sensor_msgs.msg import CameraInfo, Image
from vision_msgs.msg import Detection2D
from geometry_msgs.msg import PoseStamped, Point
from std_msgs.msg import String
from cv_bridge import CvBridge
import numpy as np

class ROICoordinateCalculator:
    def __init__(self):
        rospy.init_node('roi_coordinate_calculator')

        self.listener = tf.TransformListener()

        self.camera_info_sub = rospy.Subscriber('/front/camera_info', CameraInfo, self.camera_info_callback)
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
            rospy.loginfo("Waiting for camera info and depth image...")
            return

        bbox = detection_msg.bbox
        center = bbox.center
        
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
        Z = depth

        # 尝试从相机坐标系转换到地图坐标系
        try:
            now = rospy.Time.now()
            self.listener.waitForTransform("map", "front_camera_optical", now, rospy.Duration(4.0))
            (trans, rot) = self.listener.lookupTransform("map", "front_camera_optical", now)

            # 将检测到的点转换为地图坐标系下的点
            map_point = self.listener.transformPoint("map", Point(X, Y, Z))

            goal_pose = PoseStamped()
            goal_pose.header.stamp = rospy.Time.now()
            goal_pose.header.frame_id = "map"
            goal_pose.pose.position.x = map_point.x
            goal_pose.pose.position.y = map_point.y
            goal_pose.pose.position.z = map_point.z
            goal_pose.pose.orientation.w = 1.0

            self.goal_pub.publish(goal_pose)

            # 同时发布空字符串到/rviz_panel/goal_name
            self.goal_name_pub.publish("")
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logerr("TF error when converting point: %s", e)

if __name__ == '__main__':
    node = ROICoordinateCalculator()
    rospy.spin()
