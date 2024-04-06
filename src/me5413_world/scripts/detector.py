#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
from std_msgs.msg import String
from vision_msgs.msg import Detection2D
import os

template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "3.png")
# template_path = "me5413_world/scripts/3.png"
firsttrack_list = []
firsttrack_list.append((0, 0, 106, 137))


class Detector(object):
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/front/rgb/image_raw", Image, self.input_img_callback)
        self.depth_image_sub = rospy.Subscriber('/front/depth/image_raw', Image, self.depth_callback)
        self.template_sub = rospy.Subscriber("/rviz_panel/goal_name", String, self.template_callback)

        # Template for matching, obtained from the first frame.
        self.template = None
        self.template_coords = firsttrack_list[0]  # Initial tracking coordinates (x, y, w, h)
        self.template_path = template_path
        self.done = False

        self.depth_image = None

        # Initialize a publisher for sending msg
        self.matrix_pub = rospy.Publisher("/me5413/student_matrix", String, queue_size=10)
        self.detected_pub = rospy.Publisher("/me5413/detected", Detection2D, queue_size=10)
        self.current_depth_pub = rospy.Publisher("/me5413/current_depth", Image, queue_size=10)

    def template_callback(self, data):
        if "box" not in data.data:
            self.template = None
            return

        try:
            # Read the template img from the path
            template_img = cv2.imread(self.template_path)
            if template_img is None:
                rospy.logerr("Failed to load template image from path: {}".format(self.template_path))
            else:
                rospy.loginfo("Template image loaded successfully from path: {}".format(self.template_path))
                x, y, w, h = self.template_coords
                self.template = template_img[y:y+h, x:x+w]

        except Exception as e:
            rospy.logerr("Error loading template image: {}".format(e))

    def depth_callback(self, data):
        self.depth_image = data

    def input_img_callback(self, data):
        if self.template is None:
            detection = Detection2D()
            detection.bbox.size_x = 0
            detection.bbox.size_y = 0
            detection.bbox.center.x = 0
            detection.bbox.center.y = 0
            detection.source_img = data
            self.detected_pub.publish(detection)
            return
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "8UC3")
            current_depth = self.depth_image
            self.detect(cv_image, current_depth)
        except CvBridgeError as e:
            print(e)

    def publish_detection(self, x, y, width, height, img, current_depth):
        detection = Detection2D()
        # Configure your Detection2D message
        # For example, setting the bounding box size and position
        detection.bbox.size_x = width
        detection.bbox.size_y = height
        detection.bbox.center.x = x + width // 2
        detection.bbox.center.y = y + height // 2
        # Convert OpenCV image to ROS image message
        try:
            ros_img = self.bridge.cv2_to_imgmsg(img, "bgr8")
        except CvBridgeError as e:
            print(e)
        # Attach image itself to the message as image_raw
        detection.source_img = ros_img
        # Publish the message
        self.detected_pub.publish(detection)
        self.current_depth_pub.publish(current_depth)
        self.matrix_pub.publish("A0285282X")

    def detect(self, image, current_depth):
        # camera: 512, 640
        # template: 106, 137

        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        processed_image = image_gray
        template_gray = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)
        processed_template = template_gray

        original_height, original_width = processed_template.shape[:2]
        scales = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
        max_match_val = -1
        best_scale = None
        best_max_loc = None

        for scale in scales:
            # Calculate the new width and height
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)

            # Adjust the template size
            resized_template = cv2.resize(processed_template, (new_width, new_height),
                                          interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC)

            # Use the new template to match the image
            result = cv2.matchTemplate(processed_image, resized_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            if max_val > max_match_val:
                max_match_val = max_val
                best_scale = scale
                best_max_loc = max_loc

        # Only the match with score > 0.75 is considered as a valid detection
        if max_match_val > 0.75:
            x_d, y_d = best_max_loc
            width = int(original_width * best_scale)
            height = int(original_height * best_scale)
        else:
            x_d, y_d, width, height = 0, 0, 0, 0

        self.publish_detection(x_d, y_d, width, height, image, current_depth)

        return image


def main():
    rospy.init_node('detector', anonymous=True)
    det = Detector()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
        det.stop_thread = True
        det.process_thread.join()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
