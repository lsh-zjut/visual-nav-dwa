#!usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from queue import Queue
from threading import Thread
from std_msgs.msg import String
from vision_msgs.msg import Detection2D

template_path = "/home/amin/ME5413_Final_Project/src/me5413_world/scripts/3.png"
firsttrack_list = []
firsttrack_list.append((0, 0, 106, 137))
    
# def preprocess_image(image):
#     image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(image_gray, (3, 3), 0)
#     equalized = cv2.equalizeHist(blurred)
#     binary_image = cv2.adaptiveThreshold(equalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
#                                     cv2.THRESH_BINARY, 11, 0)

#     return preprocessed_image

class Detector(object):
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/front/image_raw", Image, self.input_img_callback)
        self.template_sub = rospy.Subscriber("/rviz_panel/goal_name", String, self.template_callback)
        # Template for matching, obtained from the first frame.
        self.template = None
        self.template_coords = firsttrack_list[0]  # Initial tracking coordinates (x, y, w, h)
        self.template_path = template_path

        # Initialize a publisher for sending msg
        self.matrix_pub = rospy.Publisher("/me5413/student_matrix", String, queue_size=10)
        self.detected_pub = rospy.Publisher("/me5413/detected", Detection2D, queue_size=10)

    def template_callback(self, data):
        if "box" not in data.data:
            rospy.loginfo("Message does not contain 'box'. detection closing...")
            return
        try:
            # 加载模板图像
            template_img = cv2.imread(self.template_path)
            if template_img is None:
                rospy.logerr("Failed to load template image from path: {}".format(self.template_path))
            else:
                rospy.loginfo("Template image loaded successfully from path: {}".format(self.template_path))
                x, y, w, h = self.template_coords
                self.template = template_img[y:y+h, x:x+w]
                
        except Exception as e:
            rospy.logerr("Error loading template image: {}".format(e))

    def input_img_callback(self, data):
        if self.template is None:
            return
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "8UC3")
            self.detect(cv_image)
            # self.frame_queue.put(cv_image)
        except CvBridgeError as e:
            print(e)

    # def process_frames(self):
    #     while not self.stop_thread:
    #         if not self.frame_queue.empty():
    #             cv_image = self.frame_queue.get()
    #             if self.template is None:
    #                 # Initialize template with the first frame using initial coordinates.
    #                 self.prev = self.template_coords
    #                 self.prev_prev = self.prev
    #             else:
    #                 frame = self.detect(cv_image)
    #             # Publish the string message
    #             self.matrix_pub.publish("A0285282X")

    def publish_detection(self, x, y, width, height, img, publisher):
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
        publisher.publish(detection)

    def detect(self, image):
        # camera:
        # height: 512
        # width: 640

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
            # 计算新的尺寸
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)

            # 调整模板大小
            resized_template = cv2.resize(processed_template, (new_width, new_height), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC)

            # 使用调整后的模板进行匹配
            result = cv2.matchTemplate(processed_image, resized_template, cv2.TM_CCOEFF_NORMED)

            # 获取最大匹配值的位置
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            
            if max_val > max_match_val:
                max_match_val = max_val
                best_scale = scale
                best_max_loc = max_loc

        if max_match_val > 0.75:
            x_d, y_d = best_max_loc
            width = int(original_width * best_scale)
            height = int(original_height * best_scale)
        else:
            x_d, y_d, width, height = 0, 0, 0, 0

        self.publish_detection(x_d, y_d, width, height, image, self.detected_pub)

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
