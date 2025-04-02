import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import imutils
import sys
import cv_lib.cv_ros_lib
class aruco_detect_node(Node):
    def __init__(self):
        super().__init__('aruco_detect_node')
        self._bridge = CvBridge()
        self.declare_parameter('image_topic', '/camera/color/image_raw')
        self.declare_parameter('use_compression', False)
        self._use_compression = self.get_parameter('use_compression').value
        self._handle_pipe=[]
        self._global_cotent = {}    
        self._image_subscriber = self.create_subscription(
            Image,
            self.get_parameter('image_topic').value,
            self.image_callback,
            10
        )
        self._handle_pipe.append(cv_lib.cv_ros_lib.ImagePublish_t(self, "/camera/image", 10))
    def image_callback(self, msg:Image):
        if self._use_compression:
            # 使用压缩格式
            image = self._bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="passthrough")
        else:
            # 使用未压缩格式
            image = self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        # 处理图像
        self._global_cotent['header'] = msg.header
        for pipe in self._handle_pipe:
            pipe.update(image,self._global_cotent)
def main ():
    rclpy.init()
    node = aruco_detect_node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
if __name__ == '__main__':
    main()