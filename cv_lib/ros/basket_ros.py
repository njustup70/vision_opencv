"""_summary_
@brief:
    1.ImagePublish_t: 将图像通过 ROS 2 发布（支持未压缩图像）
    2.ImageReceive_t: 订阅 ROS 2 图像话题（支持未压缩/压缩图像）
"""
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import numpy as np
import time
import threading

class ImagePublish_t(Node):
    """
    将OpenCV图像发布到ROS 2话题(未压缩图像)
    :param topic: 话题名称
    :param queue_size: 消息队列大小
    """
    def __init__(self, topic: str, node_name="image_publisher"):
        super().__init__(node_name)
        self._topic = topic
        self._publisher = self.create_publisher(Image, topic, 10)
        self._bridge = CvBridge()
        self.get_logger().info(f"图像发布器初始化，话题: {topic}")
        
    def update(self, image: np.ndarray, content: dict = None):
        """
        将OpenCV图像发布到ROS 2话题
        :param image: OpenCV图像对象 (np.ndarray)
        :param content: 附加内容，如时间戳、坐标系等
        """
        try:
            msg = self._bridge.cv2_to_imgmsg(image, encoding="bgr8")
            if content and 'timestamp' in content:
                msg.header.stamp = self.get_clock().now().to_msg()
            self._publisher.publish(msg)
            self.get_logger().debug(f"图像发布成功，尺寸: {image.shape}")
        except Exception as e:
            self.get_logger().error(f"图像发布失败: {str(e)}")



