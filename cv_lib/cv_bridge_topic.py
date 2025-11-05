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

class ImageSubscribe_t(Node):
    """
    订阅ROS 2图像话题并转为OpenCV图像
    """
    def __init__(self, topic: str, node_name="image_subscriber"):
        super().__init__(node_name)
        self._topic = topic
        self._subscriber = self.create_subscription(
            Image,
            topic,
            self.image_callback,
            10
        )
        self._bridge = CvBridge()
        self.latest_image = None  # 存储最新的图像
        self.get_logger().info(f"图像订阅器初始化，话题: {topic}")

    def image_callback(self, msg: Image):
        """回调函数:将ROS图像转为OpenCV图像"""
        try:
            # 将ROS的Image消息转为OpenCV的np.ndarray
            cv_image = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.latest_image = cv_image
            self.get_logger().debug(f"接收到图像，尺寸: {cv_image.shape}")
            # 在这里可以添加图像处理逻辑
            # cv2.imshow("Received Image", cv_image)
            # cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"图像转换失败: {str(e)}")

class CompressedImagePublishe_t(Node):
    """发布压缩图像 (sensor_msgs/msg/CompressedImage)"""
    def __init__(self, topic: str, node_name="compressed_image_publisher"):
        super().__init__(node_name)
        self._publisher = self.create_publisher(CompressedImage, topic, 10)
        self.get_logger().info(f"压缩图像发布器启动，话题: {topic}")

    def publish_image(self, image: np.ndarray, format="jpeg", quality=90):
        """发布OpenCV图像为压缩图像消息"""
        if format == "jpeg":
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            msg_format = "jpeg"
        elif format == "png":
            encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]
            msg_format = "png"
        else:
            self.get_logger().error("不支持的压缩格式")
            return

        result, data = cv2.imencode(f".{format}", image, encode_param)
        if not result:
            self.get_logger().error("图像编码失败")
            return

        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.format = msg_format
        msg.data = data.tobytes()

        self._publisher.publish(msg)
        self.get_logger().debug(f"发布压缩图像，尺寸: {image.shape}")


class CompressedImageSubscribe_t(Node):
    """订阅压缩图像 (sensor_msgs/msg/CompressedImage)"""
    def __init__(self, topic: str, node_name="compressed_image_subscriber"):
        super().__init__(node_name)
        self._subscriber = self.create_subscription(
            CompressedImage,
            topic,
            self.image_callback,
            10
        )
        self.latest_image = None
        self.get_logger().info(f"压缩图像订阅器启动，话题: {topic}")

    def image_callback(self, msg: CompressedImage):
        """回调:将压缩图像解码为OpenCV图像"""
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if cv_image is not None:
                self.latest_image = cv_image
                self.get_logger().debug(f"接收压缩图像，尺寸: {cv_image.shape}")
            else:
                self.get_logger().warning("压缩图像解码失败")
        except Exception as e:
            self.get_logger().error(f"压缩图像接收失败: {str(e)}")

