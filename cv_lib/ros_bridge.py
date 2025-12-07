import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import numpy as np
from typing import Dict,Callable
from rclpy.subscription import Subscription
from rclpy.publisher import Publisher
from rclpy.qos import qos_profile_sensor_data
class subinformation:
    def __init__(self,sub,callback):
        #判断callback 是否为接受opencv图像的函数
        assert callable(callback),"callback must be callable"
        self.callback:Callable[[np.ndarray]]=callback
        self.sub:Subscription=sub
class ImageBridge_t(Node):
    """
    同时支持未压缩图像和压缩图像的发布与订阅
    """
    def __init__(self, node_name="image_bridge"):
        super().__init__(node_name)
        self.image_publishers:Dict[str,Publisher]={}
        # self.image_subscribers:Dict[str,tuple[Subscription,bool]]={}
        self.image_subscribers:Dict[str,subinformation]={}
    def publish_image(self, topic:str,image: np.ndarray):
        """
        发布未压缩图像
        :param image: OpenCV图像对象 (np.ndarray)
        """
        #判断是不是openCV图像
        assert isinstance(image,np.ndarray),"image must be np.ndarray"
        if topic not in self.image_publishers:
            self.image_publishers[topic]=self.create_publisher(Image, topic, 10)
        bridge = CvBridge()
        try:
            msg = bridge.cv2_to_imgmsg(image, encoding="bgr8")
            msg.header.stamp = self.get_clock().now().to_msg()
            self.image_publishers[topic].publish(msg)
            self.get_logger().debug(f"图像发布成功，尺寸: {image.shape}")
        except Exception as e:
            self.get_logger().error(f"图像发布失败: {str(e)}")
    def create_sub(self, topic:str, callback):
        if topic not in self.image_subscribers:
            sub=self.create_subscription(
                Image,
                topic,
                lambda msg, t=topic: self._ros_callback(msg, t),
                qos_profile=qos_profile_sensor_data
            )
            self.image_subscribers[topic]=subinformation(sub,callback)
    def create_compressed_sub(self, topic:str, callback):
        if topic not in self.image_subscribers:
            sub=self.create_subscription(
                CompressedImage,
                topic,
                lambda msg, t=topic: self._ros_callback(msg, t),
                10
            )
            self.image_subscribers[topic]=subinformation(sub,callback)
    def _ros_callback(self,msg,topic:str):
        cvbridge= CvBridge()
        image: np.ndarray=np.array([])
         #判断消息类型
        if isinstance(msg,Image):
            try:
                # 将ROS的Image消息转为OpenCV的np.ndarray
                cv_image = cvbridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                image=cv_image
            except Exception as e:
                self.get_logger().error(f"图像转换失败: {str(e)}")
                return
        elif isinstance(msg,CompressedImage):
            try:
                np_arr = np.frombuffer(msg.data, np.uint8)
                cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if cv_image is not None:
                    image=cv_image
                else:
                    self.get_logger().warning("压缩图像解码失败")
                    return
            except Exception as e:
                self.get_logger().error(f"压缩图像接收失败: {str(e)}")
        self.image_subscribers[topic].callback(image)
def image_show(image: np.ndarray):
    """简单的图像显示回调函数"""
    cv2.imshow("Received Image", image)
    cv2.waitKey(1) 
def main():
    rclpy.init()
    node = ImageBridge_t()
    node.create_sub("test_image", image_show)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Joy Teleop Node stopped by user.")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()