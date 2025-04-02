import cv2
from cv_bridge import CvBridge
class ImagePublish_t:
    """
        将opencv图像发布到ROS 2话题
        :param node: ROS 2节点对象
        :param topic: 话题名称
        :param queue_size: 消息队列大小
    """
    def __init__(self,node, topic:str,queue_size:int=10):
        self._node = node
        self._topic = topic
        self._queue_size = queue_size
        self._image_publish={}
        # self._publisher=self._node.create_publisher(Image, self._topic, self._queue_size)
        self._bridge = CvBridge()
        # self._copy = copy
    def update(self, image:cv2.Mat,content:dict):
        # self.node.get_logger().info(f"Publishing image to {self.topic}")
        """
        将 OpenCV 图像发布到 ROS 2 话题
        :param image: OpenCV 图像对象 (np.ndarray)
        :param content: 附加的内容，如时间戳、坐标系等
        """
        # ros_image = self._bridge.cv2_to_imgmsg(image, encoding="bgr8")
        #默认复制传入图像的header
        # ros_image.header=content['header']
        # # 发布图像消息
        # self._publisher.publish(ros_image)
class ImageConver_t:
    """
        将ROS 2图像转换为OpenCV格式
        :param node: ROS 2节点对象
        :param topic: 话题名称
        :param queue_size: 消息队列大小
    """
    