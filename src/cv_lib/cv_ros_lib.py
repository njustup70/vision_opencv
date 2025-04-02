import cv2
import zmq 
import multiprocessing.shared_memory as shm
import numpy as np
import time
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

class ImageReceive_t:
    """
    将共享内存图像转化成cv2图像
    :param socket: ZeroMQ连接地址
    """
    def __init__(self, socket="tcp://localhost:5555"):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.connect(socket)  # 连接到指定地址
        self._socket.setsockopt_string(zmq.SUBSCRIBE, "")  # 订阅所有消息（这一行是关键）
        self.shm=None
    def update(self, image: np.ndarray, content: dict = None):
        """
        将共享内存图像转化成cv2图像
        :param image: OpenCV图像对象 (np.ndarray)
        :param content: 附加的内容，如时间戳、坐标系等
        """
        #检查是否上线
        if self._socket.poll(0) == 0:
            time.sleep(0.5)
            print("socket not ready")
            return
        # 接收图像的消息
        message = self._socket.recv_json()
        shm_key = message['shm_key']
        shape = tuple(message['shape'])
        dtype = np.dtype(message['dtype'])
        print("shm_key:", shm_key)
        # 读取共享内存（零拷贝）
        shm_image = shm.SharedMemory(name=shm_key)
        image[:] = np.ndarray(shape, dtype=dtype, buffer=shm_image.buf)  # 用共享内存的数据替换掉图像
        
        # 显示图像
        cv2.imshow("image", image) 
        cv2.waitKey(1) 
    def __del__(self):
        # 关闭 ZeroMQ 套接字
        self._socket.close()
        # 关闭 ZeroMQ 上下文
        self._context.term()

def main():
    receive = ImageReceive_t()
    while True:
        # 创建一个空的图像对象，这里用一个全黑的图像作为示例
        image = np.zeros((480, 640, 3), dtype=np.uint8)  # 假设图像大小为480x640，3通道（RGB）
        receive.update(image)  # 更新图像为共享内存中的内容
        
if __name__ == "__main__":
    main()
