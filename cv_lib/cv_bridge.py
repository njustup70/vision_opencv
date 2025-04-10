"""_summary_
@brief:
    1.ImagePublish_t: 将图像通过共享内存和zmq发布到ros2桥接节点
    2.ImageReceive_t: 将共享内存图像转化成cv2图像
"""
import cv2
import zmq 
import multiprocessing.shared_memory as shm
import numpy as np
import time
import os
class ImagePublish_t:
    """
        将opencv图像发布到ROS 2话题
        :param node: ROS 2节点对象
        :param topic: 话题名称
        :param queue_size: 消息队列大小
    """
    def __init__(self, topic:str,socket="tcp://localhost:5556"):
        self._topic = topic
        self._image_publish={}
        self.shm_name=None
        self.shm=None
        self.shm_size=None
        #zmq初始化
        ctx=zmq.Context()
        self.socket=ctx.socket(zmq.PUSH)
        self.socket.connect(socket)  # 绑定到指定地址
    def update(self, image:np.ndarray,content:dict=None):
        # self.node.get_logger().info(f"Publishing image to {self.topic}")
        """
        将 OpenCV 图像通过共享内存发布到 ROS 2 话题
        :param image: OpenCV 图像对象 (np.ndarray)
        :param content: 附加的内容，如时间戳、坐标系等
        """
        image_shape=    image.shape
        image_size=image.nbytes
        # 创建共享内存
        if self.shm_name==None or self.shm_size!=image_size or not os.path.exists(f"/dev/shm/{self.shm_name}"):
            if self.shm:
                try:
                    self.shm.close()
                    self.shm.unlink()
                except Exception as e:
                    print("Error closing shared memory:", e)
            self.shm=shm.SharedMemory(create=True, size=image_size)
            self.shm_name=self.shm.name
            self.shm_size=image_size
            print(f"创建共享内存{self.shm_name}，大小{self.shm_size}字节")
        # 将图像数据复制到共享内存
        np_shm_buf = np.ndarray(image_shape, dtype=image.dtype, buffer=self.shm.buf)
        np.copyto(np_shm_buf, image)
        self.socket.send_json({
            'shm_key': self.shm_name,
            'shape': image_shape,
            'dtype': str(image.dtype),
            'timestamp': time.time(),
            'topic': self._topic,
        })
class ImageReceive_t:
    """
    将共享内存图像转化成cv2图像
    :param socket: ZeroMQ连接地址
    """
    def __init__(self, socket="tcp://localhost:5555",print_latency=False,im_show=False):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PULL)
        self._socket.connect(socket)  # 连接到指定地址
        # self._socket.setsockopt_string(zmq.SUBSCRIBE, "")  # 订阅所有消息（这一行是关键）
        self.shm=None
        self.shm_key=None
        self.im_show=im_show
        self.print_latency=print_latency
    def update(self, image: np.ndarray, content: dict = None):
        """
        将共享内存图像转化成cv2图像
        :param image: OpenCV图像对象 (np.ndarray)
        :param content: 附加的内容，如时间戳、坐标系等
        """
        #检查是否上线
        # 接收图像的消息
        message = self._socket.recv_json()
        shm_key = message['shm_key']
        shape = tuple(message['shape'])
        dtype = np.dtype(message['dtype'])
        
        # 读取共享内存（零拷贝）
        if image.shape != shape:
            image.resize(shape, refcheck=False)
        if self.shm_key != shm_key:
            self.shm = shm.SharedMemory(name=shm_key)
            self.shm_key = shm_key
            print("shm_key:", shm_key)
        # image[:] = np.ndarray(shape, dtype=dtype, buffer=shm_image.buf)  # 用共享内存的数据替换掉图像
        image[:] = np.ndarray(shape, dtype=dtype, buffer=self.shm.buf)  # 用共享内存的数据替换掉图像
        # 显示图像
        if self.print_latency:
            #每隔60帧打印一次延迟
            if not hasattr(self, 'cnt'):
                self.cnt=0
            self.cnt+=1
            if self.cnt>60:
                now_time = time.time()
                start_time= message['timestamp']
                #转化成毫秒
                latency = (now_time-start_time)*1000
                #打印小数点后两位
                print(f"Latency: {latency:.2f} ms")
                self.cnt=0
        if self.im_show:
            # 显示图像
            cv2.imshow("image", image) 
            cv2.waitKey(1)
    def __del__(self):
        # 关闭 ZeroMQ 套接字
        self._socket.close()
        # 关闭共享内存
        self.shm.close()
        self.shm.unlink()
        # 关闭 ZeroMQ 上下文
        self._context.term()
            
def main():
    receive = ImageReceive_t(print_latency=True,im_show=True)
    pub1= ImagePublish_t("test_topic", socket="tcp://localhost:5556")
    pub2= ImagePublish_t("test_topic2")
    while True:
        # 创建一个空的图像对象，这里用一个全黑的图像作为示例
        image = np.zeros((1, 1, 3), dtype=np.uint8)  # 假设图像大小为480x640，3通道（RGB）
        receive.update(image)  # 更新图像为共享内存中的内容
        # 发布图像
        pub1.update(image)
        pub2.update(image)
        
if __name__ == "__main__":
    main()
