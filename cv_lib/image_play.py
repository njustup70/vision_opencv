import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import numpy as np
from image_geometry import PinholeCameraModel
import cv2
import time
import threading
def async_print(msg):
    threading.Thread(target=print, args=(msg,)).start()


class CameraToPixel(Node):
    def __init__(self):
        super().__init__('camera_to_pixel')
        self.cameraInfoInit = False
        self.bridge = CvBridge()
        self.model = PinholeCameraModel()
        self.PointToDepth = None
        self.depth_data = None
        self.timelist = [0] * 10
        self.timeListHead = 0
        self.create_subscription(CameraInfo, '/camera/color/camera_info', self.info_init_callback, 10)

        self.create_subscription(Image, '/camera/color/image_raw', self.color_callback, 10)

        self.get_logger().info('Waiting for point frames...')
        self.time_s = 0
        self.count = 0

    def info_init_callback(self, msg):
        if self.cameraInfoInit:
            return
        self.model.fromCameraInfo(msg)
        self.cameraInfoInit = True
        self.width = msg.width
        self.height = msg.height
        self.color_map = np.full((self.height, self.width, 3), np.nan, dtype=np.uint8)

    def color_callback(self, msg):
        time_tmp = time.time()
        color_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8').astype(np.uint8)
        # async_print('.')
        # print(".")
        # self.count += 1
        # if self.time_s != int(time.time()):
        #     async_print(f'FPS: {self.count}')
        #     self.count = 0
        #     self.time_s = int(time.time())
        cv2.imshow("Color Image", color_img)
        #async_print(f"Processing time: {time.time() - time_tmp}")
        cv2.waitKey(1)

def main():
    rclpy.init()
    node = CameraToPixel()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()