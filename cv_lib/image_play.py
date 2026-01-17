import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.qos import qos_profile_sensor_data
import numpy as np
import cv2
import time

def get_FPS(timelist, timeListHead):
    time_now = time.time()
    timelist[timeListHead] = time_now
    timeListHead = (timeListHead + 1) % len(timelist)
    time_diff = time_now - timelist[timeListHead]
    if time_diff == 0:
        fps = 0.0
    else:
        fps = len(timelist) / time_diff
    return fps, timelist, timeListHead

class ImagePlay(Node):
    def __init__(self):
        super().__init__('image_play_node')
        self.cameraInfoInit = False
        self.bridge = CvBridge()
        cv2.namedWindow("Color Image", cv2.WINDOW_NORMAL)
        self.create_subscription(Image, '/camera/color/image_raw', self.color_callback, qos_profile=qos_profile_sensor_data)

        self.timelist = [0] * 10
        self.timeListHead = 0
        self.get_logger().info('Waiting for point frames...')

    def color_callback(self, msg):
        color_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8').astype(np.uint8)
        fps, self.timelist, self.timeListHead = get_FPS(self.timelist, self.timeListHead)
        cv2.putText(color_img, f'FPS: {fps:.2f}', (color_img.shape[1] - 100, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow("Color Image", color_img)
        cv2.waitKey(1)

def main():
    rclpy.init()
    node = ImagePlay()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()