import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import numpy as np
from image_geometry import PinholeCameraModel
import time

def pix_to_cam(u, v, depth, model):
    ray = model.projectPixelTo3dRay((u, v))
    muit = 1.0 / ray[2]
    X = ray[0] * muit * depth
    Y = ray[1] * muit * depth
    Z = ray[2] * muit * depth # Z = depth
    return X, Y, Z

class PixelToCamera(Node):
    def __init__(self):
        super().__init__('pixel_to_camera')
        self.cameraInfoInit = False
        self.bridge = CvBridge()
        self.model = PinholeCameraModel()
        self.timelist = [0] * 10
        self.timeListHead = 0
        self.create_subscription(CameraInfo, '/camera/depth/camera_info', self.info_init_callback, 10)
        self.create_subscription(Image, '/camera/depth/image_raw', self.depth_callback, 10)
        self.get_logger().info('Waiting for camera_info and depth frames...')

    def info_init_callback(self, msg):
        if self.cameraInfoInit:
            return
        self.model.fromCameraInfo(msg)
        self.cameraInfoInit = True

    def depth_callback(self, msg):
        self.timelist[self.timeListHead] = time.time()
        self.get_logger().info(f"time interval: {self.timelist[self.timeListHead]-self.timelist[(self.timeListHead-1)%10]:.3f} s")
        self.get_logger().info(f"10 time average interval: {(self.timelist[self.timeListHead]-self.timelist[(self.timeListHead+1)%10])/10:.3f} s")
        self.timeListHead = (self.timeListHead + 1) % 10
        self.get_logger().info('size: {}x{}, encoding: {}'.format(msg.width, msg.height, msg.encoding))
        depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough').astype(np.float32)
        #u,v = map(int, input("Enter pixel u and v coordinates separated by space: ").split())
        center_dep = pix_to_cam(msg.width // 2, msg.height // 2, depth_img[msg.height // 2, msg.width // 2] / 1000.0, self.model)[2]
        Xmax = -1111
        Xmin = -1111
        Ymax = -1111
        Ymin = -1111
        Zmax = -1111
        Zmin = -1111
        self.get_logger().info(f"center_dep[{center_dep:.3f}] m")
        for u in range(msg.width):
            for v in range(msg.height):
                depth = depth_img[v, u] / 1000.0  # Convert mm to meters
                if depth > center_dep - 0.01 and depth < center_dep + 0.01:
                    X, Y, Z = pix_to_cam(u, v, depth, self.model)
                    if X<Xmin or Xmin==-1111:
                        Xmin = X
                    if X>Xmax:
                        Xmax = X
                    if Y<Ymin or Ymin==-1111:
                        Ymin = Y
                    if Y>Ymax:
                        Ymax = Y
                    if Z<Zmin or Zmin==-1111:
                        Zmin = Z
                    if Z>Zmax:
                        Zmax = Z
        self.get_logger().info(f"X range: [{Xmin:.3f}, {Xmax:.3f}] m")
        self.get_logger().info(f"Y range: [{Ymin:.3f}, {Ymax:.3f}] m")
        self.get_logger().info(f"Z range: [{Zmin:.3f}, {Zmax:.3f}] m")
        

        #self.get_logger().info(f"Pixel ({u},{v}) -> Camera ({X:.3f}, {Y:.3f}, {Z:.3f}) m")

def main():
    rclpy.init()
    node = PixelToCamera()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
