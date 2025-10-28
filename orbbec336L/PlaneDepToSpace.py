import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import numpy as np
from image_geometry import PinholeCameraModel
#import time
import yaml

def pix_to_cam(u, v, depth, model):
    ray = model.projectPixelTo3dRay((u, v))
    muit = 1.0 / ray[2]
    X = ray[0] * muit * depth
    Y = ray[1] * muit * depth
    Z = ray[2] * muit * depth # Z = depth
    return X, Y, Z

def load_camera_info(yaml_path: str) -> CameraInfo:
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    msg = CameraInfo()
    msg.width  = data['image_width']
    msg.height = data['image_height']
    msg.distortion_model = data['distortion_model']
    msg.header.frame_id = data.get('header', {}).get('frame_id', '')
    def to_float_list(x):
        if isinstance(x, str):
            x = x.strip('[]').replace(',', ' ').split()
        return [float(i) for i in x]
    msg.d = to_float_list(data['distortion_coefficients']['data'])
    msg.k = to_float_list(data['camera_matrix']['data'])
    msg.r = to_float_list(data['rectification_matrix']['data'])
    msg.p = to_float_list(data['projection_matrix']['data'])
    msg.binning_x = data.get('binning_x', 0)
    msg.binning_y = data.get('binning_y', 0)
    roi = data.get('roi', {})
    msg.roi.x_offset = roi.get('x_offset', 0)
    msg.roi.y_offset = roi.get('y_offset', 0)
    msg.roi.height = roi.get('height', 0)
    msg.roi.width = roi.get('width', 0)
    msg.roi.do_rectify = roi.get('do_rectify', False)
    return msg

class PixelToCamera(Node):
    def __init__(self):
        super().__init__('pixel_to_camera')
        self.cameraInfoInit = False
        self.bridge = CvBridge()
        self.model = PinholeCameraModel()
        #self.timelist = [0] * 10
        #self.timeListHead = 0
        #self.create_subscription(CameraInfo, '/camera/depth/camera_info', self.info_init_callback, 10)
        #self.model.fromCameraInfo(load_camera_info('orbbec336L/depth_camera_info.yaml'))
        #等待 camera_info
        self.get_logger().info('Waiting for /camera/depth/camera_info...')
        try:
            msg = self.wait_for_camera_info(timeout_sec=3.0)
            self.model.fromCameraInfo(msg)
            self.get_logger().info('Loaded camera info from topic.')
        except TimeoutError:
            self.get_logger().warn('Timeout waiting for /camera/depth/camera_info, loading from YAML instead.')
            msg = load_camera_info('orbbec336L/depth_camera_info.yaml')
            self.model.fromCameraInfo(msg)
            self.get_logger().info('Loaded camera info from YAML.')
        self.create_subscription(Image, '/camera/depth/image_raw', self.depth_callback, 10)
        self.get_logger().info('Waiting for camera_info and depth frames...')
    '''
    def info_init_callback(self, msg):
        if self.cameraInfoInit:
            return
        self.model.fromCameraInfo(msg)
        self.cameraInfoInit = True
    '''
    def wait_for_camera_info(self, timeout_sec=3.0):
        #阻塞等待一次 /camera/depth/camera_info 消息
        future = rclpy.task.Future()
        def callback(msg):
            if not future.done():
                future.set_result(msg)
        self.create_subscription(CameraInfo, '/camera/depth/camera_info', callback, 10)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_sec)
        if not future.done():
            raise TimeoutError("CameraInfo timeout")
        return future.result()

    def depth_callback(self, msg):
        if self.model.K is None:
            return
        #self.timelist[self.timeListHead] = time.time()
        #self.get_logger().info(f"time interval: {self.timelist[self.timeListHead]-self.timelist[(self.timeListHead-1)%10]:.3f} s")
        #self.get_logger().info(f"10 time average interval: {(self.timelist[self.timeListHead]-self.timelist[(self.timeListHead+1)%10])/10:.3f} s")
        #self.timeListHead = (self.timeListHead + 1) % 10
        #self.get_logger().info('size: {}x{}, encoding: {}'.format(msg.width, msg.height, msg.encoding))
        depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough').astype(np.float32)
        #u,v = map(int, input("Enter pixel u and v coordinates separated by space: ").split())
        center_dep = pix_to_cam(msg.width // 2, msg.height // 2, depth_img[msg.height // 2, msg.width // 2] / 1000.0, self.model)[2]
        self.get_logger().info(f"center_dep[{center_dep:.3f}] m")
        
def main():
    rclpy.init()
    node = PixelToCamera()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
