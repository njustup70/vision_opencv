# 深度相机模型及附属基础功能
import yaml
from sensor_msgs.msg import CameraInfo
from image_geometry import PinholeCameraModel
import numpy as np
from cv_bridge import CvBridge
from rclpy.node import Node
import rclpy

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


class DepthCamera:
    def __init__(self):
        self.bridge = CvBridge()
        self.model_d = PinholeCameraModel()
        self.model_c = PinholeCameraModel()

    def loadCameraInfo(self, info_c = None, info_d = None, info_d2c = None):

        if info_c is None:
            self.model_c.fromCameraInfo(load_camera_info('DepthCamera/color_camera_info.yaml'))
        else:
            self.model_c.fromCameraInfo(info_c)

        if info_d is None:
            self.model_d.fromCameraInfo(load_camera_info('DepthCamera/depth_camera_info.yaml'))
        else:
            self.model_d.fromCameraInfo(info_d)

        if info_d2c is None:
            with open('DepthCamera/depth_to_color_info.yaml', 'r') as f:
                data = yaml.safe_load(f)
            rot = data['depth_to_color_extrinsics']['rotation']['data']
            trans = data['depth_to_color_extrinsics']['translation']['data']
        else:
            rot = info_d2c['rotation']
            trans = info_d2c['translation']
        self.d2c_r = np.array(rot).reshape(3, 3)
        self.d2c_t = np.array(trans).reshape(3,)

    def depth2points(self, depth_img):
        fx = self.model_d.fx()
        fy = self.model_d.fy()
        cx = self.model_d.cx()
        cy = self.model_d.cy()
        mask = (depth_img > 0)
        v, u = np.nonzero(mask)
        d = depth_img[v, u]
        x = (u - cx) * d / fx
        y = (v - cy) * d / fy
        z = d
        pc = np.column_stack((x, y, z))
        return pc
    
class DepthCamNode(Node):
    def __init__(self, nodename):
        super().__init__(nodename)
        self.depth_camera = DepthCamera()
        self.depth_image = None
        self.color_image = None
        self.pc = None
        self.info_msg = None
        
        self.get_logger().info('Waiting for /camera/depth/camera_info...')
        try:
            self.info_msg = self.wait_for_camera_info('/camera/depth/camera_info')
            self.get_logger().info('Loaded camera info from topic.')
        except TimeoutError:
            self.get_logger().warn('Timeout waiting for /camera/depth/camera_info, loading from YAML instead.')
        
        self.get_logger().info('Waiting for /camera/color/camera_info...')
        try:
            self.info_msg = self.wait_for_camera_info('/camera/color/camera_info')
            self.get_logger().info('Loaded camera info from topic.')
        except TimeoutError:
            self.get_logger().warn('Timeout waiting for /camera/color/camera_info, loading from YAML instead.')

        self.depth_camera.loadCameraInfo(info_d=self.info_msg, info_c=self.info_msg)

    def wait_for_camera_info(self, topic_name, timeout_sec=1.0):
        #阻塞等待一次消息
        future = rclpy.task.Future()
        def callback(msg):
            if not future.done():
                future.set_result(msg)
        self.create_subscription(CameraInfo, topic_name, callback, 10)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_sec)
        if not future.done():
            raise TimeoutError("CameraInfo timeout")
        return future.result()

def pix_to_cam(u, v, depth, model):
    ray = model.projectPixelTo3dRay((u, v))
    muit = 1.0 / ray[2]
    X = ray[0] * muit * depth
    Y = ray[1] * muit * depth
    Z = ray[2] * muit * depth # Z = depth
    return X, Y, Z