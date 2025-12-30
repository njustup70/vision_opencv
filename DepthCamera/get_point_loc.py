from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import numpy as np
from image_geometry import PinholeCameraModel
import yaml
import deform_restore

import os
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"

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

def pix_to_cam(u, v, depth, model):
    ray = model.projectPixelTo3dRay((u, v))
    muit = 1.0 / ray[2]
    X = ray[0] * muit * depth
    Y = ray[1] * muit * depth
    Z = ray[2] * muit * depth # Z = depth
    return X, Y, Z

class DepthCamera:
    def __init__(self):
        self.bridge = CvBridge()
        self.model_d = PinholeCameraModel()
        self.model_c = PinholeCameraModel()

    def loadCameraInfo(self, info_d = None, info_c = None, info_d2c = None):
        if info_d is None:
            self.model_d.fromCameraInfo(load_camera_info('DepthCamera/depth_camera_info.yaml'))
        else:
            self.model_d.fromCameraInfo(info_d)

        if info_c is None:
            self.model_c.fromCameraInfo(load_camera_info('DepthCamera/color_camera_info.yaml'))
        else:
            self.model_c.fromCameraInfo(info_c)

        if info_d2c is None:
            with open('DepthCamera/depth_to_color_info.yaml', 'r') as f:
                data = yaml.safe_load(f)
            rot = data['depth_to_color_extrinsics']['rotation']['data']
            trans = data['depth_to_color_extrinsics']['translation']['data']
        else:
            rot = info_d2c['rotation']
            trans = info_d2c['translation']
        self.d2c_r = np.array(rot).reshape(3, 3)
        self.d2c_t = np.array(trans).reshape(3, 1)

if __name__ == '__main__':
    import rclpy
    from rclpy.node import Node
    import time
    import cv2

    class PixelToCamera(Node):
        def __init__(self):
            super().__init__('pixel_to_camera')
            self.info_msg = None
            self.get_logger().info('Waiting for /camera/depth/camera_info...')
            try:
                self.info_msg = self.wait_for_camera_info()
                self.get_logger().info('Loaded camera info from topic.')
            except TimeoutError:
                self.get_logger().warn('Timeout waiting for /camera/depth/camera_info, loading from YAML instead.')
            self.depth_camera = DepthCamera()
            self.depth_camera.loadCameraInfo(info_d=self.info_msg)
            self.create_subscription(Image, '/camera/depth/image_raw', self.depth_callback, 10)
            self.create_subscription(Image, '/camera/color/image_raw', self.color_callback, 10)
            self.get_logger().info('Waiting for camera_info and depth frames...')
            self.depth_img = None
            self.color_img = None
            self.points = [(200,400), (200,450), (300,400), (300,400)]
            self.click_point = 0

            cv2.namedWindow("Color Image")
            cv2.setMouseCallback("Color Image", self.mouse_callback)

        '''
        def info_init_callback(self, msg):
            if self.cameraInfoInit:
                return
            self.model_d.fromCameraInfo(msg)
            self.cameraInfoInit = True
        '''
        def wait_for_camera_info(self, timeout_sec=1.0):
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

        def color_callback(self, msg):
            self.color_img = msg
            #self.timetest()

        def depth_callback(self, msg):
            if self.color_img is None:
                return
            #self.timetest()
            self.depth_img = msg
            cv2_color_img = self.depth_camera.bridge.imgmsg_to_cv2(self.color_img, desired_encoding='passthrough')
            cv2_depth_img = self.depth_camera.bridge.imgmsg_to_cv2(self.depth_img, desired_encoding='passthrough').astype(np.uint16)
            color_resized = cv2.resize(cv2_color_img, (cv2_depth_img.shape[1], cv2_depth_img.shape[0]), interpolation=cv2.INTER_LINEAR)
            # 深度图叠加到彩色图，颜色表示深度
            depth_colored = cv2.applyColorMap(cv2.convertScaleAbs(cv2_depth_img, alpha=0.03), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(color_resized, 0.6, depth_colored, 0.4, 0)
            color_resized = overlay
            count = 0
            for (u, v) in self.points:
                if count >= self.click_point:
                    break
                depth = cv2_depth_img[int(v), int(u)] / 1000.0  # 转换为米
                #print(cv2_color_img.shape, color_resized.shape)
                cv2.circle(color_resized, (int(u), int(v)), 5, (65535,65535,0), -1) # 黄色圆点
                # 显示所有圆点坐标及深度
                x,y,z = pix_to_cam(u, v, depth, self.depth_camera.model_d)
                if z != 0:
                    cv2.putText(color_resized, f"({x:.3f},{y:.3f},{z:.3f})", (int(u)+10, int(v)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (65535,65535,0), 1)
                else:
                    cv2.putText(color_resized, f"(None)", (int(u)+10, int(v)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (65535,65535,0), 1)
                cv2.putText(color_resized, f"({u},{v})", (int(u)+10, int(v)+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (65535,65535,0), 1)
                count += 1
            cv2.imshow("Color Image", color_resized)
            cv2.waitKey(1)

        def mouse_callback(self, event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                depth = self.depth_camera.bridge.imgmsg_to_cv2(self.depth_img, desired_encoding='passthrough').astype(np.uint16)
                depth = depth[y, x] / 1000.0  # 转换为米
                if depth != 0:
                    x_3d, y_3d, z_3d = pix_to_cam(x, y, depth, self.depth_camera.model_d)
                    if self.click_point >= len(self.points):
                        self.click_point = 0
                    self.points[self.click_point] = (x, y)
                    self.get_logger().info(f"point chosen at ({x}, {y})")
                    if self.click_point >= len(self.points)-1:
                        needed_points = self.points
                        self.get_logger().info(f"All points chosen: {needed_points}")
                        # 顺时针将点排序
                        center = np.mean(needed_points, axis=0)
                        sorted_points = sorted(needed_points, key=lambda p: np.arctan2(p[1] - center[1], p[0] - center[0]))
                        sorted_3d_points = []
                        for (u, v) in sorted_points:
                            depth = self.depth_camera.bridge.imgmsg_to_cv2(self.depth_img, desired_encoding='passthrough').astype(np.uint16)
                            depth = depth[v, u] / 1000.0  # 转换为米
                            x3d, y3d, z3d = pix_to_cam(u, v, depth, self.depth_camera.model_d)
                            sorted_3d_points.append([x3d, y3d, z3d])
                        self.get_logger().info(f"3D Points: {[[round(x, 2) for x in point] for point in sorted_3d_points]}")
                        sorted_3d_points = np.array(sorted_3d_points, dtype=np.float32)
                        points_2d = deform_restore.trans3DToPlane(sorted_3d_points)
                        warped = deform_restore.ROIRestore(
                            self.depth_camera.bridge.imgmsg_to_cv2(self.color_img, desired_encoding='passthrough'),
                            points_2d
                        )
                        cv2.imshow("Warped Image", warped)
                        cv2.waitKey(0)
                        cv2.destroyWindow("Warped Image")
                    self.click_point += 1

                


def main():
    rclpy.init()
    node = PixelToCamera()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
