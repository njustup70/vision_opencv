# 相机俯角标定
# 数据点平均有异常去from_file.py改各个阈值再跑一遍
# 跑完了把结果复制进attitude_info.yaml里
import numpy as np
from image_geometry import PinholeCameraModel
from get_cam_xangle_point import fit_plane_from_depth
from plan_PC_fit import timetest
import rclpy
from rclpy.node import Node
import time
import cv2
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import yaml
import threading
from rclpy.qos import qos_profile_sensor_data
from get_a_image import get_a_image

points_list = []
normal_list = []

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


def average_normal(d2c_r=None):
    all_points = len(normal_list)
    normal_array = np.array(normal_list)
    normal_array_original = normal_array.copy()
    average = np.mean(normal_array, axis=0)
    average /= np.linalg.norm(average)  # 单位化
    delta_angle = 0.5
    flag = 1
    while flag:
        normal_array = normal_array_original.copy()
        average_good = average.copy()
        cos_thresh = np.cos(np.deg2rad(delta_angle))
        angle_error = 0
        times = [0, 0]
        while average_good @ average < cos_thresh or times[0] == 0:
            average = average_good.copy()
            dots = normal_array @ average
            k = int(0.7 * len(dots))               # 取最近 70%
            idx = np.argsort(dots)[-k:]
            normal_array_good = normal_array[idx]   #先不删
            average_good = np.mean(normal_array_good, axis=0)
            average_good /= np.linalg.norm(average_good)
            times[0] += 1
            if times[0] > 100:   # 防止死循环
                angle_error = 1
                delta_angle += 0.1
                break
        if angle_error:
            continue

        average = average_good.copy()
        # 先删一次
        dots = normal_array @ average
        k = int(0.7 * len(dots))
        idx = np.argsort(dots)[-k:]
        normal_array = normal_array[idx]
        while average_good @ average < cos_thresh or times[1] == 0:
            average = average_good.copy()
            dots = normal_array @ average
            k = int(0.7 * len(dots))
            idx = np.argsort(dots)[-k:]
            normal_array = normal_array[idx] # 删
            if len(normal_array) < 20:      # 特征点过少时记得改阈值！！！
                break
            average_good = np.mean(normal_array, axis=0)
            average_good /= np.linalg.norm(average_good)
            times[1] += 1
            if times[1] > 100:  # 防止死循环
                break
        else:
            flag = 0
            break

        delta_angle += 0.1 # 放宽条件，继续迭代,以防过小时无结果
        if delta_angle > 5:
            print("Failed to converge average normal vector within reasonable angle threshold.")
            average = None
            average_good = None
            raise RuntimeError("Failed to converge average normal vector within reasonable angle threshold.")

    if d2c_r is not None:
        average_good = np.array(d2c_r).reshape(3,3) @ average_good.reshape(3,1)
        average_good = average_good.reshape(3,)
    print(f"Average normal vector: {average_good[0]:.6f},{average_good[1]:.6f},{average_good[2]:.6f}\nPoints used: {len(normal_array)} in all points {all_points}\n")
    print(f"Iterations without deletion: {times[0]}, with deletion: {times[1]}, delta_angle: {delta_angle}")
    with open("DepthCamera/xangle.txt", "a") as f:
        f.write(f"Average normal vector: {average_good[0]:.6f},{average_good[1]:.6f},{average_good[2]:.6f}\n")
        f.write(f"Points used: {len(normal_array)}\n")
    
    

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

class GetCameraXAngle(Node):
    def __init__(self):
        super().__init__('get_camera_xangle')
        self.info_msg = None
        self.get_logger().info('Waiting for /camera/depth/camera_info...')
        try:
            self.info_msg = self.wait_for_camera_info()
            self.get_logger().info('Loaded camera info from topic.')
        except TimeoutError:
            self.get_logger().warn('Timeout waiting for /camera/depth/camera_info, loading from YAML instead.')
        self.depth_camera = DepthCamera()
        self.depth_camera.loadCameraInfo(info_d=self.info_msg)
        self.create_subscription(Image, '/camera/depth/image_raw', self.depth_callback, qos_profile=qos_profile_sensor_data)
        self.create_subscription(Image, '/camera/color/image_raw', self.color_callback, qos_profile=qos_profile_sensor_data)
        self.get_logger().info('Waiting for camera_info and depth frames...')
        self.depth_img = None
        self.color_img = None
        self.final_img = None
        self.point = 0
        self.one_point_times = 0
        self.depth_ready = threading.Event()
        #self.point_list = [(159, 375),(196, 517),(302, 683),(422, 461),(455, 584),(538, 723),(588, 418),(625, 590),(671, 733),(758, 375),(807, 600),(851, 720),(914, 414),(974, 610),(1083, 733),(1030, 441)]
        self.point_list = points_list
        self.get_logger().info(f"Total points to process: {len(self.point_list)}")

        threading.Thread(target=self.working_process, daemon=True).start()


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
        self.depth_img = msg
        self.depth_ready.set()

    def working_process(self):
        with open("DepthCamera/xangle.txt", "a+") as f:
            if f.tell() == 0:
                f.write("# 相机俯角标定结果\n")
            f.write("\n" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + "\n")
        while not self.color_img:
            time.sleep(0.01)
        while True:
            if self.point >= len(self.point_list):
                break
            self.depth_ready.wait()
            msg = self.depth_img
            self.depth_ready.clear()
            cv2_color_img = self.depth_camera.bridge.imgmsg_to_cv2(self.color_img, desired_encoding='passthrough')
            cv2_depth_img = self.depth_camera.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough').astype(np.uint16)
            color_resized = cv2.resize(cv2_color_img, (cv2_depth_img.shape[1], cv2_depth_img.shape[0]), interpolation=cv2.INTER_LINEAR)
            u, v = self.point_list[self.point]
            self.one_point_times += 1
            if self.one_point_times >= 3:
                self.one_point_times = 0
                self.point += 1
            depth = cv2_depth_img[int(v), int(u)] / 1000.0  # 转换为米
            x,y,z = pix_to_cam(u, v, depth, self.depth_camera.model_d)
            try:
                result = fit_plane_from_depth(cv2_depth_img, self.depth_camera.model_d, u, v, depth_scale=1000.0)
                self.get_logger().info(f"point {self.point-int(not self.one_point_times)+1} time {(self.one_point_times+2)%3+1}: ({u},{v}) depth: {depth:.3f}m -> X:{x:.3f} Y:{y:.3f} Z:{z:.3f} -> normal: {result['seed_normal']}")
            except Exception as e:
                self.get_logger().warn(f"failed: {e}")
                self.final_img = color_resized
                continue
            seed_normal = result["seed_normal"]
            normal_list.append(seed_normal)
            with open("DepthCamera/xangle.txt", "a") as f:
                f.write(f"{seed_normal[0]:.6f},{seed_normal[1]:.6f},{seed_normal[2]:.6f}\n")
        try:
            average_normal(self.depth_camera.d2c_r)
        except RuntimeError as e:
            self.get_logger().error(str(e))
        rclpy.shutdown()
        

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        point = (x, y)
        img = param.copy()
        print(f"point chosen at ({x}, {y})")
        points_list.append(point)
        for point in points_list:
            cv2.circle(img, point, 5, (0, 0, 255), -1)
        num = len(points_list)
        cv2.putText(img, f"Point {num}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.imshow("Retrieved Image", img)

def main():
    img = get_a_image()
    cv2.namedWindow("Retrieved Image", cv2.WINDOW_NORMAL)
    if img is not None:
        cv2.imshow("Retrieved Image", img)
        cv2.putText(img, "Click to select points for calibration. Press any key when done.", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.setMouseCallback("Retrieved Image", mouse_callback, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    rclpy.init()
    node = GetCameraXAngle()
    rclpy.spin(node)
    node.destroy_node()

if __name__ == '__main__':
    main()
