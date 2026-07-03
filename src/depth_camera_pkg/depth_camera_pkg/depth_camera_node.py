import rclpy
import rclpy.node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from DepthCamera import check_spearhead, rot_x, rot_y, rot_z, DepthCamNode, img_preprocess, get_yolo_result
import numpy as np
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
import threading
from tf2_ros import Buffer, TransformListener, TransformException
import json
import yaml
from ultralytics import YOLO
import subprocess
import cv2
from deform_restore import trans3DToPlane, ROIRestore
from shelf_outline_recog import process_image_and_detect_3x3
from check_KFS import check_red_blue
from light_preproc import preproc_hsv, clahe_on_v
from cv_bridge import CvBridge
import time

# 相机内参
K = np.array([
    [611.544189453125, 0.0, 636.7078857421875],
    [0.0, 611.3428955078125, 397.55560302734375],
    [0.0, 0.0, 1.0]
], dtype=np.float64)

def quat_to_rot(qx, qy, qz, qw):
    x, y, z, w = qx, qy, qz, qw
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    return np.array([
        [1 - 2*(yy+zz),     2*(xy-wz),     2*(xz+wy)],
        [    2*(xy+wz), 1 - 2*(xx+zz),     2*(yz-wx)],
        [    2*(xz-wy),     2*(yz+wx), 1 - 2*(xx+yy)],
    ], dtype=float)

def tfmsg_to_Rt(tf_msg):
    tr = tf_msg.transform.translation
    q  = tf_msg.transform.rotation
    t = np.array([tr.x, tr.y, tr.z], dtype=float)
    R = quat_to_rot(q.x, q.y, q.z, q.w)
    return R, t

def tfmsg_to_T(tf_msg):
    R, t = tfmsg_to_Rt(tf_msg)
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def T_to_rvec_tvec(T):
    T = np.asarray(T, dtype=np.float64)
    
    R = T[:3, :3].astype(np.float64)
    t = T[:3, 3].reshape(3, 1).astype(np.float64)

    rvec, _ = cv2.Rodrigues(R)
    tvec = t

    return rvec, tvec


def transform_points(T, points):
    points = np.asarray(points, dtype=np.float64)

    ones = np.ones((points.shape[0], 1), dtype=np.float64)
    points_h = np.hstack([points, ones])

    points_out_h = (T @ points_h.T).T

    return points_out_h[:, :3]

revc_USB = None
tevc_USB = None

class ImageNode(DepthCamNode):
    def __init__(self):
        super().__init__('ros_image_node')

        self.bridge = CvBridge()
        self.depcam_color_image = None
        self.depcam_depth_image = None
        self.depcam_color_hsv = None
        self.pc_need = 0
        self.vc = None
        
        self.yolo_model = YOLO('1.20.pt')
        self.yolo_names = self.yolo_model.names

        self.clahe_v = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        self.spearhead_need = threading.Event()
        self.YOLO_need = threading.Event()
        self.rb_check_need = threading.Event()

        self.grp = ReentrantCallbackGroup() # 组内可并发

        self.color_subscriber = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.depcam_color_callback,
            qos_profile_sensor_data,
            callback_group=self.grp)
        self.depth_subscriber = self.create_subscription(
            Image,
            '/camera/depth/image_raw',
            self.depcam_depth_callback,
            qos_profile_sensor_data,
            callback_group=self.grp)
        self.fuction_subscriber = self.create_subscription(
            String,
            '/update_exec_req',
            self.fuction_check,
            10,
            callback_group=self.grp)
        self.data_publisher = self.create_publisher(
            String,
            '/exec_result',
            10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)

        threading.Thread(target=self.spearhead_check_thread, daemon=True).start()
        threading.Thread(target=self.YOLO_detection_thread, daemon=True).start()
        threading.Thread(target=self.rb_check_thread, daemon=True).start()

    def depcam_color_callback(self, msg):
        self.depcam_color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        self.depcam_color_hsv = cv2.cvtColor(self.depcam_color_image, cv2.COLOR_BGR2HSV)

    def depcam_depth_callback(self, msg):
        self.depcam_depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough').astype(np.float32) / 1000.0
        if self.pc_need > 0:
            self.pc = self.depth_camera.depth2points(self.depcam_depth_image)

    def fuction_check(self, msg):
        if msg.data == 'spearhead':
            self.spearhead_need.set()
            self.get_logger().info('Spearhead check requested.')
            self.pc_need += 1
        elif msg.data == 'spearhead_stop':
            self.spearhead_need.clear()
            self.get_logger().info('Spearhead check stopped.')
            self.pc_need = max(0, self.pc_need - 1)
        elif msg.data == 'YOLO':
            self.YOLO_need.set()
            self.get_logger().info('YOLO detection requested.')
        elif msg.data == 'YOLO_stop':
            self.YOLO_need.clear()
            self.get_logger().info('YOLO detection stopped.')
        elif msg.data == 'rb_check':
            self.rb_check_need.set()
            self.get_logger().info('RB check requested.')
            # self.rb_camera_init()
        elif msg.data == 'rb_check_stop':
            self.rb_check_need.clear()
            self.get_logger().info('RB check stopped.')
        # else:
        #     self.get_logger().warn(f'Unknown function request: {msg.data}')

    def spearhead_check_thread(self):
        while True:
            self.spearhead_need.wait()
            pc = self.pc
            T_box_base_map = np.array([0, 0.33, 1.0]) # 填充 目标为右，下，前
            R_box_base_map = rot_z(00) @ rot_y(00) @ rot_x(-0) # 填充 摇摆角@俯角@右转角，即为欧拉角
            source = "map"
            target = "camera_color_optical_frame"
            try:
                tf_base_map = self.tf_buffer.lookup_transform(
                    target, source, rclpy.time.Time()
                )
                R_base_map, t_base_map = tfmsg_to_Rt(tf_base_map)

                T_box_cam = T_box_base_map @ R_base_map.T + t_base_map      # box 在 cam 下的平移
                R_box_cam = R_box_base_map @ R_base_map.T 
            except TransformException as ex:
                self.get_logger().warn(f"TF not ready: {ex}")

            box_check_t = check_spearhead(pc, self.depth_camera, T_box_cam, R_box_cam)
            self.get_logger().info(f"Spearhead check result: {box_check_t}")
            result_msg = String()
            result_dic = {"topic_name": "spearhead_check", "data": box_check_t}
            result_msg.data = json.dumps(result_dic)
            self.data_publisher.publish(result_msg)

    def YOLO_detection_thread(self):
        while True:
            self.YOLO_need.wait()
            color_img = self.depcam_color_image
            depression_angle = self.depth_camera.depression_angle
            target_loc = (0, 0, 1.0)  # 填充 目标位置为右，下，前
            target_direct = 0  # 填充 目标朝向为正前方
            target_size = (512, 512)  # 填充 目标尺寸为宽，高

            roi_img, roi_2d = img_preprocess(color_img, depression_angle, target_loc=target_loc, target_direct=target_direct, target_size=target_size)
            result = get_yolo_result(self.yolo_model, roi_img)
            self.get_logger().info(f"YOLO detection completed with {len(result)} results.")
            result_msg = String()
            result_dic = {"topic_name": "YOLO_detection", "data": []}
            res = result[0]
                
            frame_boxes = []
            xyxy = res.boxes.xyxy.tolist()
            conf = res.boxes.conf.tolist()
            cls  = res.boxes.cls.tolist()

            for b, c, k in zip(xyxy, conf, cls):
                k_int = int(k)
                frame_boxes.append({
                    "bbox_xyxy": b,                 # [x1,y1,x2,y2]
                    "conf": float(c),               # 置信度
                    "class_id": k_int,              # 类别id
                    "class_name": self.yolo_names[k_int]      # 类别名
                })

            result_dic["data"] = frame_boxes
            result_msg.data = json.dumps(result_dic)
            self.data_publisher.publish(result_msg)
    # 数据示例{'topic_name': 'YOLO_detection', 'data': [{'bbox_xyxy': [284.434814453125, 95.02224731445312, 594.1102294921875, 418.7121276855469], 'conf': 0.7210436463356018, 'class_id': 16, 'class_name': 'T_17'}]}

    def rb_check_thread(self):
        while True:
            self.rb_check_need.wait()
            # ret,fram = self.vc.read()
            fram = self.depcam_color_hsv
            # if not ret:
            #     self.get_logger().warn('RB camera read failed.')
            #     continue
            if fram is None:
                self.get_logger().warn('RB camera read failed.')
                time.sleep(0.1)
                continue
            else:
                # 用世界坐标算角点（缺内参）

                # 红区
                x1 = (12100-100-440-1620)/1000
                x2 = (12100-100-440)/1000
                y1 = (6000-125)/1000
                z1 = (400-75)/1000
                z2 = (2020-75)/1000

                H_zone3 = (400+350)/1000

                corners = np.array([
                    [x1, y1, z1],
                    [x2, y1, z1],
                    [x2, y1, z2],
                    [x1, y1, z2],
                ], dtype=np.float64)

                point_source = "map"
                point_target = "base_link"

                try:
                    tf_base_map = self.tf_buffer.lookup_transform(
                        point_target, point_source, rclpy.time.Time()
                    )
                    R_base_map, t_base_map = tfmsg_to_Rt(tf_base_map)
                    T_base_map = tfmsg_to_T(tf_base_map)
                    T_base_map[2, 3] = -H_zone3
                    corners_base = transform_points(T_base_map, corners)
                except TransformException as ex:
                    self.get_logger().warn(f"TF not ready: {ex}")
                    time.sleep(0.1)
                    continue
                    # 计算外参
                # def euler_to_R(roll, pitch, yaw):
                #     return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)

                # # 相机相对坐标
                # t_cam_in_vehicle = np.array([0.18, 0, 0.5])   # 米
                # roll, pitch, yaw = 0.1, 0.2, 0.05             # 弧度

                # rvec_USB = euler_to_R(roll, pitch, yaw)
                # tvec_USB = - rvec_USB @ t_cam_in_vehicle

                # rvec, _ = cv2.Rodrigues(rvec_USB) # 旋转矩阵转换为旋转向量
                # tvec = tvec_USB

                T_cam_base = [[ 0.05309445, -0.99672328, -0.06102193, -0.06469527],
                [ 0.41525076,  0.0776115,  -0.90639024, -0.41730752],
                [ 0.90815626,  0.02278489,  0.41801084, -0.11940187],
                [ 0.,          0.,          0.,          1.        ]]

                rvec, tvec = T_to_rvec_tvec(T_cam_base)

                corners_base = corners_base[::-1] # 逆序，变成左下角开始，顺时针
                corners_2d = trans3DToPlane(corners_base, camera_matrix=K , rvec=rvec, tvec=tvec)

                restored_img = ROIRestore(fram, corners_2d, image_shape=[300,300])
                cv2.imshow("restored", restored_img)
                cv2.waitKey(1)

                # 图像预处理
                restored_img = preproc_hsv(restored_img)
                restored_img = clahe_on_v(restored_img, clahe=self.clahe_v)

                rb_result = check_red_blue(restored_img)

                # 图像解角点，稳定性差
                # proc_res = process_image_and_detect_3x3(fram)
                # if proc_res is None:
                #     self.get_logger().warn('RB check failed to detect 3x3 grid.')
                #     continue
                # warped = proc_res['warped']
                # rb_result = check_red_blue(warped)

                result_msg = String()
                result_dic = {"topic_name": "rb_check", "data": rb_result}
                result_msg.data = json.dumps(result_dic)
                self.data_publisher.publish(result_msg)
                print(f"RB check result: {rb_result}")

    def rb_camera_init(self):
        file_name = "USB_capture.yaml"
        file_path = "cv_lib/" + file_name

        with open(file_path, "r") as f:
            cfg = yaml.safe_load(f)

        cam = cfg["camera"]
        ctrls = cam["controls"]

        dev = cam["device"]
        
        self.vc = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        self.vc.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3) 

        def set_ctrl(name, value):
            try:
                subprocess.run(
                    ["v4l2-ctl", "-d", dev, "-c", f"{name}={value}"],
                    check=True
                )
            except subprocess.CalledProcessError as e:
                print(f"Failed to set control {name} to {value}: {e}")
                raise

        # 下发所有控制参数
        for k, v in ctrls.items():
            set_ctrl(k, v)

def main():
    rclpy.init()
    node = ImageNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    executor.spin()

    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()
    cv2.waitKey(1)  # Ensure all OpenCV windows are closed

if __name__ == '__main__':
    main()