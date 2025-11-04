import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import MarkerArray, Marker
from cv_bridge import CvBridge
import os
import subprocess
import time

class ArucoDetectorNode(Node):
    def __init__(self):
        super().__init__('aruco_detector_node')
        
        # --------------------------
        # 核心参数配置（7x7字典+15cm码，适配屏幕识别）
        # --------------------------
        self.declare_parameter("aruco_dict_type", "DICT_7X7_1000")
        self.declare_parameter("marker_length", 0.15)  # 15cm码
        self.continuous_detect = {} 
        self.min_continuous_frames = 2
        self.declare_parameter("image_topic", "/usb_cam/image_raw")
        self.declare_parameter("camera_info_topic", "/usb_cam/camera_info")
        
        # 获取参数
        self.aruco_dict_type = self.get_parameter("aruco_dict_type").value
        self.marker_length = self.get_parameter("marker_length").value
        self.image_topic = self.get_parameter("image_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value
        
        try:
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(
                getattr(cv2.aruco, self.aruco_dict_type)
            )
            # 【关键调整】参考旧代码的宽松检测参数（适配屏幕码）
            self.aruco_params = cv2.aruco.DetectorParameters()
            self.aruco_params.adaptiveThreshConstant = 7
            self.aruco_params.minMarkerPerimeterRate = 0.01
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        except AttributeError:
            self.get_logger().error(f"❌ 不支持的字典：{self.aruco_dict_type}，请使用DICT_7X7_1000")
            rclpy.shutdown()
            return
        
        # --------------------------
        # 初始化变量
        # --------------------------
        self.bridge = CvBridge()
        self.camera_matrix = None
        self.dist_coeffs = None
        self.marker_3d = self._get_marker_3d()
        self.border_color = (0, 255, 0)
        
        # 日志节流
        self.last_log_time = self.get_clock().now()
        self.log_interval = 1.0
        
        # --------------------------
        # 摄像头重连配置
        # --------------------------
        self.camera_device = "/dev/video10"
        self.usb_hub_pci = "0000:00:14.0"
        self.sudo_password = "qing"
        self.reconnect_count = 0
        self.max_reconnect = 3
        self.script_dir = os.path.join(os.path.dirname(__file__), "../cv_lib/")
        self.unbind_script = os.path.join(self.script_dir, "usb_unbind.sh")
        self.bind_script = os.path.join(self.script_dir, "usb_bind.sh")
        
        # --------------------------
        # 话题订阅与发布
        # --------------------------
        self.camera_info_sub = self.create_subscription(
            CameraInfo, self.camera_info_topic, self.camera_info_cb, 10
        )
        self.image_sub = self.create_subscription(
            Image, self.image_topic, self.image_cb, 10
        )
        self.aruco_pub = self.create_publisher(MarkerArray, "/aruco_markers", 10)
        self.img_pub = self.create_publisher(Image, "/aruco/detected_img", 10)
        
        self.get_logger().info(f"✅ Aruco识别节点启动（7x7字典，码尺寸15cm，支持屏幕识别）")
        self.get_logger().info(f"📷 摄像头设备：{self.camera_device}，支持自动重连")

    # --------------------------
    # 摄像头重连核心函数
    # --------------------------
    def is_camera_online(self):
        return os.path.exists(self.camera_device)

    def reset_usb_hub(self):
        self.get_logger().warn(f"⚠️ 开始重置USB Hub（PCI地址：{self.usb_hub_pci}）")
        if not os.path.exists(self.unbind_script) or not os.path.exists(self.bind_script):
            self.get_logger().error(f"❌ 重连脚本不存在！请确认路径：{self.script_dir}")
            return False
        try:
            cmd_unbind = f"echo '{self.sudo_password}' | sudo -S sh {self.unbind_script} {self.usb_hub_pci}"
            subprocess.run(cmd_unbind, shell=True, check=True, capture_output=True, text=True)
            self.get_logger().info("✅ USB Hub卸载成功")
            time.sleep(2)
            cmd_bind = f"echo '{self.sudo_password}' | sudo -S sh {self.bind_script} {self.usb_hub_pci}"
            subprocess.run(cmd_bind, shell=True, check=True, capture_output=True, text=True)
            self.get_logger().info("✅ USB Hub重新绑定成功")
            time.sleep(3)
            return True
        except subprocess.CalledProcessError as e:
            self.get_logger().error(f"❌ 重置USB失败：{e.stderr}")
            return False
        except Exception as e:
            self.get_logger().error(f"❌ 重置USB异常：{str(e)}")
            return False

    def reconnect_camera(self):
        self.reconnect_count += 1
        if self.reconnect_count > self.max_reconnect:
            self.get_logger().error(f"❌ 重连失败（已尝试{self.max_reconnect}次），请检查：")
            self.get_logger().error("  1. USB线是否插紧  2. 摄像头是否损坏  3. 换一个USB端口")
            return False
        self.get_logger().warn(f"⚠️ 第{self.reconnect_count}次尝试重连摄像头...")
        if self.is_camera_online():
            self.get_logger().info("📌 摄像头设备在线，尝试重启usb_cam节点")
            try:
                subprocess.run(
                    f"echo '{self.sudo_password}' | sudo -S ros2 service call /usb_cam_node/reset std_srvs/srv/Empty",
                    shell=True, check=True, capture_output=True, text=True
                )
                time.sleep(2)
                self.get_logger().info("✅ usb_cam节点重启成功")
                return True
            except Exception as e:
                self.get_logger().error(f"❌ 重启节点失败：{str(e)}")
                return False
        else:
            self.get_logger().info("📌 摄像头设备离线，尝试重置USB Hub")
            if self.reset_usb_hub():
                if self.is_camera_online():
                    self.get_logger().info(f"✅ 摄像头已恢复（{self.camera_device}重新出现）")
                    self.reconnect_count = 0
                    return True
                else:
                    self.get_logger().error(f"❌ USB重置后仍未找到{self.camera_device}")
                    return False

    def _get_marker_3d(self):
        """生成15cm码的3D坐标"""
        half = self.marker_length / 2.0
        return np.array([
            [-half, -half, 0.0],
            [half, -half, 0.0],
            [half, half, 0.0],
            [-half, half, 0.0]
        ], dtype=np.float32)

    def camera_info_cb(self, msg):
        """仅加载一次相机内参"""
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape(3, 3)
            self.dist_coeffs = np.array(msg.d) if msg.d else np.zeros(5, dtype=np.float32)
            self.get_logger().info(f"📊 相机内参加载完成（fx={self.camera_matrix[0,0]:.1f}）")
            self.destroy_subscription(self.camera_info_sub)

    def rotvec_to_quat(self, rvec):
        """旋转向量转四元数"""
        mat, _ = cv2.Rodrigues(rvec)
        tr = mat[0,0] + mat[1,1] + mat[2,2]
        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2
            return [(mat[2,1]-mat[1,2])/S, (mat[0,2]-mat[2,0])/S, (mat[1,0]-mat[0,1])/S, 0.25*S]
        elif mat[0,0] > mat[1,1] and mat[0,0] > mat[2,2]:
            S = np.sqrt(1.0 + mat[0,0] - mat[1,1] - mat[2,2]) * 2
            return [0.25*S, (mat[0,1]+mat[1,0])/S, (mat[0,2]+mat[2,0])/S, (mat[2,1]-mat[1,2])/S]
        elif mat[1,1] > mat[2,2]:
            S = np.sqrt(1.0 + mat[1,1] - mat[0,0] - mat[2,2]) * 2
            return [(mat[0,1]+mat[1,0])/S, 0.25*S, (mat[1,2]+mat[2,1])/S, (mat[0,2]-mat[2,0])/S]
        else:
            S = np.sqrt(1.0 + mat[2,2] - mat[0,0] - mat[1,1]) * 2
            return [(mat[0,2]+mat[2,0])/S, (mat[1,2]+mat[2,1])/S, 0.25*S, (mat[1,0]-mat[0,1])/S]

    def image_cb(self, msg):
        """核心回调：适配屏幕识别+优化位姿解算"""
        if self.camera_matrix is None:
            return
        
        # 1. ROS图像转OpenCV（保留重连触发）
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            if self.reconnect_count > 0:
                self.reconnect_count = 0
                self.get_logger().info("✅ 摄像头正常工作，重连计数器重置")
        except Exception as e:
            self.get_logger().error(f"❌ 图像转换失败（可能掉线）: {str(e)}")
            if not self.reconnect_camera():
                return
        
        # 2. 检测Aruco码
        corners, ids, rejected = self.detector.detectMarkers(cv_img)
        marker_array = MarkerArray()
        detected_ids = []
        detected_bin_info = []
        detected_pose_info = []
        
        # 3. 处理识别结果
        if ids is not None and len(ids) > 0:
            ids = np.array(ids, dtype=np.int32)
            
            for i in range(len(ids)):
                marker_id = int(ids[i][0])
                curr_corners = corners[i]
                curr_corners_2d = curr_corners.squeeze()
                
                # 连续识别过滤
                if marker_id not in self.continuous_detect:
                    self.continuous_detect[marker_id] = 1
                else:
                    self.continuous_detect[marker_id] += 1
                
                # 仅连续识别到min_continuous_frames帧才处理
                if self.continuous_detect[marker_id] < self.min_continuous_frames:
                    continue
                
                detected_ids.append(marker_id)
                # 解析10位ID
                binary_10bit = bin(marker_id)[2:].zfill(10)
                first_8bit = binary_10bit[:8]
                last_2bit = binary_10bit[8:]
                detected_bin_info.append(
                    f"ID={marker_id}（10位二进制：{binary_10bit}，前8位：{first_8bit}，后2位：{last_2bit}）"
                )
                
                # 4. 位姿解算优化
                try:
                    _, rvec, tvec = cv2.solvePnP(
                        objectPoints=self.marker_3d,
                        imagePoints=curr_corners_2d,
                        cameraMatrix=self.camera_matrix,
                        distCoeffs=self.dist_coeffs,
                        flags=cv2.SOLVEPNP_EPNP
                    )
                    detected_pose_info.append(
                        f"x={tvec[0][0]:.2f}m,y={tvec[1][0]:.2f}m,z={tvec[2][0]:.2f}m"
                    )
                except Exception as e:
                    self.get_logger().warn(f"⚠️ ID={marker_id} 位姿解算失败: {str(e)}")
                    detected_pose_info.append("位姿解算失败")
                    continue
                
                # 5. 绘制识别结果
                cv2.aruco.drawDetectedMarkers(cv_img, [curr_corners], ids[i:i+1], self.border_color)
                cv2.drawFrameAxes(
                    cv_img, self.camera_matrix, self.dist_coeffs,
                    rvec, tvec, self.marker_length / 2
                )
                
                # 6. 发布Marker消息
                marker_msg = Marker()
                marker_msg.header = msg.header
                marker_msg.header.frame_id = "camera"
                marker_msg.id = marker_id
                marker_msg.type = Marker.CUBE
                marker_msg.action = Marker.ADD
                marker_msg.scale.x = self.marker_length
                marker_msg.scale.y = self.marker_length
                marker_msg.scale.z = 0.01
                marker_msg.color.r = 0.0
                marker_msg.color.g = 1.0
                marker_msg.color.b = 0.0
                marker_msg.color.a = 0.5
                marker_msg.pose.position.x = float(tvec[0][0])
                marker_msg.pose.position.y = float(tvec[1][0])
                marker_msg.pose.position.z = float(tvec[2][0])
                qx, qy, qz, qw = self.rotvec_to_quat(rvec)
                marker_msg.pose.orientation.x = qx
                marker_msg.pose.orientation.y = qy
                marker_msg.pose.orientation.z = qz
                marker_msg.pose.orientation.w = qw
                marker_array.markers.append(marker_msg)
            
            # 7. 发布识别结果
            self.aruco_pub.publish(marker_array)
        
        # 8. 发布带标记的图像
        try:
            img_msg = self.bridge.cv2_to_imgmsg(cv_img, "bgr8")
            img_msg.header = msg.header
            self.img_pub.publish(img_msg)
        except Exception as e:
            self.get_logger().error(f"❌ 图像发布失败: {str(e)}")
        
        # 9. 节流打印日志
        current_time = self.get_clock().now()
        if len(detected_ids) > 0 and (current_time - self.last_log_time).nanoseconds / 1e9 > self.log_interval:
            self.get_logger().info(
                f"📤 识别到{len(detected_ids)}个码：{', '.join(detected_bin_info)} | 位姿：{', '.join(detected_pose_info)}"
            )
            self.last_log_time = current_time

def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("🛑 节点已停止")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
