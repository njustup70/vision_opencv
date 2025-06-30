import sys
import os
import time
import cv2
import numpy as np
import json
import rclpy
from rclpy.node import Node
from cv_lib.ros.basket_ros import ImagePublish_t
from PoseSolver.PoseSolver import PoseSolver
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge
from std_msgs.msg import Float32, String
from PoseSolver.Aruco import Aruco

class ImageProcessingNode(Node):
    def __init__(self):
        super().__init__('image_processing_node')

        # 相机内参矩阵
        self.camera_matrix = np.array([
            [606.634521484375, 0, 433.2264404296875],
            [0, 606.5910034179688, 247.10369873046875],
            [0.000000, 0.000000, 1.000000]
        ], dtype=np.float32)

        # 畸变系数
        self.dist_coeffs = np.array([[0, 0, 0, 0, 0]], dtype=np.float32)
        
        # Y轴偏移量校正值（单位：米）
        self.y_offset_correction = 0.16

        # 初始化各组件
        self.aruco_detector = Aruco("DICT_5X5_1000", if_draw=True)
        self.image_publisher = ImagePublish_t("aruco")
        self.pose_solver = PoseSolver(
            self.camera_matrix,
            self.dist_coeffs,
            marker_length=0.0885,
            marker_width=None,
            print_result=True
        )

        # 创建发布者
        self.yaw_publisher = self.create_publisher(Float32, 'aruco_yaw', 10)
        self.json_publisher = self.create_publisher(String, 'aruco_yaw_json', 10)

        # 图像缓冲区
        self.image = np.zeros((480, 640, 3), dtype=np.uint8)
        self.bridge = CvBridge()

        self.sub_com = self.create_subscription(CompressedImage, "/camera/color/image_raw/compressed", self.compressed_image_callback, 1)
        self.sub_raw = self.create_subscription(Image, "/camera/color/image_raw", self.regular_image_callback, 1)

    def compressed_image_callback(self, msg: CompressedImage):
        """处理压缩图像消息的回调函数"""
        try:
            self.image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
            self.process_image()
        except Exception as e:
            import traceback
            self.get_logger().error(f"堆栈跟踪: {traceback.format_exc()}")

    def regular_image_callback(self, msg: Image):
        """处理非压缩图像消息的回调函数"""
        try:
            self.image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.process_image()
        except Exception as e:
            import traceback
            self.get_logger().error(f"堆栈跟踪: {traceback.format_exc()}")

    def is_detect_aruco(self):
        """返回校正后的yaw值和y_offset并发布JSON格式"""
        yaw_msg = Float32()
        json_msg = String()

        if not hasattr(self.pose_solver, 'pnp_result'):
            yaw_msg.data = 0.0
            # 构造JSON: {"has_yaw": false, "has_offset": false}
            json_data = {"has_yaw": False, "has_offset": False}
            json_msg.data = json.dumps(json_data)
            self.yaw_publisher.publish(yaw_msg)
            self.json_publisher.publish(json_msg)
            return 0

        pose_info = self.pose_solver.pnp_result

        # 检查yaw和y_offset是否存在
        has_yaw = 'yaw' in pose_info and pose_info['yaw'] is not None
        has_offset = 'y_offset' in pose_info and pose_info['y_offset'] is not None

        if not has_yaw or not has_offset:
            yaw_msg.data = 0.0
            # 构造JSON: {"has_yaw": false, "has_offset": false}
            json_data = {"has_yaw": False, "has_offset": False}
            json_msg.data = json.dumps(json_data)
            self.yaw_publisher.publish(yaw_msg)
            self.json_publisher.publish(json_msg)
            return 0

        yaw_value = float(pose_info['yaw'])
        raw_y_offset = float(pose_info['y_offset'])
        
        # 校正Y轴偏移量（减去0.16米）
        corrected_y_offset = raw_y_offset - self.y_offset_correction
        
        yaw_msg.data = yaw_value
        
        # 构造JSON: 只包含校正后的偏移量
        json_data = {
            "yaw": yaw_value,
            "corrected_y_offset": corrected_y_offset,
            "has_yaw": True,
            "has_offset": True
        }
        json_msg.data = json.dumps(json_data)
        self.yaw_publisher.publish(yaw_msg)
        self.json_publisher.publish(json_msg)
        return yaw_value

    def process_image(self):
        # ArUco码检测
        aruco_results = self.aruco_detector.detect_image(self.image, if_draw=True)
        if isinstance(aruco_results, np.ndarray):
            aruco_corners = [result["corners"] for result in aruco_results]
        else:
            aruco_corners = aruco_results

        all_corners = []
        if aruco_corners is not None:
            all_corners.extend(aruco_corners)

        # 发布处理后的图像
        self.image_publisher.update(self.image)

        # 位姿解算
        if all_corners is not None and len(all_corners) > 0:
            corners_list = [np.array(c, dtype=np.float32) for c in all_corners] if isinstance(all_corners, list) else [np.array(all_corners, dtype=np.float32)]
            self.pose_solver.update(self.image, corners_list)
        else:
            if hasattr(self.pose_solver, 'pnp_result'):
                delattr(self.pose_solver, 'pnp_result')

        # 发布yaw值、校正后的y_offset和JSON
        yaw_value = self.is_detect_aruco()
        
        if yaw_value != 0:
            # 从pose_solver中获取校正后的y_offset（需要确保pose_solver已更新）
            if hasattr(self.pose_solver, 'pnp_result'):
                corrected_y_offset = self.pose_solver.pnp_result.get('corrected_y_offset', 0)
                print(f"  Yaw角度: {yaw_value:.1f}°, 校正后Y轴偏移: {corrected_y_offset:.3f}m")

        # 显示结果
        cv2.imshow("Detection Result", self.image)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    try:
        node = ImageProcessingNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
