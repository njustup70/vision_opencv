#test code (abandoned)

import sys
import os
import time
import cv2
import numpy as np
import rclpy
import json
from rclpy.node import Node
from cv_lib.ros.basket_ros import ImagePublish_t
from PoseSolver.PoseSolver import PoseSolver
from YOLOv11.yolo_lib import MyYOLO
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge
from std_msgs.msg import Float32, String

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

        # 初始化各组件
        self.yolo_detector = MyYOLO("/home/Elaina/yolo/weights/nanjing.pt", show=True)
        self.image_publisher = ImagePublish_t("yolo/image")
        self.pose_solver = PoseSolver(
            self.camera_matrix,
            self.dist_coeffs,
            marker_length=0.1,
            print_result=True
        )

        # 创建发布者
        self.yaw_publisher = self.create_publisher(Float32, 'basket_yaw', 10)
        self.json_publisher = self.create_publisher(String, 'yaw_json', 10)  # 新增JSON发布者

        # 图像缓冲区
        self.image = np.zeros((480, 640, 3), dtype=np.uint8)
        self.bridge = CvBridge()

        self.sub_com = self.create_subscription(CompressedImage, "/camera/color/image_raw/compressed", self.compressed_image_callback, 1)
        self.sub_raw = self.create_subscription(Image, "/camera/color/image_raw", self.regular_image_callback, 1)

        self.get_logger().info("ImageProcessingNode 初始化完成")

    def preprocess_backboard(self, img):
        """透明篮板专用预处理函数"""
        # 1. 动态范围压缩 (处理强光过曝)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)

        # 自适应对比度增强
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l_channel)

        # 2. 眩光区域修复
        _, glare_mask = cv2.threshold(l_enhanced, 220, 255, cv2.THRESH_BINARY)
        glare_mask = cv2.morphologyEx(glare_mask, cv2.MORPH_CLOSE, np.ones((15, 15)))
        repaired = cv2.inpaint(cv2.merge([l_enhanced, a, b]), glare_mask, 3, cv2.INPAINT_TELEA)
        return cv2.cvtColor(repaired, cv2.COLOR_LAB2BGR)

    def compressed_image_callback(self, msg: CompressedImage):
        """处理压缩图像消息的回调函数"""
        self.get_logger().info("接收到压缩图像消息")
        try:
            self.image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
            # 对图像进行预处理
            self.image = self.preprocess_backboard(self.image)
            self.process_image()
        except Exception as e:
            self.get_logger().error(f"处理压缩图像时出错: {str(e)}")
            import traceback
            self.get_logger().error(f"堆栈跟踪: {traceback.format_exc()}")

    def regular_image_callback(self, msg: Image):
        """处理非压缩图像消息的回调函数"""
        self.get_logger().info("接收到非压缩图像消息")
        try:
            self.image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            # 对图像进行预处理
            self.image = self.preprocess_backboard(self.image)
            self.process_image()
        except Exception as e:
            self.get_logger().error(f"处理非压缩图像时出错: {str(e)}")
            import traceback
            self.get_logger().error(f"堆栈跟踪: {traceback.format_exc()}")

    def is_detect_basket(self):
        """返回yaw值并发布JSON格式"""
        yaw_msg = Float32()
        json_msg = String()

        if not hasattr(self.pose_solver, 'pnp_result'):
            yaw_msg.data = 0.0
            # 构造JSON: {"has_yaw": false}
            json_data = {"has_yaw": False}
            json_msg.data = json.dumps(json_data)
            self.yaw_publisher.publish(yaw_msg)
            self.json_publisher.publish(json_msg)
            return 0

        pose_info = self.pose_solver.pnp_result

        if 'yaw' not in pose_info or pose_info['yaw'] is None:
            yaw_msg.data = 0.0
            # 构造JSON: {"has_yaw": false}
            json_data = {"has_yaw": False}
            json_msg.data = json.dumps(json_data)
            self.yaw_publisher.publish(yaw_msg)
            self.json_publisher.publish(json_msg)
            return 0

        yaw_value = float(pose_info['yaw'])
        yaw_msg.data = yaw_value
        # 构造JSON: {"yaw": 角度值, "has_yaw": true}
        json_data = {"yaw": yaw_value, "has_yaw": True}
        json_msg.data = json.dumps(json_data)
        self.yaw_publisher.publish(yaw_msg)
        self.json_publisher.publish(json_msg)
        return yaw_value


    def process_image(self):
        # YOLO检测
        corners = self.yolo_detector.update(self.image)

        # 发布处理后的图像
        self.image_publisher.update(self.image)

        # 位姿解算
        if corners is not None and len(corners) > 0:
            corners_list = [np.array(c, dtype=np.float32) for c in corners] if isinstance(corners, list) else [np.array(corners, dtype=np.float32)]
            self.pose_solver.update(self.image, corners_list)
        else:
            if hasattr(self.pose_solver, 'pnp_result'):
                delattr(self.pose_solver, 'pnp_result')

        # 发布yaw值和JSON
        self.is_detect_basket()

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
