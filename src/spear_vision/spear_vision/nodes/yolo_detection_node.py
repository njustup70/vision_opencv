#!/usr/bin/env python3
"""
YOLO 目标检测 ROS 2 节点
从 cv_lib/YOLO2topic.py 迁入 spear_vision 包

发布话题:
  - YOLO_detection       (std_msgs/String)   检测到的类别名

调试显示:
  - 本地 OpenCV 弹窗 (不占用通信带宽)

参数:
  - model_path   : YOLO 模型文件路径 (默认: workspace 根目录下的 1.20.pt)
  - config_file  : USB 相机配置文件 (默认: 包内 config/usb_camera.yaml)
  - rate         : 检测循环间隔秒数 (默认: 0.1)
"""

import os
import subprocess
import threading
import time

import cv2
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from std_msgs.msg import String
from ultralytics import YOLO


def get_yolo_result(model, img):
    result = model.predict(source=img, save=False, save_txt=False,
                           conf=0.5, iou=0.45, verbose=False)
    return result


class YOLODetectionNode(Node):
    def __init__(self):
        super().__init__('YOLO_recog_node')

        # ---- 参数 ----
        pkg_share = get_package_share_directory('spear_vision')
        self.declare_parameter('model_path', '1.20.pt')
        self.declare_parameter('config_file',
                               os.path.join(pkg_share, 'config', 'usb_camera.yaml'))
        self.declare_parameter('rate', 0.1)

        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        config_file = self.get_parameter('config_file').get_parameter_value().string_value
        self.rate = self.get_parameter('rate').get_parameter_value().double_value

        self.get_logger().info(f'模型路径: {model_path}')
        self.get_logger().info(f'相机配置: {config_file}')

        # ---- 模型 & 相机 ----
        self.yolo_model = YOLO(model_path)
        self.yolo_names = self.yolo_model.names
        self.vc = None

        # ---- 发布者 ----
        self.data_publisher = self.create_publisher(String, 'YOLO_detection', 10)

        # ---- 初始化相机 & 启动检测线程 ----
        self._init_camera(config_file)
        threading.Thread(target=self._detection_loop, daemon=True).start()

    def _init_camera(self, config_file: str):
        with open(config_file, "r") as f:
            cfg = yaml.safe_load(f)

        cam = cfg["camera"]
        ctrls = cam["controls"]
        dev = cam["device"]

        self.vc = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        if not self.vc.isOpened():
            self.get_logger().error(f"Failed to open camera: {dev}")
            raise RuntimeError(f"Failed to open camera: {dev}")

        for k, v in ctrls.items():
            try:
                subprocess.run(
                    ["v4l2-ctl", "-d", dev, "-c", f"{k}={v}"],
                    check=True
                )
            except subprocess.CalledProcessError as e:
                self.get_logger().error(f"Failed to set {k}={v}: {e}")
                raise

    def _detection_loop(self):
        while rclpy.ok():
            ret, frame = self.vc.read()
            if not ret:
                self.get_logger().warn('wait RB camera')
                time.sleep(0.1)
                continue

            result = get_yolo_result(self.yolo_model, frame)
            res = result[0]

            xyxy = res.boxes.xyxy.tolist()
            conf = res.boxes.conf.tolist()
            cls = res.boxes.cls.tolist()

            # 发送类别名
            for k in cls:
                k_int = int(k)
                class_name = self.yolo_names[k_int]
                msg = String()
                msg.data = class_name
                self.data_publisher.publish(msg)

            # 画框 & 本地弹窗调试
            annotated = frame.copy()
            for b, c, k in zip(xyxy, conf, cls):
                x1, y1, x2, y2 = [int(v) for v in b]
                k_int = int(k)
                label = f"{self.yolo_names[k_int]} {c:.2f}"
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow('YOLO Detection (debug)', annotated)
            cv2.waitKey(1)

            time.sleep(self.rate)

    def destroy_node(self):
        if self.vc is not None:
            self.vc.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = YOLODetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
