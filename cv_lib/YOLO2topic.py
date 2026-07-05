import json
import yaml
import cv2
import subprocess
from cv_bridge import CvBridge
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from ultralytics import YOLO
import time
import threading
import rclpy

rate = 0.1
model = '1.20.pt'
yaml_file = 'USB_capture.yaml'

def get_yolo_result(model, img):
    result = model.predict(source=img, save=False, save_txt=False, conf=0.5, iou=0.45, verbose=False)
    return result

class YOLONode(Node):
    def __init__(self):
        super().__init__('YOLO_recog_node')

        self.vc = None
        self.yolo_model = YOLO(model)
        self.yolo_names = self.yolo_model.names
        self.bridge = CvBridge()

        # 仅发送 class_name 的话题（原 YOLO_detection 改为只发类别名）
        self.data_publisher = self.create_publisher(String, 'YOLO_detection', 10)
        # 带检测框标注的视频流话题
        self.image_publisher = self.create_publisher(Image, 'YOLO_detection_image', 10)

        self.rb_camera_init()
        threading.Thread(target=self.YOLO_detection_thread, daemon=True).start()

    def rb_camera_init(self):
        file_name = yaml_file
        file_path = "cv_lib/" + file_name

        with open(file_path, "r") as f:
            cfg = yaml.safe_load(f)

        cam = cfg["camera"]
        ctrls = cam["controls"]
        dev = cam["device"]

        self.vc = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        if not self.vc.isOpened():
            self.get_logger().error(f"Failed to open camera: {dev}")
            raise RuntimeError(f"Failed to open camera: {dev}")

        # self.vc.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)

        def set_ctrl(name, value):
            try:
                subprocess.run(
                    ["v4l2-ctl", "-d", dev, "-c", f"{name}={value}"],
                    check=True
                )
            except subprocess.CalledProcessError as e:
                self.get_logger().error(f"Failed to set control {name} to {value}: {e}")
                raise

        # 下发所有控制参数
        for k, v in ctrls.items():
            set_ctrl(k, v)

    def YOLO_detection_thread(self):
        while rclpy.ok():
            ret, fram = self.vc.read()
            if not ret:
                self.get_logger().warn('wait RB camera')
                time.sleep(0.1)
                continue

            result = get_yolo_result(self.yolo_model, fram)
            res = result[0]

            xyxy = res.boxes.xyxy.tolist()
            conf = res.boxes.conf.tolist()
            cls  = res.boxes.cls.tolist()

            # ===== 1. 只在话题里发送 class_name =====
            for k in cls:
                k_int = int(k)
                class_name = self.yolo_names[k_int]
                msg = String()
                msg.data = class_name
                self.data_publisher.publish(msg)

            # ===== 2. 框出检测结果并发送视频流话题 =====
            annotated_frame = fram.copy()
            for b, c, k in zip(xyxy, conf, cls):
                x1, y1, x2, y2 = [int(v) for v in b]
                k_int = int(k)
                label = f"{self.yolo_names[k_int]} {c:.2f}"
                # 画框
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # 画标签
                cv2.putText(annotated_frame, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            image_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding='bgr8')
            self.image_publisher.publish(image_msg)

            time.sleep(rate)

def main(args=None):
    rclpy.init(args=args)
    node = YOLONode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        node.vc.release()

if __name__ == '__main__':
    main()
