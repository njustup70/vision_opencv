from __future__ import annotations
import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')

        self.declare_parameter('brightness', 10.0)
        self.declare_parameter('contrast', 8.0)
        self.declare_parameter('exposure', 100.0)

        self.declare_parameter('camera_index', 0)
        self.camera_index = self.get_parameter('camera_index').value

        self.declare_parameter('fps', 60)
        self.fps = self.get_parameter('fps').value

        self.publisher_ = self.create_publisher(Image, 'camera/image_raw', 100)
        self.timer = self.create_timer(1.0 / self.fps, self.timer_callback)

        self.bridge = CvBridge()

        # 打开摄像头
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise RuntimeError('无法打开任何摄像头')

        self.apply_camera_parameters()

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        self.get_logger().info('摄像头节点已启动')

    def _set_camera_prop(self, name: str, prop_id: int, value):
        if value is None:
            return
        success = self.cap.set(prop_id, value)
        self.get_logger().info(f'设置{name}: {value}, 成功: {success}')

    def apply_camera_parameters(self):
        """应用摄像头参数设置"""
        try:
            brightness = self.get_parameter('brightness').value
            contrast = self.get_parameter('contrast').value
            exposure = self.get_parameter('exposure').value

            self._set_camera_prop('亮度', cv2.CAP_PROP_BRIGHTNESS, brightness)
            self._set_camera_prop('对比度', cv2.CAP_PROP_CONTRAST, contrast)

            if exposure is not None:
                try:
                    self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
                except Exception:
                    pass

                self._set_camera_prop('曝光', cv2.CAP_PROP_EXPOSURE, exposure)

        except Exception as e:
            self.get_logger().warning(f'设置摄像头参数时出错: {e}')

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            msg = self.bridge.cv2_to_imgmsg(rgb_frame, encoding='rgb8')
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'camera_frame'
            self.publisher_.publish(msg)

            cv2.imshow('Camera View', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.get_logger().info('用户请求退出')
                self.destroy_node()

        except Exception as e:
            self.get_logger().error(f'发布失败: {e}')

    def destroy_node(self):
        """清理资源"""
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = CameraNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
