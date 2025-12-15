#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')
        
        self.camera_index = 1
        self.fps = 60
        
        self.publisher_ = self.create_publisher(Image, 'camera/image_raw', 100)
        self.timer = self.create_timer(1.0/self.fps, self.timer_callback)
        
        self.bridge = CvBridge()
        
        # 打开摄像头
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise RuntimeError('无法打开任何摄像头')
        
        # 设置摄像头参数
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        self.get_logger().info('摄像头节点已启动')
    
    def timer_callback(self):
        ret, frame = self.cap.read()
        if ret:
            try:
                # 发布图像
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                msg = self.bridge.cv2_to_imgmsg(rgb_frame, encoding='rgb8')
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = 'camera_frame'
                self.publisher_.publish(msg)
                
                # 显示摄像头窗口
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
    
    try:
        node = CameraNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()