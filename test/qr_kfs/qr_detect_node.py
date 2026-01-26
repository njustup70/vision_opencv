import os
import time
import threading
import cv2
import rclpy
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from pyzbar.pyzbar import decode

os.environ['QT_QPA_PLATFORM'] = 'xcb'
from .qr_core import QRCoder

class QRDetectNode(Node):
    def __init__(self):
        super().__init__('qr_detect_node')
        
        self.declare_parameter('node_type', 'R2')  # R1: 二维码显示, R2: 摄像头+识别
        self.node_type = self.get_parameter('node_type').value
        
        self.bridge = CvBridge()
        self.running = True
        
        if self.node_type == 'R1':
            # R1机器人：二维码显示模式
            self.get_logger().info('启动R1机器人（二维码显示模式）')
            
            # 订阅code_recognize话题
            self.create_subscription(String, 'code_recognize', self.code_callback, 10)
            
            # 初始化二维码相关变量
            self.current_qr_img = None
            self.displaying = False
            
            self.display_thread = threading.Thread(target=self.display_loop, daemon=True)
            self.display_thread.start()
            
        elif self.node_type == 'R2':
            # R2机器人：摄像头+识别模式
            self.get_logger().info('启动R2机器人（摄像头+识别模式）')
            
            # 订阅摄像头图像
            self.create_subscription(Image, 'camera/image_raw', self.image_callback, 100)
            
            # 发布识别结果
            self.result_publisher = self.create_publisher(String, 'qr_detection_result', 10)
            
            self.last_detected_data = None
            
        else:
            self.get_logger().error(f'未知的节点类型: {self.node_type}')
            raise ValueError(f'节点类型必须是 R1 或 R2，当前为: {self.node_type}')
    
    def code_callback(self, msg):
        """R1: 接收KFS状态并生成显示二维码"""
        try:
            states_str = msg.data.strip()
            states_list = states_str.split()
            
            if len(states_list) != 12:
                self.get_logger().error(f'需要12个状态，收到 {len(states_list)} 个')
                return
            
            valid_states = ["空", "R1", "R2", "假"]
            for state in states_list:
                if state not in valid_states:
                    self.get_logger().error(f'无效状态: {state}')
                    return
            
            self.get_logger().info(f'生成二维码，状态: {states_list}')
            path, hex_data = QRCoder.encode(
                states_list,
                size_cm=15,
                dpi=220,
                save_dir="./qr_codes"
            )
            
            img = cv2.imread(path)
            if img is not None:
                self.current_qr_img = img
                self.displaying = True
                self.get_logger().info(f'二维码已生成: {hex_data}')
            else:
                self.get_logger().error('无法加载二维码图片')
                
        except Exception as e:
            self.get_logger().error(f'处理KFS状态时出错: {e}')
    
    def display_loop(self):
        """R1: 在便携屏上持续显示二维码"""
        screen = {'width': 2160, 'height': 1440, 'x': 2560, 'y': 0}
        window_name = "QR Display - R1"
        
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(window_name, screen['x'], screen['y'])
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        while rclpy.ok() and self.running:
            if self.current_qr_img is not None and self.displaying:
                qr_h, qr_w = self.current_qr_img.shape[:2]
                x = max(0, (screen['width'] - qr_w) // 2)
                y = max(0, (screen['height'] - qr_h) // 2)
                
                blank = np.ones((screen['height'], screen['width'], 3), dtype=np.uint8) * 255
                blank[y:y+qr_h, x:x+qr_w] = self.current_qr_img
                
                # 显示二维码
                cv2.imshow(window_name, blank)
            else:
                blank = np.ones((screen['height'], screen['width'], 3), dtype=np.uint8) * 255
                cv2.imshow(window_name, blank)
            
            if cv2.waitKey(30) & 0xFF == ord('q'):
                self.running = False
                break
        
        cv2.destroyAllWindows()
    
    def image_callback(self, msg):
        """R2: 处理摄像头图像进行二维码识别"""
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            
            # 二维码识别
            qr_codes = decode(frame)
            
            if qr_codes:
                data = qr_codes[0].data.decode().strip()
                
                # 解码状态
                states = QRCoder.decode(data)
                if states:
                    if data != self.last_detected_data:
                        self.last_detected_data = data
                        
                        # 发布识别结果
                        result_msg = String()
                        result_msg.data = data
                        self.result_publisher.publish(result_msg)
                        
                        # 打印识别结果
                        print(f"\n✅ 检测到二维码: {data}")
                        print(f"解码状态:")
                        for i in range(0, 12, 6):
                            line = "  "
                            for j in range(6):
                                pos = i + j + 1
                                if pos <= 12:
                                    line += f"{pos:2d}:{states[i+j]}  "
                            print(line)
                    
                    for qr in qr_codes:
                        pts = qr.polygon
                        if len(pts) == 4:
                            pts = [(pt.x, pt.y) for pt in pts]
                            for i in range(4):
                                cv2.line(frame, pts[i], pts[(i+1)%4], (0, 255, 0), 2)
            
            cv2.imshow('Camera View - R2', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.get_logger().info('用户请求退出')
                self.running = False
            
        except Exception as e:
            self.get_logger().warn(f'图像处理错误: {str(e)[:50]}')
    
    def destroy_node(self):
        """清理资源"""
        self.running = False
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    
    node = QRDetectNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()

if __name__ == '__main__':
    main()