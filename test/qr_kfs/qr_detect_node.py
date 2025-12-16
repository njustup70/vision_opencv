#!/usr/bin/env python3
import os
import time
import threading
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from pyzbar.pyzbar import decode

os.environ['QT_QPA_PLATFORM'] = 'xcb'
from .qr_core import QRCoder

class QRDetectNode(Node):
    def __init__(self):
        super().__init__('qr_detect_node')
        self.bridge = CvBridge()
        self.running = True
        
        self.create_subscription(Image, 'camera/image_raw', self.callback, 100)
        print("QR检测节点启动")
        print("输入12个状态(空 R1 R2 假），以空格分隔")
        print("输入'q'退出")
    
    def callback(self, msg):
        """检测二维码"""
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            qr_codes = decode(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            
            if qr_codes:
                data = qr_codes[0].data.decode().strip()
                states = QRCoder.decode(data)
                
                if states:
                    print(f"\n检测到: {data}")
                    for i, s in enumerate(states, 1):
                        print(f"  位置{i}: {s}")
        except:
            pass
    
    def process_input(self, states):
        """处理输入的状态"""
        try:
            path, hex_data = QRCoder.encode(states)
            print(f"生成: {hex_data}")
            
            # 播放100ms
            img = cv2.imread(path)
            if img is not None:
                cv2.namedWindow("QR Player", cv2.WINDOW_NORMAL)
                cv2.setWindowProperty("QR Player", cv2.WND_PROP_FULLSCREEN, 1)
                cv2.imshow("QR Player", img)
                
                start = time.time()
                while (time.time() - start) * 1000 < 100:
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                cv2.destroyWindow("QR Player")
                print("播放完成")
                
        except Exception as e:
            print(f"错误: {e}")

def main():
    rclpy.init()
    node = QRDetectNode()
    
    spin_thread = threading.Thread(
        target=lambda: rclpy.spin(node) if node.running else None,
        daemon=True
    )
    spin_thread.start()
    
    try:
        while node.running:
            try:
                cmd = input("> ").strip()
                
                if cmd == 'q':
                    break
                
                states = cmd.split()
                if len(states) == 12:
                    node.process_input(states)
                else:
                    print(f"需要12个状态，当前{len(states)}个")
                    
            except KeyboardInterrupt:
                break
    finally:
        node.running = False
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()
        print("结束")

if __name__ == '__main__':
    main()