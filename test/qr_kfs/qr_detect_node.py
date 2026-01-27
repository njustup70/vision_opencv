import os
import time
import threading
import cv2
import rclpy
import numpy as np
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
        self.detected_data = None
        
        self.create_subscription(Image, 'camera/image_raw', self.callback, 100)
        print("QR检测节点启动")
    
    def callback(self, msg):
        """检测二维码"""
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            qr_codes = decode(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            
            if qr_codes:
                self.detected_data = qr_codes[0].data.decode().strip()
                states = QRCoder.decode(self.detected_data)
                if states:
                    print(f"\n✅ 检测到: {self.detected_data}")
                    for i in range(0, 12, 6):
                        line = "  "
                        for j in range(6):
                            pos = i + j + 1
                            if pos <= 12:
                                line += f"{pos:2d}:{states[i+j]}  "
                        print(line)
        except:
            pass
    
    def show_qr(self, img, duration_ms):
        """在便携屏显示二维码"""
        screen = {'width': 2160, 'height': 1440, 'x': 2560, 'y': 0}
        qr_h, qr_w = img.shape[:2]
        x = max(0, (screen['width'] - qr_w) // 2)
        y = max(0, (screen['height'] - qr_h) // 2)
        
        window_name = "QR Display"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(window_name, screen['x'], screen['y'])
        cv2.resizeWindow(window_name, screen['width'], screen['height'])
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, 1)
        
        blank = np.ones((screen['height'], screen['width'], 3), dtype=np.uint8) * 255
        blank[y:y+qr_h, x:x+qr_w] = img
        
        cv2.imshow(window_name, blank)
        cv2.waitKey(10)
        
        start = time.time()
        while (time.time() - start) * 1000 < duration_ms:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.destroyWindow(window_name)

# def main():
#     rclpy.init()
#     node = QRDetectNode()
    
#     spin_thread = threading.Thread(
#         target=lambda: rclpy.spin(node) if node.running else None,
#         daemon=True
#     )
#     spin_thread.start()
    
#     try:
#         while node.running:
#             states = input("\n输入12个状态 (空格分隔) 或 'q'退出: ").strip()
#             if states == 'q':
#                 break
            
#             states_list = states.split()
#             if len(states_list) != 12:
#                 print("需要12个状态")
#                 continue
            
#             path, hex_data = QRCoder.encode(
#                 states_list,
#                 size_cm=15,
#                 dpi=220,
#                 save_dir="./qr_codes"
#             )
#             print(f"生成: {hex_data}")
            
#             img = cv2.imread(path)
#             if img is None:
#                 print("无法加载图片")
#                 continue
            
#             print("在便携屏显示2秒...")
#             node.show_qr(img, 200)
            
#             print("\n请对准摄像头，等待识别...")
#             node.detected_data = None
            
#             start_time = time.time()
#             while time.time() - start_time < 10:
#                 if node.detected_data:
#                     if node.detected_data == hex_data:
#                         print("✅ 验证成功")
#                     else:
#                         print(f"⚠️  不一致: {node.detected_data}")
#                     break
#                 time.sleep(0.1)
#             else:
#                 print("❌ 识别超时")
                
#     except KeyboardInterrupt:
#         print("\n退出")
#     finally:
#         node.running = False
#         node.destroy_node()
#         rclpy.shutdown()
#         cv2.destroyAllWindows()

# if __name__ == '__main__':
#     main()