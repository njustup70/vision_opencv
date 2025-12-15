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
from .qr_core import QRCoder

os.environ['QT_QPA_PLATFORM'] = 'xcb'

class QRDetectNode(Node):
    def __init__(self):
        super().__init__('qr_detect_node')
        
        self.bridge = CvBridge()
        self.states = {i: "未知" for i in range(1, 13)}
        self.last_qr_data = None
        self.play_process = None
        self.qr_path = None
        self.running = True
        
        # 订阅摄像头
        self.create_subscription(
            Image,
            'camera/image_raw',
            self.image_callback,
            100
        )
        
        print("QR检测节点已启动")
        print("正在监听摄像头...")
        print("按Ctrl+C退出")
    
    def image_callback(self, msg):
        """自动检测二维码"""
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            qr_codes = decode(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            
            if qr_codes:
                data = qr_codes[0].data.decode('utf-8').strip()
                decoded = QRCoder.decode(data)
                
                if decoded:
                    self.states = {i+1: state for i, state in enumerate(decoded)}
                    self.last_qr_data = data
                    
                    # 打印检测结果
                    print(f"\n=== 检测到二维码 ===")
                    print(f"数据: {data}")
                    print("状态:")
                    for pos in range(1, 13):
                        print(f"  位置{pos}: {self.states[pos]}")
                    print("==================\n")
                        
        except:
            pass

def main():
    rclpy.init()
    node = QRDetectNode()
    
    def spin():
        try:
            while node.running and rclpy.ok():
                rclpy.spin_once(node, timeout_sec=0.1)
        except KeyboardInterrupt:
            pass
        except Exception:
            pass
    
    spin_thread = threading.Thread(target=spin, daemon=True)
    spin_thread.start()
    
    # 用户交互
    try:
        print("\n输入命令:")
        print("1. 输入12个状态生成二维码")
        print("2. 按Ctrl+C退出")
        print("-" * 30)
        
        while node.running:
            try:
                import sys
                if sys.stdin.isatty():
                    cmd = input("> ").strip()
                else:
                    line = sys.stdin.readline()
                    if not line:
                        break
                    cmd = line.strip()
                
                if cmd.lower() == 'q':
                    break
                
                states = cmd.split()
                if len(states) == 12:
                    # 生成二维码
                    try:
                        path, hex_data = QRCoder.encode(states)
                        print(f"生成: {os.path.basename(path)} ({hex_data})")
                        node.qr_path = path
                        
                        # 播放
                        if input("播放? (y/n): ").lower() == 'y':
                            duration = input("时长(ms，默认2000): ").strip()
                            duration_ms = int(duration) if duration.isdigit() else 2000
                            
                            def play():
                                img = cv2.imread(path)
                                if img is not None:
                                    cv2.namedWindow("QR Player", cv2.WINDOW_NORMAL)
                                    cv2.setWindowProperty("QR Player", cv2.WND_PROP_FULLSCREEN, 1)
                                    cv2.imshow("QR Player", img)
                                    
                                    start = time.time()
                                    while (time.time() - start) * 1000 < duration_ms:
                                        if cv2.waitKey(1) & 0xFF == ord('q'):
                                            break
                                    
                                    cv2.destroyWindow("QR Player")
                                    print("播放结束")
                            
                            # 在当前线程播放，避免多进程问题
                            play()
                    except Exception as e:
                        print(f"错误: {e}")
                else:
                    print(f"需要12个状态，当前{len(states)}个")
                    
            except KeyboardInterrupt:
                print("\n退出...")
                break
            except EOFError:
                break
                
    finally:
        node.running = False
        spin_thread.join(timeout=1.0)
        node.destroy_node()
        try:
            rclpy.try_shutdown()
        except:
            pass
        cv2.destroyAllWindows()
        
        print("程序结束")

if __name__ == '__main__':
    main()