#!/usr/bin/env python3
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

os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ.pop("XDG_SESSION_TYPE", None)
from .qr_core import QRCoder

class QRPortableTest(Node):
    def __init__(self):
        super().__init__('qr_portable_test')
        self.bridge = CvBridge()
        
        self.qr_data = []
        self.qr_images = []
        
        self.detections = []
        self.is_testing = False
        
        self.subscription = self.create_subscription(
            Image,
            'camera/image_raw',
            self.callback,
            100
        )
        
        print("="*50)
        print("便携屏QR通信测试")
        print("="*50)
    
    def callback(self, msg):
        """检测二维码并解码KFS状态"""
        try:
            if not self.is_testing:
                return
            
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            qr_codes = decode(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            
            if qr_codes:
                data = qr_codes[0].data.decode().strip()
                self.detections.append(data)
                
                # 解码并打印KFS状态
                states = QRCoder.decode(data)
                if states:
                    if len(self.qr_data) >= 2:
                        if data == self.qr_data[0]:
                            print(f"    [QR1] ✓ 检测到")
                            self.print_kfs_status(states, "QR1状态")
                        elif data == self.qr_data[1]:
                            print(f"    [QR2] ✓ 检测到")
                            self.print_kfs_status(states, "QR2状态")
                        else:
                            print(f"    [未知] ⚠️ 检测到: {data[:8]}...")
        except Exception as e:
            pass
    
    def print_kfs_status(self, states, label=""):
        """打印KFS状态"""
        if not states:
            return
        
        if label:
            print(f"      {label}:")
        else:
            print(f"      KFS状态:")
        
        for i in range(0, 12, 6):
            line = "        "
            for j in range(6):
                pos = i + j + 1
                if pos <= 12:
                    state = states[i+j]
                    line += f"{pos:2d}:{state}  "
            print(line)
    
    def create_portable_window(self):
        """在便携屏上创建窗口"""
        screen = {
            'width': 2160,
            'height': 1440,
            'x': 2560,
            'y': 0,
            'name': 'XWAYLAND2'
        }
        
        print(f"便携屏: {screen['name']} {screen['width']}x{screen['height']}")
        print(f"位置: ({screen['x']}, {screen['y']})")
        
        window_name = "QR Portable Test"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(window_name, screen['x'], screen['y'])
        cv2.resizeWindow(window_name, screen['width'], screen['height'])
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, 1)
        
        blank = np.ones((screen['height'], screen['width'], 3), dtype=np.uint8) * 255
        
        cv2.imshow(window_name, blank)
        cv2.waitKey(100)
        
        return window_name, screen, blank
    
    def show_qr_code(self, window_name, screen, blank, qr_img, qr_index, duration_ms):
        """显示二维码"""
        if qr_img is None:
            print(f"⚠️  QR{qr_index+1}图片为空")
            return False
        
        try:
            qr_h, qr_w = qr_img.shape[:2]
            print(f"  QR{qr_index+1}原始尺寸: {qr_w}x{qr_h}像素")
            
            x = (screen['width'] - qr_w) // 2
            y = (screen['height'] - qr_h) // 2
            print(f"  显示位置: ({x}, {y})")
            
            frame = blank.copy()
            frame[y:y+qr_h, x:x+qr_w] = qr_img
            
            label = f"QR Code {qr_index + 1}"
            cv2.putText(frame, label, (x + 20, y + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
            
            size_text = f"Size: {qr_w}x{qr_h}px (Original)"
            cv2.putText(frame, size_text, (x + 20, y + qr_h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            cv2.imshow(window_name, frame)
            cv2.waitKey(10)
            
            print(f"  ✅ 显示{label} ({duration_ms}ms)")
            
            start_time = time.time()
            while (time.time() - start_time) * 1000 < duration_ms:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("用户中断")
                    return False
                time.sleep(0.001)
            
            return True
            
        except Exception as e:
            print(f"❌ 显示错误: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def print_qr_info(self, hex_str, label):
        """打印二维码信息"""
        states = QRCoder.decode(hex_str)
        print(f"\n{label}: {hex_str}")
        if states:
            self.print_kfs_status(states, f"{label} KFS状态")
        else:
            print(f"  {label} 解码失败")
    
    def run_portable_test(self):
        """运行便携屏测试"""
        print("\n生成测试二维码...")
        
        try:
            # 生成QR1
            qr1_states = ["空", "R1", "空", "R2", "空", "空", "空", "空", "空", "空", "空", "空"]
            print(f"QR1状态序列: {qr1_states}")
            path1, data1 = QRCoder.encode(
                qr1_states,
                size_cm=7,
                save_dir="./test_qr_codes",
                dpi=250
            )
            img1 = cv2.imread(path1)
            
            # 生成QR2
            qr2_states = ["R2", "假", "R1", "空", "R2", "假", "空", "R1", "假", "R2", "空", "R1"]
            print(f"\nQR2状态序列: {qr2_states}")
            path2, data2 = QRCoder.encode(
                qr2_states,
                size_cm=7,
                save_dir="./test_qr_codes",
                dpi=250
            )
            img2 = cv2.imread(path2)
            
            if img1 is None or img2 is None:
                print("❌ 无法加载二维码图片")
                return
            
            self.qr_images = [img1, img2]
            self.qr_data = [data1, data2]
            
            self.print_qr_info(data1, "QR1")
            self.print_qr_info(data2, "QR2")
            
        except Exception as e:
            print(f"❌ 生成失败: {e}")
            return
        
        print("\n创建便携屏窗口...")
        window_name, screen, blank = self.create_portable_window()
        
        time.sleep(1)
        
        print("\n3秒后开始测试...")
        for i in range(3, 0, -1):
            print(f"  {i}秒后启动")
            time.sleep(1)
        
        print("\n" + "="*50)
        print("开始50次通信测试")
        print("每轮: QR1(50ms) → QR2(150ms) → 间隔1秒")
        print("="*50)
        
        total = 0
        success = 0
        qr1_count = 0
        qr2_count = 0
        
        for i in range(50):
            print(f"\n第{i+1:02d}/50次:")
            
            self.is_testing = True
            self.detections = []
            
            # 显示QR1
            print("  QR1显示中...")
            self.show_qr_code(window_name, screen, blank, self.qr_images[0], 0, 50)
            
            # 显示QR2
            print("  QR2显示中...")
            self.show_qr_code(window_name, screen, blank, self.qr_images[1], 1, 150)
            
            self.is_testing = False
            
            qr1_detected = self.qr_data[0] in self.detections
            qr2_detected = self.qr_data[1] in self.detections
            
            total += 1
            if qr1_detected:
                qr1_count += 1
            if qr2_detected:
                qr2_count += 1
            
            if qr1_detected and qr2_detected:
                success += 1
                print(f"  ✅ 通信成功")
            else:
                print(f"  ❌ 失败 QR1:{'✓' if qr1_detected else '✗'} "
                      f"QR2:{'✓' if qr2_detected else '✗'}")
            
            qr1_times = self.detections.count(self.qr_data[0])
            qr2_times = self.detections.count(self.qr_data[1])
            print(f"  检测统计: QR1检测{qr1_times}次, QR2检测{qr2_times}次")
            
            if i < 49:
                print(f"  间隔1秒...")
                cv2.imshow(window_name, blank)
                cv2.waitKey(1)
                time.sleep(1)
            
            # 每5次显示统计
            if (i + 1) % 5 == 0:
                rate = (success / total * 100) if total > 0 else 0
                print(f"\n📊 当前统计 ({total}次):")
                print(f"  成功率: {rate:.1f}%")
                print(f"  QR1识别: {qr1_count}/{total}")
                print(f"  QR2识别: {qr2_count}/{total}")
                print("-" * 30)

        print("\n" + "="*50)
        print("测试完成")
        print("="*50)
        
        if total > 0:
            success_rate = success / total * 100
            print(f"📈 最终结果:")
            print(f"  总通信: {total}次")
            print(f"  成功: {success}次")
            print(f"  失败: {total - success}次")
            print(f"  成功率: {success_rate:.1f}%")
            print(f"  QR1识别率: {qr1_count/total*100:.1f}%")
            print(f"  QR2识别率: {qr2_count/total*100:.1f}%")
            
            # 打印最终的KFS状态
            print(f"\n🔍 最终KFS状态:")
            self.print_qr_info(self.qr_data[0], "QR1预期")
            self.print_qr_info(self.qr_data[1], "QR2预期")

        print("="*50)
        
        print("\n3秒后关闭窗口...")
        time.sleep(3)
        cv2.destroyWindow(window_name)

def main():
    rclpy.init()
    
    try:
        print("初始化测试节点...")
        node = QRPortableTest()
        
        spin_thread = threading.Thread(
            target=rclpy.spin,
            args=(node,),
            daemon=True
        )
        spin_thread.start()
        
        time.sleep(1)
        
        node.run_portable_test()
        
        node.destroy_node()
        
    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as e:
        print(f"测试异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        rclpy.shutdown()
        cv2.destroyAllWindows()
        print("\n测试结束")

if __name__ == '__main__':
    main()