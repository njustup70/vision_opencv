import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from pyzbar.pyzbar import decode
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import os
import subprocess
import time

# 二进制与状态映射
BIN_TO_STATUS = {
    "00": "空",
    "01": "R1KFS",
    "10": "R2KFS",
    "11": "假KFS"
}

class QrKfsDecoderNode(Node):
    def __init__(self):
        super().__init__('qr_kfs_decoder_node')
        
        # 配置参数
        self.declare_parameter("camera_topic", "/usb_cam/image_raw")
        self.camera_topic = self.get_parameter("camera_topic").value

        # 优化QR码识别参数
        self.qr_config = {
            "scale": 1.5,
            "blur": (3, 3),
            "threshold": True
        }
        
        # 工具初始化
        self.bridge = CvBridge()
        self.last_decoded = None
        
        # --------------------------
        # 摄像头重连配置（你的参数）
        # --------------------------
        self.camera_device = "/dev/video10" 
        self.usb_hub_pci = "0000:00:14.0"
        self.sudo_password = "qing"
        self.reconnect_count = 0 
        self.max_reconnect = 3 

        self.script_dir = os.path.join(os.path.dirname(__file__), "../../cv_lib/")
        self.unbind_script = os.path.join(self.script_dir, "usb_unbind.sh")
        self.bind_script = os.path.join(self.script_dir, "usb_bind.sh")

        # 订阅相机图像
        self.image_sub = self.create_subscription(
            Image, self.camera_topic, self.image_callback, 10
        )
        
        self.get_logger().info("✅ QR-KFS解析节点启动！支持摄像头自动重连")
        self.get_logger().info(f"📡 订阅相机话题：{self.camera_topic}")
        self.get_logger().info(f"📷 摄像头设备：{self.camera_device}")
        self.get_logger().info("💡 支持识别：纸质QR码 / 屏幕显示的QR码")

    # --------------------------
    # 判断摄像头是否在线
    # --------------------------
    def is_camera_online(self):
        """检查/dev/video10是否存在"""
        return os.path.exists(self.camera_device)

    # --------------------------
    # 重置USB Hub（设备离线时用）
    # --------------------------
    def reset_usb_hub(self):
        self.get_logger().warn(f"⚠️ 开始重置USB Hub（PCI地址：{self.usb_hub_pci}）")
        
        # 检查脚本是否存在
        if not os.path.exists(self.unbind_script) or not os.path.exists(self.bind_script):
            self.get_logger().error(f"❌ 重连脚本不存在！请确认路径：{self.script_dir}")
            return False
        
        try:
            # 1. 卸载USB Hub
            cmd_unbind = (
                f"echo '{self.sudo_password}' | sudo -S sh {self.unbind_script} {self.usb_hub_pci}"
            )
            result = subprocess.run(
                cmd_unbind, shell=True, check=True, capture_output=True, text=True
            )
            self.get_logger().info(f"✅ USB Hub卸载成功")
            time.sleep(2)
            
            # 2. 重新绑定USB Hub
            cmd_bind = (
                f"echo '{self.sudo_password}' | sudo -S sh {self.bind_script} {self.usb_hub_pci}"
            )
            result = subprocess.run(
                cmd_bind, shell=True, check=True, capture_output=True, text=True
            )
            self.get_logger().info(f"✅ USB Hub重新绑定成功")
            time.sleep(3)  # 等待设备初始化
            return True
        
        except subprocess.CalledProcessError as e:
            self.get_logger().error(f"❌ 重置USB失败：{e.stderr}")
            return False
        except Exception as e:
            self.get_logger().error(f"❌ 重置USB异常：{str(e)}")
            return False

    # --------------------------
    # 自动重连摄像头
    # --------------------------
    def reconnect_camera(self):
        self.reconnect_count += 1
        if self.reconnect_count > self.max_reconnect:
            self.get_logger().error(f"❌ 重连失败（已尝试{self.max_reconnect}次），请检查：")
            self.get_logger().error("  1. USB线是否插紧  2. 摄像头是否损坏  3. 换一个USB端口")
            return False
        
        self.get_logger().warn(f"⚠️ 第{self.reconnect_count}次尝试重连摄像头...")
        
        # 情况1：摄像头设备还在线（只是读取失败）
        if self.is_camera_online():
            self.get_logger().info("📌 摄像头设备在线，尝试重启usb_cam节点")
            try:
                # 调用usb_cam节点的重置服务
                subprocess.run(
                    f"echo '{self.sudo_password}' | sudo -S ros2 service call /usb_cam_node/reset std_srvs/srv/Empty",
                    shell=True, check=True, capture_output=True, text=True
                )
                time.sleep(2)
                self.get_logger().info("✅ usb_cam节点重启成功")
                return True
            except Exception as e:
                self.get_logger().error(f"❌ 重启节点失败：{str(e)}")
                return False
        
        # 情况2：摄像头设备离线
        else:
            self.get_logger().info("📌 摄像头设备离线，尝试重置USB Hub")
            if self.reset_usb_hub():
                # 重置后检查设备是否恢复
                if self.is_camera_online():
                    self.get_logger().info(f"✅ 摄像头已恢复（{self.camera_device}重新出现）")
                    self.reconnect_count = 0
                    return True
                else:
                    self.get_logger().error(f"❌ USB重置后仍未找到{self.camera_device}")
                    return False

    def preprocess_image(self, cv_img):
        """图像预处理（适配屏幕反光、纸质模糊）"""
        # 1. 放大图像（小QR码识别率提升）
        h, w = cv_img.shape[:2]
        cv_img = cv2.resize(cv_img, (int(w*self.qr_config["scale"]), int(h*self.qr_config["scale"])))
        
        # 2. 转灰度图
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        
        # 3. 轻微模糊（去除摩尔纹/噪声）
        gray = cv2.GaussianBlur(gray, self.qr_config["blur"], 0)
        
        # 4. 二值化（增强黑白对比，适配屏幕反光）
        if self.qr_config["threshold"]:
            _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return gray

    def decode_qr_to_kfs(self, hex_str):
        """解码QR码中的十六进制字符串为KFS状态+预留位"""
        try:
            # 1. 十六进制转32位二进制（补0至32位）
            total_bin = bin(int(hex_str, 16))[2:].zfill(32)
            if len(total_bin) != 32:
                raise ValueError(f"十六进制无效，转二进制后长度≠32位（实际：{len(total_bin)}）")
            
            # 2. 拆分：前24位=12个位置状态，后8位=预留位
            kfs_bin = total_bin[:24]
            reserve_bits = total_bin[24:]
            
            # 3. 解码12个位置状态（每2位对应一个状态）
            kfs_states = []
            for i in range(12):
                bin_segment = kfs_bin[i*2 : (i+1)*2]
                state = BIN_TO_STATUS.get(bin_segment, "无效")
                kfs_states.append((i+1, state))  # (位置号, 状态)
            
            return kfs_states, reserve_bits
        except Exception as e:
            self.get_logger().error(f"❌ 解码失败：{str(e)}")
            return None, None

    def image_callback(self, msg):
        """接收相机图像，识别QR码并解码（添加重连逻辑）"""
        try:
            # 1. 图像转换（ROS2 Image → OpenCV）
            cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            # 读取成功，重置重连计数器
            if self.reconnect_count > 0:
                self.reconnect_count = 0
                self.get_logger().info("✅ 摄像头正常工作，重连计数器重置")
        except Exception as e:
            self.get_logger().error(f"❌ 图像转换失败（可能掉线）：{str(e)}")
            # 触发自动重连，重连失败则直接返回
            if not self.reconnect_camera():
                return
        
        # 2. 图像预处理（提升识别率）
        gray_img = self.preprocess_image(cv_img)
        
        # 3. 识别QR码（支持多个QR码，取第一个有效）
        qr_codes = decode(gray_img)
        if not qr_codes:
            return  # 未识别到QR码，不输出
        
        # 4. 解码QR码数据
        qr_data = qr_codes[0].data.decode("utf-8").strip()
        if qr_data == self.last_decoded:
            return  # 避免重复输出
        
        self.last_decoded = qr_data
        self.get_logger().info(f"📤 识别到QR码，数据：{qr_data}")
        
        # 5. 解析KFS状态
        kfs_states, reserve_bits = self.decode_qr_to_kfs(qr_data)
        if kfs_states:
            self.get_logger().info("🔍 解码后的KFS状态：")
            for pos, status in kfs_states:
                self.get_logger().info(f"  位置{pos}：{status}")
            self.get_logger().info(f"📌 预留位（8位二进制）：{reserve_bits}")
        
        # 6. 绘制识别框
        for qr in qr_codes:
            pts = np.array([qr.polygon], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(cv_img, [pts], True, (0, 255, 0), 2)

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = QrKfsDecoderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("🛑 QR-KFS解析节点已停止")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
