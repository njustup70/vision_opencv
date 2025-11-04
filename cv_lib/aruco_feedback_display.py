import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from visualization_msgs.msg import MarkerArray

# 解决ROS2与OpenCV窗口兼容问题
cv2.ocl.setUseOpenCL(False)

class ArucoFeedbackDisplay(Node):
    def __init__(self):
        super().__init__('aruco_feedback_display')
        
        # 关键配置（与你的系统完全一致）
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_1000)
        self.display_size = (300, 300)  # 窗口大小（宽x高）
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.last_id = None  # 记录上一个ID，避免重复刷新
        
        # 订阅Aruco识别结果（话题与你的节点一致）
        self.subscription = self.create_subscription(
            MarkerArray, '/aruco_markers', self.callback, 10
        )
        self.subscription  # 防止未使用警告
        
        # 初始化显示窗口（非阻塞模式）
        cv2.namedWindow('Aruco识别反馈', cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
        cv2.resizeWindow('Aruco识别反馈', *self.display_size)
        
        self.get_logger().info("✅ 免编译Aruco反馈节点启动！识别到码自动显示对应ID")

    def generate_aruco(self, marker_id):
        """生成指定ID的7x7 Aruco码（带白色背景+居中）"""
        # 生成原始Aruco码（60x60像素，保证清晰度）
        aruco_raw = cv2.aruco.generateImageMarker(self.aruco_dict, marker_id, 60)
        # 缩放至窗口80%大小（留边更美观）
        scale = min(self.display_size[0], self.display_size[1]) * 0.8 / 60
        aruco_scaled = cv2.resize(aruco_raw, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        
        # 创建白色背景图
        bg = np.ones((self.display_size[1], self.display_size[0]), dtype=np.uint8) * 255
        # 居中放置Aruco码
        h, w = aruco_scaled.shape
        x0 = (self.display_size[0] - w) // 2
        y0 = (self.display_size[1] - h) // 2
        bg[y0:y0+h, x0:x0+w] = aruco_scaled
        
        # 叠加ID文本（底部居中）
        text = f"识别到 ID: {marker_id}"
        text_size = cv2.getTextSize(text, self.font, 0.9, 2)[0]
        text_x = (self.display_size[0] - text_size[0]) // 2
        text_y = self.display_size[1] - 20
        cv2.putText(bg, text, (text_x, text_y), self.font, 0.9, (0,0,0), 2, cv2.LINE_AA)
        
        return bg

    def callback(self, msg):
        """接收识别结果，实时更新显示"""
        # 获取当前识别到的第一个有效ID（多个码时取最新）
        current_id = msg.markers[0].id if msg.markers else None
        
        # 只有ID变化时才刷新（避免卡顿）
        if current_id != self.last_id:
            self.last_id = current_id
            
            if current_id is not None:
                # 生成并显示对应Aruco码
                aruco_img = self.generate_aruco(current_id)
                cv2.imshow('Aruco识别反馈', aruco_img)
                self.get_logger().info(f"📢 已显示 ID={current_id} 的Aruco码")
            else:
                # 未识别到码时显示提示
                empty_img = np.ones(self.display_size[::-1], dtype=np.uint8) * 255
                tip_text = "未识别到任何Aruco码"
                tip_size = cv2.getTextSize(tip_text, self.font, 0.9, 2)[0]
                tip_x = (self.display_size[0] - tip_size[0]) // 2
                tip_y = self.display_size[1] // 2
                cv2.putText(empty_img, tip_text, (tip_x, tip_y), self.font, 0.9, (100,100,100), 2, cv2.LINE_AA)
                cv2.imshow('Aruco识别反馈', empty_img)
        
        cv2.waitKey(1)

    def destroy_node(self):
        """关闭窗口，释放资源"""
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = ArucoFeedbackDisplay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("🛑 反馈节点已停止")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
