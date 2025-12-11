from ultralytics import YOLO
import deform_restore as rs
import cv2
import numpy as np

def get_yolo_result(model, img, roi_3d):
    roi_2d = rs.trans3DToPlane(roi_3d)
    roi_img = rs.ROIRestore(img, roi_2d, image_shape=[500,500])

    result = model.predict(source=roi_img, save=False, save_txt=False, conf=0.25, iou=0.45)
    return result, roi_img, roi_2d

if __name__ == "__main__":
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image, CameraInfo
    from cv_bridge import CvBridge
    import numpy as np
    from image_geometry import PinholeCameraModel
    import cv2
    import time
    import threading
    from rclpy.qos import qos_profile_sensor_data

class CameraToPixel(Node):
    def __init__(self):
        super().__init__('camera_to_pixel')
        self.cameraInfoInit = False
        self.bridge = CvBridge()
        self.model = PinholeCameraModel()
        self.PointToDepth = None
        self.depth_data = None
        self.timelist = [0] * 10
        self.timeListHead = 0
        
        self.create_subscription(Image, '/camera/color/image_raw', self.color_callback, qos_profile=qos_profile_sensor_data)

        self.get_logger().info('Waiting for color frames...')
        self.time_s = 0
        self.count = 0
        self.yolo_model = YOLO("best.pt")


    def color_callback(self, msg):
        color_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8').astype(np.uint8)
        roi_3d = np.array([
            [-0.25, -0.25, 0],
            [0.25, -0.25, 0],
            [0.25, 0.25, 0],    
            [-0.25, 0.25, 0]
        ], dtype=np.float32)
        result, roi_img, roi_2d = get_yolo_result(self.yolo_model, color_img, roi_3d)
        # 绘制roi区域
        for i in range(4):
            pt1 = (int(roi_2d[i][0]), int(roi_2d[i][1]))
            pt2 = (int(roi_2d[(i+1)%4][0]), int(roi_2d[(i+1)%4][1]))
            cv2.line(color_img, pt1, pt2, (0, 255, 0), 2)
        scale_x = (roi_2d[1][0] - roi_2d[0][0]) / 500
        scale_y = (roi_2d[3][1] - roi_2d[0][1]) / 500
        if len(result[0].boxes) > 0:
            cls = result[0].boxes.cls[0]  # 第一个检测框的类别
            conf = result[0].boxes.conf[0]  # 第一个检测框的置信度
            label = f"{self.yolo_model.names[int(cls)]}: {conf:.2f}"
            xyxy = result[0].boxes.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy
            x1 = x1 * scale_x + roi_2d[0][0]
            x2 = x2 * scale_x + roi_2d[0][0]
            y1 = y1 * scale_y + roi_2d[0][1]
            y2 = y2 * scale_y + roi_2d[0][1]

            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            cv2.rectangle(color_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        else:
            print("没有检测到目标")
            label = "No detection"
        cv2.putText(color_img, str(label), (int(roi_2d[0][0]), int(roi_2d[0][1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            
        cv2.imshow("Color Image", color_img)
        #async_print(f"Processing time: {time.time() - time_tmp}")
        cv2.waitKey(1)

def main():
    rclpy.init()
    node = CameraToPixel()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()