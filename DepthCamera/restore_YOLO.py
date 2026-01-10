from ultralytics import YOLO
import deform_restore as rs
import cv2
import numpy as np
import yaml

def img_preprocess(img, depression_angle, target_loc, target_direct = 0, target_size = (500,500)): # 默认看正面
    """
    图像进入YOLO的预处理
    
    :param img: 原图像 :type:`np.ndarray`
    :param depression_angle: 相机俯仰角 (nx, ny, nz) :type:`list`
    :param target_loc: 目标3D位置 (x, y, z)
    :param target_direct: 目标方向，默认看正面，看上面传1 :type:`int`
    :param target_size: 目标大小，默认(500,500) :type:`tuple`
    """
    
    up_normal = np.array(depression_angle).reshape(3, 1) # 水平面(上面)法向量
    forward_normal = np.cross(up_normal.reshape(3,), np.array([1,0,0])).reshape(3,1) # 前方向法向量
    if target_direct == 0: # 看正面
        plane_normal = forward_normal
    else: # 看上面
        plane_normal = up_normal
    roi_img, roi_2d = rs.deformRestore(img, target_loc, (target_size[0], target_size[1], plane_normal.reshape(3,)), image_shape=target_size)
    return roi_img, roi_2d

def get_yolo_result(model, img):

    result = model.predict(source=img, save=False, save_txt=False, conf=0.25, iou=0.45)
    return result

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

class RestoreYOLO(Node):
    def __init__(self):
        print(" 1  ")
        super().__init__('restore_YOLO_node')
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

        cv2.namedWindow("ROI Image", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Color Image", cv2.WINDOW_NORMAL)

        with open('DepthCamera/attitude_info.yaml', 'r') as f:
            data = yaml.safe_load(f)
        self.depression_angle = data['attitude_angle']['depression_angle']['data']


    def color_callback(self, msg):
        color_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8').astype(np.uint8)
        roi_img, roi_2d = img_preprocess(color_img, self.depression_angle, target_loc=(-200,-200,1000), target_direct=0, target_size=(500,500))
        result = get_yolo_result(self.yolo_model, roi_img)
        # 绘制roi区域
        for i in range(4):
            pt1 = (int(roi_2d[i][0]), int(roi_2d[i][1]))
            pt2 = (int(roi_2d[(i+1)%4][0]), int(roi_2d[(i+1)%4][1]))
            cv2.line(color_img, pt1, pt2, (0, 255, 0), 2)
        scale_x = (roi_2d[1][0] - roi_2d[0][0]) / 500
        scale_y = (roi_2d[3][1] - roi_2d[0][1]) / 500
        if len(result[0].boxes) > 0:
            cls1 = result[0].boxes.cls[0]  # 第一个检测框的类别
            conf1 = result[0].boxes.conf[0]  # 第一个检测框的置信度
            label = f"{self.yolo_model.names[int(cls1)]}: {conf1:.2f}"
            if len(result[0].boxes.cls) > 1:
                cls2 = result[0].boxes.cls[1]
                conf2 = result[0].boxes.conf[1]
                label2 = f"{self.yolo_model.names[int(cls2)]}: {conf2:.2f}"
                label = label + " | " + label2
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
        cv2.imshow("ROI Image", roi_img)
        #async_print(f"Processing time: {time.time() - time_tmp}")
        cv2.waitKey(1)

def main():
    rclpy.init()
    node = RestoreYOLO()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()