# 对deform_restore的测试文件
from sensor_msgs.msg import Image, CameraInfo
import numpy as np
import deform_restore

import os
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"

import rclpy
import cv2
from DepthCamera import pix_to_cam, DepthCamNode
from rclpy.qos import qos_profile_sensor_data

class Restore(DepthCamNode):
    def __init__(self):
        super().__init__('restore_node')
        self.create_subscription(Image, '/camera/depth/image_raw', self.depth_callback, qos_profile_sensor_data)
        self.create_subscription(Image, '/camera/color/image_raw', self.color_callback, qos_profile_sensor_data)
        self.get_logger().info('Waiting for camera_info and depth frames...')
        self.depth_img = None
        self.color_img = None
        self.points = [(200,400), (200,450), (300,400), (300,400)]
        self.click_point = 0

        cv2.namedWindow("Color Image", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Color Image", self.mouse_callback)

    def color_callback(self, msg):
        self.color_img = msg
        #self.timetest()

    def depth_callback(self, msg):
        if self.color_img is None:
            return
        #self.timetest()
        self.depth_img = msg
        cv2_color_img = self.depth_camera.bridge.imgmsg_to_cv2(self.color_img, desired_encoding='passthrough')
        cv2_depth_img = self.depth_camera.bridge.imgmsg_to_cv2(self.depth_img, desired_encoding='passthrough').astype(np.uint16)
        color_resized = cv2.resize(cv2_color_img, (cv2_depth_img.shape[1], cv2_depth_img.shape[0]), interpolation=cv2.INTER_LINEAR)
        # 深度图叠加到彩色图，颜色表示深度
        depth_colored = cv2.applyColorMap(cv2.convertScaleAbs(cv2_depth_img, alpha=0.03), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(color_resized, 0.6, depth_colored, 0.4, 0)
        color_resized = overlay
        count = 0
        for (u, v) in self.points:
            if count >= self.click_point:
                break
            depth = cv2_depth_img[int(v), int(u)] / 1000.0  # 转换为米
            #print(cv2_color_img.shape, color_resized.shape)
            cv2.circle(color_resized, (int(u), int(v)), 5, (65535,65535,0), -1) # 黄色圆点
            # 显示所有圆点坐标及深度
            x,y,z = pix_to_cam(u, v, depth, self.depth_camera.model_d)
            if z != 0:
                cv2.putText(color_resized, f"({x:.3f},{y:.3f},{z:.3f})", (int(u)+10, int(v)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (65535,65535,0), 1)
            else:
                cv2.putText(color_resized, f"(None)", (int(u)+10, int(v)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (65535,65535,0), 1)
            cv2.putText(color_resized, f"({u},{v})", (int(u)+10, int(v)+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (65535,65535,0), 1)
            count += 1
        cv2.imshow("Color Image", color_resized)
        cv2.waitKey(1)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            depth = self.depth_camera.bridge.imgmsg_to_cv2(self.depth_img, desired_encoding='passthrough').astype(np.uint16)
            depth = depth[y, x] / 1000.0  # 转换为米
            if depth != 0:
                x_3d, y_3d, z_3d = pix_to_cam(x, y, depth, self.depth_camera.model_d)
                if self.click_point >= len(self.points):
                    self.click_point = 0
                self.points[self.click_point] = (x, y)
                self.get_logger().info(f"point chosen at ({x}, {y})")
                if self.click_point >= len(self.points)-1:
                    needed_points = self.points
                    self.get_logger().info(f"All points chosen: {needed_points}")
                    # 顺时针将点排序
                    center = np.mean(needed_points, axis=0)
                    sorted_points = sorted(needed_points, key=lambda p: np.arctan2(p[1] - center[1], p[0] - center[0]))
                    sorted_3d_points = []
                    for (u, v) in sorted_points:
                        depth = self.depth_camera.bridge.imgmsg_to_cv2(self.depth_img, desired_encoding='passthrough').astype(np.uint16)
                        depth = depth[v, u] / 1000.0  # 转换为米
                        x3d, y3d, z3d = pix_to_cam(u, v, depth, self.depth_camera.model_d)
                        sorted_3d_points.append([x3d, y3d, z3d])
                    self.get_logger().info(f"3D Points: {[[round(x, 2) for x in point] for point in sorted_3d_points]}")
                    sorted_3d_points = np.array(sorted_3d_points, dtype=np.float32)
                    points_2d = deform_restore.trans3DToPlane(sorted_3d_points)
                    warped = deform_restore.ROIRestore(
                        self.depth_camera.bridge.imgmsg_to_cv2(self.color_img, desired_encoding='passthrough'),
                        points_2d
                    )
                    cv2.imshow("Warped Image", warped)
                    cv2.waitKey(0)
                    cv2.destroyWindow("Warped Image")
                self.click_point += 1

def main():
    rclpy.init()
    node = Restore()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
