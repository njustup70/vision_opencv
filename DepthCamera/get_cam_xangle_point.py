# 相机俯角测量
# 只包含单点的平面法向量测量
import numpy as np
from plan_PC_fit import region_growing_plane, depth_to_point_cloud, timetest

def fit_plane_from_depth(depth_img, camera_model, u, v, depth_scale=1.0, points = None):
    """
    从深度图中指定像素(u,v)出发，找到该平面法向量
    """
    # 转点云
    if points is None:
        pcd, valid_mask = depth_to_point_cloud(depth_img, camera_model, depth_scale)
        h, w = depth_img.shape
        if not valid_mask[int(v), int(u)]:
            raise ValueError("该点无效或深度为0")

    #直接新加点,平均2.2s
    target_point = np.array([
        (u - camera_model.cx()) * depth_img[int(v), int(u)] / (camera_model.fx() * depth_scale),
        (v - camera_model.cy()) * depth_img[int(v), int(u)] / (camera_model.fy() * depth_scale),
        depth_img[int(v), int(u)] / depth_scale
    ])
    pcd.points.append(target_point)
    seed_idx = len(pcd.points) - 1

    # 区域向量
    _, plane_model, _, _, _ = region_growing_plane(pcd, seed_idx)
    seed_normal = np.array(plane_model[:3])
    if seed_normal[2] > 0:
        seed_normal = -seed_normal  # 保持法向量朝着相机

    return {
        "seed_normal": seed_normal
    }

if __name__ == '__main__':
    import rclpy
    import time
    import cv2
    from sensor_msgs.msg import Image
    import threading
    from rclpy.qos import qos_profile_sensor_data
    from DepthCamera import DepthCamera, pix_to_cam, DepthCamNode

    class GetCameraXAngle(DepthCamNode):
        def __init__(self):
            super().__init__('get_X_angle_node')
            self.create_subscription(Image, '/camera/depth/image_raw', self.depth_callback, qos_profile=qos_profile_sensor_data)
            self.create_subscription(Image, '/camera/color/image_raw', self.color_callback, qos_profile=qos_profile_sensor_data)
            self.get_logger().info('Waiting for camera_info and depth frames...')
            self.depth_img = None
            self.color_img = None
            self.final_img = None
            self.point = (424,240)
            self.depth_ready = threading.Event()

            cv2.namedWindow("Color Image", cv2.WINDOW_NORMAL)
            cv2.setMouseCallback("Color Image", self.mouse_callback)
            threading.Thread(target=self.working_process, daemon=True).start()

        def color_callback(self, msg):
            self.color_img = msg
            #self.timetest()

        def depth_callback(self, msg):
            self.depth_img = msg
            self.depth_ready.set()
            if self.final_img is not None:
                cv2.imshow("Color Image", self.final_img)
                cv2.waitKey(1)

        def working_process(self):
            with open("DepthCamera/xangle.txt", "a+") as f:
                if f.tell() == 0:
                    f.write("# 这是相机俯角标定结果\n")
                f.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + "\n")
            while not self.color_img:
                time.sleep(0.01)
            while True:
                self.depth_ready.wait()
                msg = self.depth_img
                self.depth_ready.clear()
                timetest()
                self.depth_img = msg
                cv2_color_img = self.depth_camera.bridge.imgmsg_to_cv2(self.color_img, desired_encoding='passthrough')
                cv2_depth_img = self.depth_camera.bridge.imgmsg_to_cv2(self.depth_img, desired_encoding='passthrough').astype(np.uint16)
                color_resized = cv2.resize(cv2_color_img, (cv2_depth_img.shape[1], cv2_depth_img.shape[0]), interpolation=cv2.INTER_LINEAR)
                # 深度图叠加到彩色图，颜色表示深度
                depth_colored = cv2.applyColorMap(cv2.convertScaleAbs(cv2_depth_img, alpha=0.03), cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(color_resized, 0.6, depth_colored, 0.4, 0)
                color_resized = overlay
                u, v = self.point
                depth = cv2_depth_img[int(v), int(u)] / 1000.0  # 转换为米
                #print(cv2_color_img.shape, color_resized.shape)
                cv2.circle(color_resized, (int(u), int(v)), 5, (65535,65535,0), -1) # 黄色圆点
                # 显示圆点坐标及深度
                x,y,z = pix_to_cam(u, v, depth, self.depth_camera.model_d)
                if z != 0:
                    cv2.putText(color_resized, f"({x:.3f},{y:.3f},{z:.3f})", (int(u)+10, int(v)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (65535,65535,0), 1)
                else:
                    cv2.putText(color_resized, f"(None)", (int(u)+10, int(v)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (65535,65535,0), 1)
                cv2.putText(color_resized, f"({u},{v})", (int(u)+10, int(v)+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (65535,65535,0), 1)
                try:
                    result = fit_plane_from_depth(cv2_depth_img, self.depth_camera.model_d, u, v, depth_scale=1000.0)
                except Exception as e:
                    self.get_logger().warn(f"failed: {e}")
                    self.final_img = color_resized
                    continue
                seed_normal = result["seed_normal"]
                with open("DepthCamera/xangle.txt", "a") as f:
                    f.write(f"{seed_normal[0]:.6f},{seed_normal[1]:.6f},{seed_normal[2]:.6f}\n")
                px, py, pz = seed_normal + np.array([x, y, z])
                edu, edv = self.depth_camera.model_d.project3dToPixel((px, py, pz))
                cv2.circle(color_resized, (int(u), int(v)), 5, (65535,0,65535), -1) # 红色圆点表示种子点
                cv2.putText(color_resized, f"Normal: ({seed_normal[0]:.3f},{seed_normal[1]:.3f},{seed_normal[2]:.3f})", (int(u)+10,int(v)-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (65535,0,65535), 2)
                # 画出坐标轴及法向量方向
                x_axis_3d = np.array([x+1.0,y,z])
                y_axis_3d = np.array([x,y+1.0,z])
                z_axis_3d = np.array([x,y,z+1.0])
                x_axis_2d = self.depth_camera.model_d.project3dToPixel(x_axis_3d)
                y_axis_2d = self.depth_camera.model_d.project3dToPixel(y_axis_3d)
                z_axis_2d = self.depth_camera.model_d.project3dToPixel(z_axis_3d)
                cv2.arrowedLine(color_resized, (int(u), int(v)), (int(edu), int(edv)), (0,65535,65535), 2, tipLength=0.2)
                cv2.arrowedLine(color_resized, (int(u), int(v)), (int(x_axis_2d[0]), int(x_axis_2d[1])), (0,65535,65535), 2, tipLength=0.2)
                cv2.arrowedLine(color_resized, (int(u), int(v)), (int(y_axis_2d[0]), int(y_axis_2d[1])), (0,65535,65535), 2, tipLength=0.2)
                cv2.arrowedLine(color_resized, (int(u), int(v)), (int(z_axis_2d[0]), int(z_axis_2d[1])), (0,65535,65535), 2, tipLength=0.2)
                cv2.putText(color_resized, "X", (int(x_axis_2d[0]-10), int(x_axis_2d[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,65535,65535), 2)
                cv2.putText(color_resized, "Y", (int(y_axis_2d[0]-10), int(y_axis_2d[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,65535,65535), 2)
                cv2.putText(color_resized, "Z", (int(z_axis_2d[0]-10), int(z_axis_2d[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,65535,65535), 2)
                self.final_img = color_resized

        def mouse_callback(self, event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                depth = self.depth_camera.bridge.imgmsg_to_cv2(self.depth_img, desired_encoding='passthrough').astype(np.uint16)
                depth = depth[y, x] / 1000.0  # 转换为米
                self.point = (x, y)
                self.get_logger().info(f"point chosen at ({x}, {y})")

    def main():
        rclpy.init()
        node = GetCameraXAngle()
        rclpy.spin(node)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
