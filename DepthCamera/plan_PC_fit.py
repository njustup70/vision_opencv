import numpy as np
import open3d as o3d
from image_geometry import PinholeCameraModel
import sensor_msgs_py.point_cloud2 as pc2

timelist = [0]*10
timeListHead = 0

def timetest():
    import time
    global timelist, timeListHead
    timelist[timeListHead] = time.time()
    print(f"time interval: {timelist[timeListHead]-timelist[(timeListHead-1)%10]:.3f} s")
    print(f"10 time average interval: {(timelist[timeListHead]-timelist[(timeListHead+1)%10])/10:.3f} s")
    timeListHead = (timeListHead + 1) % 10

def depth_to_point_cloud(depth_img, camera_model, depth_scale=1.0):
    """将深度图转换为Open3D点云"""
    h, w = depth_img.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth_img / depth_scale
    valid = (z > 0) & np.isfinite(z)
    x = (u[valid] - camera_model.cx()) * z[valid] / camera_model.fx()
    y = (v[valid] - camera_model.cy()) * z[valid] / camera_model.fy()
    points = np.stack((x, y, z[valid]), axis=-1)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    return pcd, valid

def region_growing_plane(point_cloud, seed_idx, nb_neighbors=50, angle_threshold=5.0, distance_threshold=0.02):
    """局部平面生长算法"""
    # 基于 KNN 邻域为点云中每个点估计局部表面法向量
    point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=nb_neighbors))
    normals = np.asarray(point_cloud.normals)
    points = np.asarray(point_cloud.points)
    visited = np.zeros(len(points), dtype=bool)
    # 构建 KD-Tree，用于快速邻域搜索
    kd_tree = o3d.geometry.KDTreeFlann(point_cloud)
    plane_indices = [seed_idx]
    queue = [seed_idx]
    visited[seed_idx] = True
    plane_normal = normals[seed_idx] # 初始平面法向量
    plane_point = points[seed_idx] # 初始平面上的点
    # 区域生长
    # 不断从队列中取出当前点，并检查其邻域是否可并入同一平面
    while queue:
        current = queue.pop(0)
        # 邻域查询
        _, idx, _ = kd_tree.search_knn_vector_3d(points[current], nb_neighbors)
        # 去掉已经访问过的点
        idx = np.array(idx)
        mask = ~visited[idx]
        idx = idx[mask]
        if len(idx) == 0:
            continue
        cur_normal = normals[current]           # shape (3,)
        neigh_normals = normals[idx]            # shape (K, 3)
        # 计算邻域点法向量与当前点法向量的点积，反映法向量夹角余弦值
        dots = np.einsum("ij,j->i", neigh_normals, cur_normal)
        # 数值稳定性处理，防止反三角函数越界
        dots = np.clip(dots, -1.0, 1.0)
        # 反三角计算法向量夹角
        normal_diff = np.degrees(np.arccos(dots))
        vecs = points[idx] - plane_point
        # 计算点到平面的距离
        dist = np.abs(vecs @ plane_normal)
        good = (normal_diff < angle_threshold) & (dist < distance_threshold)
        good_idx = idx[good]
        visited[good_idx] = True
        lst = good_idx.tolist()
        queue.extend(lst)
        plane_indices.extend(lst)
    # 拟合平面
    selected_points = points[plane_indices]
    centroid = np.mean(selected_points, axis=0)
    # 计算协方差矩阵并进行SVD分解以获取法向量
    cov = np.cov((selected_points - centroid).T)
    _, _, vh = np.linalg.svd(cov)
    normal = vh[2, :]
    d = -np.dot(normal, centroid)
    seed_point = points[seed_idx]
    #d = -np.dot(normal, seed_point)
    #print(f"if seed in plane:{seed_point in selected_points}")
    return plane_indices, (normal[0], normal[1], normal[2], d), seed_idx, selected_points, seed_point

def get_plane_corners(points_3d):
    """求局部平面四个3D角点"""
    if len(points_3d) < 4:
        return np.array([])
    hull, _ = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_3d)).compute_convex_hull()
    hull_vertices = np.asarray(hull.vertices)
    if len(hull_vertices) < 4:
        return np.array([])
    return np.asarray(hull_vertices[:4])

def fit_plane_from_depth(depth_img, camera_model, u, v, depth_scale=1.0):
    """
    从深度图中指定像素(u,v)出发，找到该平面并返回平面方程及角点
    """
    # 转点云
    pcd, valid_mask = depth_to_point_cloud(depth_img, camera_model, depth_scale)
    h, w = depth_img.shape
    if not valid_mask[int(v), int(u)]:
        raise ValueError("该点无效或深度为0")

    # 找到种子点在点云中的索引
    valid_indices = np.argwhere(valid_mask.flatten()).flatten()

    #直接新加点,平均2.2s
    target_point = np.array([
        (u - camera_model.cx()) * depth_img[int(v), int(u)] / (camera_model.fx() * depth_scale),
        (v - camera_model.cy()) * depth_img[int(v), int(u)] / (camera_model.fy() * depth_scale),
        depth_img[int(v), int(u)] / depth_scale
    ])
    pcd.points.append(target_point)
    seed_idx = len(pcd.points) - 1


    # 区域生长平面
    plane_indices, plane_model, seed_idx, selected_points, seed_point = region_growing_plane(pcd, seed_idx)

    # 角点计算(几乎不耗时)
    plane_points = np.asarray(pcd.points)[plane_indices]
    plane_points = np.vstack([plane_points, np.asarray(pcd.points)[seed_idx]])
    corner_3d = get_plane_corners(plane_points)
    if corner_3d.size == 0:
        raise ValueError("未检测到足够平面点")
    corner_2d = np.array([camera_model.project3dToPixel(pt) for pt in corner_3d])

    return {
        "plane_model": plane_model,
        "corner_3d": corner_3d,
        "corner_2d": corner_2d,
        "selected_points": selected_points,
        "seed_point": seed_point
    }

if __name__ == '__main__':
    import rclpy
    from rclpy.node import Node
    import time
    import cv2
    from sensor_msgs.msg import Image, CameraInfo, PointCloud2
    from cv_bridge import CvBridge
    import yaml

    def load_camera_info(yaml_path: str) -> CameraInfo:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        msg = CameraInfo()
        msg.width  = data['image_width']
        msg.height = data['image_height']
        msg.distortion_model = data['distortion_model']
        msg.header.frame_id = data.get('header', {}).get('frame_id', '')
        def to_float_list(x):
            if isinstance(x, str):
                x = x.strip('[]').replace(',', ' ').split()
            return [float(i) for i in x]
        msg.d = to_float_list(data['distortion_coefficients']['data'])
        msg.k = to_float_list(data['camera_matrix']['data'])
        msg.r = to_float_list(data['rectification_matrix']['data'])
        msg.p = to_float_list(data['projection_matrix']['data'])
        msg.binning_x = data.get('binning_x', 0)
        msg.binning_y = data.get('binning_y', 0)
        roi = data.get('roi', {})
        msg.roi.x_offset = roi.get('x_offset', 0)
        msg.roi.y_offset = roi.get('y_offset', 0)
        msg.roi.height = roi.get('height', 0)
        msg.roi.width = roi.get('width', 0)
        msg.roi.do_rectify = roi.get('do_rectify', False)
        return msg

    def pix_to_cam(u, v, depth, model):
        ray = model.projectPixelTo3dRay((u, v))
        muit = 1.0 / ray[2]
        X = ray[0] * muit * depth
        Y = ray[1] * muit * depth
        Z = ray[2] * muit * depth # Z = depth
        return X, Y, Z

    class DepthCamera:
        def __init__(self):
            self.bridge = CvBridge()
            self.model_d = PinholeCameraModel()
            self.model_c = PinholeCameraModel()

        def loadCameraInfo(self, info_d = None, info_c = None, info_d2c = None):
            if info_d is None:
                self.model_d.fromCameraInfo(load_camera_info('DepthCamera/depth_camera_info.yaml'))
            else:
                self.model_d.fromCameraInfo(info_d)

            if info_c is None:
                self.model_c.fromCameraInfo(load_camera_info('DepthCamera/color_camera_info.yaml'))
            else:
                self.model_c.fromCameraInfo(info_c)

            if info_d2c is None:
                with open('DepthCamera/depth_to_color_info.yaml', 'r') as f:
                    data = yaml.safe_load(f)
                rot = data['depth_to_color_extrinsics']['rotation']['data']
                trans = data['depth_to_color_extrinsics']['translation']['data']
            else:
                rot = info_d2c['rotation']
                trans = info_d2c['translation']
            self.d2c_r = np.array(rot).reshape(3, 3)
            self.d2c_t = np.array(trans).reshape(3, 1)

    class PixelToCamera(Node):
        def __init__(self):
            super().__init__('pixel_to_camera')
            self.info_msg = None
            self.get_logger().info('Waiting for /camera/depth/camera_info...')
            try:
                self.info_msg = self.wait_for_camera_info()
                self.get_logger().info('Loaded camera info from topic.')
            except TimeoutError:
                self.get_logger().warn('Timeout waiting for /camera/depth/camera_info, loading from YAML instead.')
            self.depth_camera = DepthCamera()
            self.depth_camera.loadCameraInfo(info_d=self.info_msg)
            self.create_subscription(Image, '/camera/depth/image_raw', self.depth_callback, 10)
            self.create_subscription(Image, '/camera/color/image_raw', self.color_callback, 10)
            self.create_subscription(PointCloud2, '/camera/depth/points', self.pc_callback, 10)
            self.get_logger().info('Waiting for camera_info and depth frames...')
            self.depth_img = None
            self.color_img = None
            self.point_cloud = None
            self.point = (424,240)

            cv2.namedWindow("Color Image")
            cv2.setMouseCallback("Color Image", self.mouse_callback)

        def wait_for_camera_info(self, timeout_sec=1.0):
            #阻塞等待一次 /camera/depth/camera_info 消息
            future = rclpy.task.Future()
            def callback(msg):
                if not future.done():
                    future.set_result(msg)
            self.create_subscription(CameraInfo, '/camera/depth/camera_info', callback, 10)
            rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_sec)
            if not future.done():
                raise TimeoutError("CameraInfo timeout")
            return future.result()

        def color_callback(self, msg):
            self.color_img = msg
            #self.timetest()

        def pc_callback(self, msg):
            self.point_cloud = pc2.read_points_numpy(msg, field_names=("x","y","z"))

        def depth_callback(self, msg):
            if self.color_img is None:
                return
            if self.point_cloud is None:
                return
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
                self.get_logger().warn(f"Plane fitting failed: {e}")
                cv2.imshow("Color Image", color_resized)
                cv2.waitKey(1)
                return
            plane_model = result["plane_model"]
            points_3d = result["corner_3d"]
            points_2d = result["corner_2d"]
            selected_points = result["selected_points"]
            seed_point = result["seed_point"]
            px, py = self.depth_camera.model_d.project3dToPixel(seed_point)
            cv2.circle(color_resized, (int(px), int(py)), 5, (65535,0,65535), -1) # 红色圆点表示种子点
            #将选中的平面点投影到图像上
            for pt in selected_points:
                px, py = self.depth_camera.model_d.project3dToPixel(pt)
                cv2.circle(color_resized, (int(px), int(py)), 2, (0,0,65535), -1) # 蓝色圆点
            for i in range(4):
                px, py = points_2d[i]
                x, y, z = points_3d[i]
                cv2.circle(color_resized, (int(px), int(py)), 5, (0,65535,0), -1) # 绿色圆点
                cv2.putText(color_resized, f"({x:.3f},{y:.3f},{z:.3f})", (int(px)+10, int(py)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,65535,0), 1)
                cv2.putText(color_resized, f"({int(px)},{int(py)})", (int(px)+10, int(py)+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,65535,0), 1)
            # 画出平面边界线
            for i in range(4):
                pt1 = (int(points_2d[i][0]), int(points_2d[i][1]))
                pt2 = (int(points_2d[(i+1)%4][0]), int(points_2d[(i+1)%4][1]))
                cv2.line(color_resized, pt1, pt2, (0,65535,0), 2)
            cv2.imshow("Color Image", color_resized)
            cv2.waitKey(1)

        def mouse_callback(self, event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                depth = self.depth_camera.bridge.imgmsg_to_cv2(self.depth_img, desired_encoding='passthrough').astype(np.uint16)
                depth = depth[y, x] / 1000.0  # 转换为米
                self.point = (x, y)
                self.get_logger().info(f"point chosen at ({x}, {y})")

    def main():
        rclpy.init()
        node = PixelToCamera()
        rclpy.spin(node)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
