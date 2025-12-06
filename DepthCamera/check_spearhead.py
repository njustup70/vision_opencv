import yaml
import numpy as np
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

qos = QoSProfile(
    depth=1,  # 只保留最新一帧
    reliability=QoSReliabilityPolicy.RELIABLE,  # 尽量保证丢帧而不阻塞
    history=QoSHistoryPolicy.KEEP_LAST          # 只保存最后 depth 帧
)

# 子区域局部中心（立方体坐标系）
x_centers = np.array([-0.5, -0.3, -0.1, 0.1, 0.3, 0.5])  # m
T_local_centers = np.stack([x_centers,
                            np.zeros_like(x_centers),
                            np.zeros_like(x_centers)], axis=1)

# 尺寸半长
hx, hy, hz = 0.10, 0.10, 0.15  # 200x200x300 mm 左右宽*上下高*前后长 需要适当修改 

def check_spearhead(pc, model, T_box_cam, R_box_cam):
    '''
    检查点云中每个子区域的点数是否满足要求

    :param pc: 点云数据 :type:`sensor_msgs.msg.PointCloud2`
    :param model: 相机模型 :type:`DepthCamera`
    :param T_box_cam: 立方体中心在相机坐标系的位置 :type:`np.ndarray` (右， 下， 前)
    :param R_box_cam: 立方体相对于相机的旋转矩阵 :type:`np.ndarray` (俯角， 右转角)
    :return: 每个子区域是否满足点数要求的布尔列表 :type:`List[bool]`
    '''
    depth_pc = pc2.read_points_numpy(pc, field_names=("x","y","z"))
    counts, _ = filter_rotated_subboxes(depth_pc, T_box_cam, R_box_cam, model)
    counts_check = [c >= 50 for c in counts]  # 每个子区域至少50个点
    return counts_check

def rot_x(deg):
    rad = np.deg2rad(deg)
    return np.array([
        [1,0,0],
        [0,np.cos(rad),-np.sin(rad)],
        [0,np.sin(rad), np.cos(rad)]
    ])

def rot_y(deg):
    rad = np.deg2rad(deg)
    return np.array([
        [ np.cos(rad),0,np.sin(rad)],
        [0,1,0],
        [-np.sin(rad),0,np.cos(rad)]
    ])

def filter_rotated_subboxes(pc_dep, T_box_cam, R_box_cam, model):
    '''
    过滤出落在旋转立方体六个子区域内的点
    :param pts_cam:   Nx3的3D点数组 :type:`np.ndarray`
    :return: 6个布尔数组，表示每个点是否落在对应子区域内 :type:`List[np.ndarray]`
    '''

    results = []
    counts = []

    pc_color = (model.d2c_r @ pc_dep.T).T + model.d2c_t
    pc_rot = pc_color @ R_box_cam.T


    for i in range(6):
        # 子区域中心
        T_i_cam = T_box_cam + R_box_cam @ T_local_centers[i]
        tx, ty, tz = T_i_cam

        # 直接比较，不构造 pts_i
        mask = (
            (np.abs(pc_rot[:,0] - tx) <= hx) &
            (np.abs(pc_rot[:,1] - ty) <= hy) &
            (np.abs(pc_rot[:,2] - tz) <= hz)
        )

        counts.append(mask.sum())
        results.append(pc_color[mask])
    return counts, results


# def project_points(Pc, K):
#     """
#     投影3D点到像素平面
#     Pc: Nx3 相机坐标点，K: 3x3 内参
#     """
#     x = Pc[:,0]
#     y = Pc[:,1]
#     z = Pc[:,2]
#     u = K[0,0]*x/z + K[0,2]
#     v = K[1,1]*y/z + K[1,2]
#     return np.stack([u,v], axis=1)

def project_subboxes(model, T_box_cam, R_box_cam):
    """
    生成每个子框在像素上的8点投影
    
    返回: list of [8×2] 的像素坐标数组
    """
    uv_boxes = []
    # 角点局部坐标
    xs = np.array([+hx, +hx, -hx, -hx, +hx, +hx, -hx, -hx])
    ys = np.array([+hy, -hy, -hy, +hy, +hy, -hy, -hy, +hy])
    zs = np.array([+hz, +hz, +hz, +hz, -hz, -hz, -hz, -hz])
    corners_local = np.stack([xs,ys,zs], axis=1)

    for i in range(6):
        # 子立方体中心在相机坐标系
        T_i_cam = T_box_cam + R_box_cam @ T_local_centers[i]

        # 旋转 + 平移
        Pc = (R_box_cam @ corners_local.T).T + T_i_cam  # 8x3

        # 投影到像素
        uv = np.array([model.project3dToPixel(p) for p in Pc])  # 8x2

        uv_boxes.append(uv)

    return uv_boxes

if __name__ == '__main__':
    import rclpy
    from rclpy.node import Node
    import time
    import cv2
    from image_geometry import PinholeCameraModel
    import sensor_msgs_py.point_cloud2 as pc2
    import threading


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


class DepthCamera:
    def __init__(self):
        self.bridge = CvBridge()
        self.model_d = PinholeCameraModel()
        self.model_c = PinholeCameraModel()

    def loadCameraInfo(self, info_c = None, info_d = None, info_d2c = None):

        if info_c is None:
            self.model_c.fromCameraInfo(load_camera_info('DepthCamera/color_camera_info.yaml'))
        else:
            self.model_c.fromCameraInfo(info_c)

        if info_d is None:
            self.model_d.fromCameraInfo(load_camera_info('DepthCamera/depth_camera_info.yaml'))
        else:
            self.model_d.fromCameraInfo(info_d)

        if info_d2c is None:
            with open('DepthCamera/depth_to_color_info.yaml', 'r') as f:
                data = yaml.safe_load(f)
            rot = data['depth_to_color_extrinsics']['rotation']['data']
            trans = data['depth_to_color_extrinsics']['translation']['data']
        else:
            rot = info_d2c['rotation']
            trans = info_d2c['translation']
        self.d2c_r = np.array(rot).reshape(3, 3)
        self.d2c_t = np.array(trans).reshape(3,)

def resize_intrinsics(model, new_w, new_h):
    old_w = model.width
    old_h = model.height
    sx = new_w / old_w
    sy = new_h / old_h
    K = model.K.copy()
    P = model.P.copy()
    K[0, 0] *= sx    # fx
    K[1, 1] *= sy    # fy
    K[0, 2] *= sx    # cx
    K[1, 2] *= sy    # cy
    P[0, 0] *= sx
    P[1, 1] *= sy
    P[0, 2] *= sx
    P[1, 2] *= sy
    model.width = new_w
    model.height = new_h

    model.K = K
    model.P = P

    model._intrinsicMatrix = K
    model._fullIntrinsicMatrix = K
    model._projectionMatrix = P

    return model

# def get_offset_target(model_d, d2c_r, d2c_t, T_box_cam):
#     T_box_cam = np.asarray(T_box_cam).reshape(3)
#     ud, vd = model_d.project3dToPixel(T_box_cam)
#     T_color = d2c_r @ T_box_cam + d2c_t
#     uc, vc = model_d.project3dToPixel(T_color)
#     dx = uc - ud
#     dy = vc - vd
#     return dx, dy

# def shift_depth_by_offset(depth_img, dx, dy):
#     H, W = depth_img.shape
#     depth_img = depth_img.astype(np.float64)
#     depth_img[depth_img <= 0] = np.nan
#     shifted = np.full_like(depth_img, np.nan)
#     if dx >= 0:
#         xs_src = slice(0, W - dx)
#         xs_dst = slice(dx, W)
#     else:
#         xs_src = slice(-dx, W)
#         xs_dst = slice(0, W + dx)
#     if dy >= 0:
#         ys_src = slice(0, H - dy)
#         ys_dst = slice(dy, H)
#     else:
#         ys_src = slice(-dy, H)
#         ys_dst = slice(0, H + dy)
#     shifted[ys_dst, xs_dst] = depth_img[ys_src, xs_src]
#     return shifted

def get_FPS(timelist, timeListHead):
    time_now = time.time()
    timelist[timeListHead] = time_now
    timeListHead = (timeListHead + 1) % len(timelist)
    time_diff = time_now - timelist[timeListHead]
    if time_diff == 0:
        fps = 0.0
    else:
        fps = len(timelist) / time_diff
    return fps, timelist, timeListHead

class PixelToCamera(Node):
    def __init__(self):
        super().__init__('pixel_to_camera')
        self.info_msg = None
        self.depth_camera = DepthCamera()
        self.depth_camera.loadCameraInfo()
        self.create_subscription(Image, '/camera/color/image_raw', self.color_callback, 10)
        self.create_subscription(PointCloud2, '/camera/depth/points', self.pc_callback, 10)
        self.get_logger().info('Waiting for color frames...')
        self.color_img = None
        self.suit_pc = None
        self.counts = None
        self.shape =[848,480]
        # 立方体中心在相机前方1.0m, 向下0.33m
        self.T_box_cam = np.array([0, 0.33, 1.0])
        # 立方体相对于相机的旋转：俯视0° + 向右0°
        self.R_box_cam = rot_y(00) @ rot_x(-0)
        self.timelist = [0] * 10
        self.timeListHead = 0

        cv2.namedWindow("Color Image")

    def pc_callback(self, msg):
        depth_pc = pc2.read_points_numpy(msg, field_names=("x","y","z"))
        # 用外参偏移点云
        
        self.counts, self.suit_pc = filter_rotated_subboxes(depth_pc, self.T_box_cam, self.R_box_cam, self.depth_camera)

    def color_callback(self, msg):
        if self.suit_pc is None:
            return
        time_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        time0 = time.time()
        self.color_img = msg
        cv2_color_img = self.depth_camera.bridge.imgmsg_to_cv2(self.color_img, desired_encoding='passthrough')
        cv2_img_bgr = cv2.cvtColor(cv2_color_img, cv2.COLOR_RGB2BGR)
        color_resized = cv2.resize(cv2_img_bgr, (self.shape[0], self.shape[1]), interpolation=cv2.INTER_LINEAR)
        self.depth_camera.model_c = resize_intrinsics(
            self.depth_camera.model_c,
            self.shape[0],
            self.shape[1]
        )
        fps, self.timelist, self.timeListHead = get_FPS(self.timelist, self.timeListHead)
        cv2.putText(color_resized, f'FPS: {fps:.2f}', (self.shape[0] - 100, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        uv_boxes = project_subboxes(self.depth_camera.model_c, self.T_box_cam, self.R_box_cam)
        # 转为整数
        for i, uv in enumerate(uv_boxes):
            uv = uv.astype(int)
            # 画边（角点顺序按四边形连接）
            for j in range(4):
                cv2.line(color_resized, tuple(uv[j]), tuple(uv[(j+1)%4]), (0,255,0), 2)
            for j in range(4, 8):
                cv2.line(color_resized, tuple(uv[j]), tuple(uv[4+(j+1-4)%4]), (0,255,0), 2)
            for j in range(4):
                cv2.line(color_resized, tuple(uv[j]), tuple(uv[j+4]), (0,255,0), 1)
        color = 0
        colors = [
            (0, 0, 255),    # 红
            (0, 255, 0),    # 绿
            (255, 0, 0),    # 蓝
            (0, 255, 255),  # 黄
            (255, 0, 255),  # 紫
            (255, 255, 0),  # 青
        ]
        for suit_cloud in self.suit_pc:
            pts = suit_cloud  # Nx3

            fx = self.depth_camera.model_c.fx()
            fy = self.depth_camera.model_c.fy()
            cx = self.depth_camera.model_c.cx()
            cy = self.depth_camera.model_c.cy()

            # vectorized projection
            px = fx * pts[:, 0] / pts[:, 2] + cx
            py = fy * pts[:, 1] / pts[:, 2] + cy
            pts_int = np.vstack([px, py]).T.astype(np.int32)

            # fastest-point drawing (direct pixel assign)
            valid = (
                (pts_int[:,0] >= 0) & (pts_int[:,0] < color_resized.shape[1]) &
                (pts_int[:,1] >= 0) & (pts_int[:,1] < color_resized.shape[0])
            )

            pts_valid = pts_int[valid]
            color_resized[pts_valid[:,1], pts_valid[:,0]] = colors[color]

            color += 1
            cv2.putText(color_resized, f'Count: {self.counts[color-1]}', (10, 30 + 30 * (color-1)), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[color-1], 2)
        time_delay = time.time() - time_stamp
        cv2.putText(color_resized, f'Delay: {time_delay*1000:.1f} ms', (10, self.shape[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        time_cost = time.time() - time0
        cv2.putText(color_resized, f'Cost: {time_cost*1000:.1f} ms', (self.shape[0]-200, self.shape[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow("Color Image", color_resized)
        cv2.waitKey(1)

class PixelToCamera_t(Node):
    def __init__(self):
        super().__init__('pixel_to_camera_t')
        self.info_msg = None
        self.depth_camera = DepthCamera()
        self.depth_camera.loadCameraInfo()
        self.create_subscription(PointCloud2, '/camera/depth/points', self.pc_callback, 10)
        self.get_logger().info('Waiting for color frames...')
        self.color_img = None
        self.suit_pc = None
        self.counts = None
        self.shape =[848,480]
        self.box_check = None
        # 立方体中心在相机前方1.0m, 向下0.33m
        self.T_box_cam = np.array([0, 0.33, 1.0])
        # 立方体相对于相机的旋转：俯视0° + 向右0°
        self.R_box_cam = rot_y(00) @ rot_x(-0)

        cv2.namedWindow("Color Image")

    def pc_callback(self, msg):
        box_check_t = check_spearhead(msg, self.depth_camera, self.T_box_cam, self.R_box_cam)
        if box_check_t != self.box_check:
            self.box_check = box_check_t
            print("Box check changed:", self.box_check)
    
class PixelToCamera_tt(Node):
    def __init__(self):
        super().__init__('pixel_to_camera')
        self.info_msg = None
        self.depth_camera = DepthCamera()
        self.depth_camera.loadCameraInfo()
        self.create_subscription(Image, '/camera/depth/image_raw', self.depth_callback, qos)
        self.get_logger().info('Waiting for depth frames...')
        self.suit_pc = None
        self.pc = None
        self.depth_image = None
        self.counts = None
        self.shape =[848,480]
        self.img = np.zeros((self.shape[1], self.shape[0], 3), dtype=np.uint8)
        # 立方体中心在相机前方1.0m, 向下0.33m
        self.T_box_cam = np.array([0, 0.33, 1.0])
        # 立方体相对于相机的旋转：俯视0° + 向右0°
        self.R_box_cam = rot_y(00) @ rot_x(-0)
        self.timelist = [0] * 10
        self.timeListHead = 0
        self.final_img = None
        self.delay = 0.0
        self.depth_ready = threading.Event()
        self.pc_ready = threading.Event()
        threading.Thread(target=self.pc_worker, daemon=True).start()
        threading.Thread(target=self.dep2pc_worker, daemon=True).start()

        cv2.namedWindow("Color Image")

    def depth_callback(self, msg):
        self.depth_image = msg
        self.depth_ready.set()
        if self.final_img is not None:
            cv2.imshow("Color Image", self.final_img)
            cv2.waitKey(1)
        
    def dep2pc_worker(self):
        while True:
            if self.depth_camera.model_d.P is None:
                continue
            self.depth_ready.wait()  # 阻塞直到收到新数据
            depth = self.depth_camera.bridge.imgmsg_to_cv2(self.depth_image, desired_encoding='passthrough').astype(np.float32) / 1000.0
            self.depth_ready.clear()
            time_stamp = self.depth_image.header.stamp.sec + self.depth_image.header.stamp.nanosec * 1e-9
            self.depth_image = None
            fx = self.depth_camera.model_d.fx()
            fy = self.depth_camera.model_d.fy()
            cx = self.depth_camera.model_d.cx()
            cy = self.depth_camera.model_d.cy()
            mask = (depth > 0)
            v, u = np.nonzero(mask)
            d = depth[v, u]

            x = (u - cx) * d / fx
            y = (v - cy) * d / fy
            z = d

            self.pc = np.column_stack((x, y, z))
            self.pc_ready.set()
            self.delay = time.time() - time_stamp


    def pc_worker(self):
        while True:
            self.pc_ready.wait()
            msg = self.pc
            self.pc_ready.clear()
            time0 = time.time()
            #depth_pc = pc2.read_points_numpy(msg, field_names=("x","y","z"))
            depth_pc = msg
            pc_count = depth_pc.shape[0]
            # 用外参偏移点云
            
            self.counts, self.suit_pc = filter_rotated_subboxes(depth_pc, self.T_box_cam, self.R_box_cam, self.depth_camera)

            if self.suit_pc is None:
                continue
            cv2_color_img = self.img
            cv2_img_bgr = cv2.cvtColor(cv2_color_img, cv2.COLOR_RGB2BGR)
            color_resized = cv2.resize(cv2_img_bgr, (self.shape[0], self.shape[1]), interpolation=cv2.INTER_LINEAR)
            self.depth_camera.model_c = resize_intrinsics(
                self.depth_camera.model_c,
                self.shape[0],
                self.shape[1]
            )
            fps, self.timelist, self.timeListHead = get_FPS(self.timelist, self.timeListHead)
            cv2.putText(color_resized, f'FPS: {fps:.2f}', (self.shape[0] - 100, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(color_resized, f'Points: {pc_count}', (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            uv_boxes = project_subboxes(self.depth_camera.model_c, self.T_box_cam, self.R_box_cam)
            # 转为整数
            for i, uv in enumerate(uv_boxes):
                uv = uv.astype(int)
                # 画边（角点顺序按四边形连接）
                for j in range(4):
                    cv2.line(color_resized, tuple(uv[j]), tuple(uv[(j+1)%4]), (0,255,0), 2)
                for j in range(4, 8):
                    cv2.line(color_resized, tuple(uv[j]), tuple(uv[4+(j+1-4)%4]), (0,255,0), 2)
                for j in range(4):
                    cv2.line(color_resized, tuple(uv[j]), tuple(uv[j+4]), (0,255,0), 1)
            color = 0
            colors = [
                (0, 0, 255),    # 红
                (0, 255, 0),    # 绿
                (255, 0, 0),    # 蓝
                (0, 255, 255),  # 黄
                (255, 0, 255),  # 紫
                (255, 255, 0),  # 青
            ]
            for suit_cloud in self.suit_pc:
                pts = suit_cloud  # Nx3

                fx = self.depth_camera.model_c.fx()
                fy = self.depth_camera.model_c.fy()
                cx = self.depth_camera.model_c.cx()
                cy = self.depth_camera.model_c.cy()

                # vectorized projection
                px = fx * pts[:, 0] / pts[:, 2] + cx
                py = fy * pts[:, 1] / pts[:, 2] + cy
                pts_int = np.vstack([px, py]).T.astype(np.int32)

                # fastest-point drawing (direct pixel assign)
                valid = (
                    (pts_int[:,0] >= 0) & (pts_int[:,0] < color_resized.shape[1]) &
                    (pts_int[:,1] >= 0) & (pts_int[:,1] < color_resized.shape[0])
                )

                pts_valid = pts_int[valid]
                color_resized[pts_valid[:,1], pts_valid[:,0]] = colors[color]

                color += 1
                cv2.putText(color_resized, f'Count: {self.counts[color-1]}', (10, 30 + 30 * (color-1)), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[color-1], 2)
            time_delay = self.delay
            cv2.putText(color_resized, f'Delay: {time_delay*1000:.1f} ms', (10, self.shape[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            time_cost = time.time() - time0
            cv2.putText(color_resized, f'Cost: {time_cost*1000:.1f} ms', (self.shape[0]-200, self.shape[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            self.final_img = color_resized
            

def main():
    rclpy.init()
    node = PixelToCamera()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

def main1():
    rclpy.init()
    node = PixelToCamera_t()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

def main2():
    rclpy.init()
    node = PixelToCamera_tt()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main2()