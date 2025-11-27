import yaml
import numpy as np
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from cv_bridge import CvBridge

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

# 立方体相对于相机的旋转：俯视5° + 向右90°
R_box_cam = rot_y(00) @ rot_x(-0)

# 立方体中心在相机前方1m
T_box_cam = np.array([0, 0.28, 1.11])

# 子区域局部中心（立方体坐标系）
x_centers = np.array([-0.5, -0.3, -0.1, 0.1, 0.3, 0.5])  # m
T_local_centers = np.stack([x_centers,
                            np.zeros_like(x_centers),
                            np.zeros_like(x_centers)], axis=1)

# 尺寸半长
hx, hy, hz = 0.10, 0.15, 0.15  # 200x300x300 mm

def filter_rotated_subboxes(pts_cam):
    '''
    过滤出落在旋转立方体六个子区域内的点
    :param pts_cam:   Nx3的3D点数组 :type:`np.ndarray`
    :return: 6个布尔数组，表示每个点是否落在对应子区域内 :type:`List[np.ndarray]`
    '''

    results = []

    for i in range(6):
        # 相机坐标下的子区域中心
        T_i_cam = T_box_cam + (R_box_cam @ T_local_centers[i])

        # 相机→子区域局部
        pts_i = (pts_cam - T_i_cam) @ R_box_cam.T

        mask = (
            (np.abs(pts_i[:,0]) <= hx) &
            (np.abs(pts_i[:,1]) <= hy) &
            (np.abs(pts_i[:,2]) <= hz)
        )

        results.append(mask)

    return results  # 返回 6 个 mask

# ---------------------------
# 投影函数
# ---------------------------
def project_points(Pc, K):
    """Pc: Nx3 相机坐标点，K: 3x3 内参"""
    x = Pc[:,0]
    y = Pc[:,1]
    z = Pc[:,2]
    u = K[0,0]*x/z + K[0,2]
    v = K[1,1]*y/z + K[1,2]
    return np.stack([u,v], axis=1)

# ---------------------------
# 生成每个子框在像素上的8点投影
# ---------------------------
def project_subboxes(model):
    """
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

def get_offset_target(model_d, d2c_r, d2c_t, T_box_cam):

    T_box_cam = np.asarray(T_box_cam).reshape(3)
    ud, vd = model_d.project3dToPixel(T_box_cam)
    T_color = d2c_r @ T_box_cam + d2c_t
    uc, vc = model_d.project3dToPixel(T_color)
    dx = uc - ud
    dy = vc - vd

    return dx, dy

def shift_depth_by_offset(depth_img, dx, dy):
    H, W = depth_img.shape
    depth_img = depth_img.astype(np.float64)
    depth_img[depth_img <= 0] = np.nan
    shifted = np.full_like(depth_img, np.nan)

    if dx >= 0:
        xs_src = slice(0, W - dx)
        xs_dst = slice(dx, W)
    else:
        xs_src = slice(-dx, W)
        xs_dst = slice(0, W + dx)
    if dy >= 0:
        ys_src = slice(0, H - dy)
        ys_dst = slice(dy, H)
    else:
        ys_src = slice(-dy, H)
        ys_dst = slice(0, H + dy)
    shifted[ys_dst, xs_dst] = depth_img[ys_src, xs_src]

    return shifted

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
        self.dx, self.dy = get_offset_target(
            self.depth_camera.model_d,
            self.depth_camera.d2c_r,
            self.depth_camera.d2c_t,
            T_box_cam
        )
        self.create_subscription(Image, '/camera/depth/image_raw', self.depth_callback, 10)
        self.create_subscription(Image, '/camera/color/image_raw', self.color_callback, 10)
        self.create_subscription(PointCloud2, '/camera/depth/points', self.pc_callback, 10)
        self.get_logger().info('Waiting for camera_info and depth frames...')
        self.depth_img = None
        self.color_img = None
        self.point_cloud = None

        cv2.namedWindow("Color Image")

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
        # 用外参偏移点云
        pc_color = (self.depth_camera.d2c_r @ self.point_cloud.T).T + self.depth_camera.d2c_t
        self.point_cloud = pc_color

    def depth_callback(self, msg):
        if self.color_img is None:
            return
        if self.point_cloud is None:
            return
        #self.timetest()
        self.depth_img = msg
        cv2_color_img = self.depth_camera.bridge.imgmsg_to_cv2(self.color_img, desired_encoding='passthrough')
        cv2_depth_img = self.depth_camera.bridge.imgmsg_to_cv2(self.depth_img, desired_encoding='passthrough').astype(np.uint16)
        color_resized = cv2.resize(cv2_color_img, (cv2_depth_img.shape[1], cv2_depth_img.shape[0]), interpolation=cv2.INTER_LINEAR)
        self.depth_camera.model_c = resize_intrinsics(
            self.depth_camera.model_c,
            cv2_depth_img.shape[1],
            cv2_depth_img.shape[0]
        )
        dx = int(round(self.dx))
        dy = int(round(self.dy))
        depth_trans = shift_depth_by_offset(cv2_depth_img, dx, dy)
        # 深度图叠加到彩色图，颜色表示深度
        depth_colored = cv2.applyColorMap(cv2.convertScaleAbs(depth_trans, alpha=0.03), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(color_resized, 0.6, depth_colored, 0.4, 0)
        color_resized = overlay
        uv_boxes = project_subboxes(self.depth_camera.model_c)
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
        suit_clouds = filter_rotated_subboxes(self.point_cloud)
        color = 0
        colors = [
            (0, 0, 255),    # 红
            (0, 255, 0),    # 绿
            (255, 0, 0),    # 蓝
            (0, 255, 255),  # 黄
            (255, 0, 255),  # 紫
            (255, 255, 0),  # 青
        ]
        for suit_cloud in suit_clouds:
            suit_cloud = self.point_cloud[suit_cloud]
            for pt in suit_cloud:
                px, py = self.depth_camera.model_c.project3dToPixel(pt)
                cv2.circle(color_resized, (int(px), int(py)), 2, colors[color], -1)
            color += 1
        cv2.imshow("Color Image", color_resized)
        cv2.waitKey(1)

    


def main():
    rclpy.init()
    node = PixelToCamera()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()