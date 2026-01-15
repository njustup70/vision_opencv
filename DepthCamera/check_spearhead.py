# 矛头检测模块
import numpy as np
from sensor_msgs.msg import Image
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, qos_profile_sensor_data

qos = QoSProfile(
    depth=1,  # 只保留最新一帧
    reliability=QoSReliabilityPolicy.BEST_EFFORT,  # 尽量保证丢帧而不阻塞
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

    :param pc: 点云数据 :type:`np.ndarray` (Nx3)
    :param model: 相机模型 :type:`DepthCamera`
    :param T_box_cam: 立方体中心在相机坐标系的位置 :type:`np.ndarray` (右， 下， 前)
    :param R_box_cam: 立方体相对于相机的旋转矩阵 :type:`np.ndarray` (俯角， 右转角) rot_y(Theta1) @ rot_x(-Theta2)
    :return: 每个子区域是否满足点数要求的布尔列表 :type:`List[bool]`
    '''
    #depth_pc = pc2.read_points_numpy(pc, field_names=("x","y","z"))
    depth_pc = pc
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
    from DepthCamera import DepthCamera, DepthCamNode
    import sensor_msgs_py.point_cloud2 as pc2
    import threading

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

    class CheckSpearhead(DepthCamNode):
        def __init__(self):
            super().__init__('spearhead_cheak_node')
            self.create_subscription(Image, '/camera/color/image_raw', self.color_callback, qos_profile_sensor_data)
            self.create_subscription(Image, '/camera/depth/image_raw', self.depth_callback, qos)
            self.get_logger().info('Waiting for color frames...')
            self.suit_pc = None
            self.counts = None
            self.shape =[848,480]
            # 立方体中心在相机前方1.0m, 向下0.33m
            self.T_box_cam = np.array([0, 0.33, 1.0])
            # 立方体相对于相机的旋转：俯视0° + 向右0°
            self.R_box_cam = rot_y(00) @ rot_x(-0)
            self.timelist = [0] * 10
            self.timeListHead = 0

            cv2.namedWindow("Color Image", cv2.WINDOW_NORMAL)

        def depth_callback(self, msg):

            self.depth_image = msg
            depth = self.depth_camera.bridge.imgmsg_to_cv2(self.depth_image, desired_encoding='passthrough').astype(np.float32) / 1000.0
            if self.shape != (depth.shape[1], depth.shape[0]):
                self.shape = depth.shape[1], depth.shape[0]
            self.pc = self.depth_camera.depth2points(depth)
            depth_pc = self.pc
            
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
            if self.shape[0] != cv2_color_img.shape[0] or self.shape[1] != cv2_color_img.shape[1]:
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

    class CheckSpearhead_t(DepthCamNode):
        def __init__(self):
            super().__init__('spearhead_cheak_node')
            self.create_subscription(Image, '/camera/depth/image_raw', self.depth_callback, qos)
            self.get_logger().info('Waiting for color frames...')
            self.suit_pc = None
            self.counts = None
            self.box_check = None
            # 立方体中心在相机前方1.0m, 向下0.33m
            self.T_box_cam = np.array([0, 0.33, 1.0])
            # 立方体相对于相机的旋转：俯视0° + 向右0°
            self.R_box_cam = rot_y(00) @ rot_x(-0)

        def depth_callback(self, msg):
            self.depth_image = msg
            depth = self.depth_camera.bridge.imgmsg_to_cv2(self.depth_image, desired_encoding='passthrough').astype(np.float32) / 1000.0
            self.pc = self.depth_camera.depth2points(depth)
            box_check_t = check_spearhead(self.pc, self.depth_camera, self.T_box_cam, self.R_box_cam)
            if box_check_t != self.box_check:
                self.box_check = box_check_t
                print("Box check changed:", self.box_check)
            elif self.box_check is None:
                self.box_check = box_check_t
                print("Initial box check:", self.box_check)
        
    class CheckSpearhead_tt(DepthCamNode):
        def __init__(self):
            super().__init__('spearhead_cheak_node')
            self.create_subscription(Image, '/camera/depth/image_raw', self.depth_callback, qos)
            self.get_logger().info('Waiting for depth frames...')
            self.suit_pc = None
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

            cv2.namedWindow("Color Image, cv2.WINDOW_NORMAL")

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
                self.pc = self.depth_camera.depth2points(depth)
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
        node = CheckSpearhead()
        rclpy.spin(node)
        node.destroy_node()
        rclpy.shutdown()

    def main1():
        rclpy.init()
        node = CheckSpearhead_t()
        rclpy.spin(node)
        node.destroy_node()
        rclpy.shutdown()

    def main2():
        rclpy.init()
        node = CheckSpearhead_tt()
        rclpy.spin(node)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()