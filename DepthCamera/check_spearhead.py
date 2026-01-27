import numpy as np

# from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, qos_profile_sensor_data

# qos = QoSProfile(
#     depth=1,  # 只保留最新一帧
#     reliability=QoSReliabilityPolicy.BEST_EFFORT,  # 尽量保证丢帧而不阻塞
#     history=QoSHistoryPolicy.KEEP_LAST          # 只保存最后 depth 帧
# )

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

def rot_z(deg):
    rad = np.deg2rad(deg)
    return np.array([
        [np.cos(rad),-np.sin(rad),0],
        [np.sin(rad), np.cos(rad),0],
        [0,0,1]
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
