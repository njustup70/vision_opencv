import cv2
import numpy as np

# 这个内参矩阵和畸变系数是深度相机彩色图像的
camera_matrix = np.array([
    [611.544189453125, 0.0, 636.7078857421875, 0.0, 611.3428955078125, 397.55560302734375, 0.0, 0.0, 1.0]      
], dtype=np.float32)

dist_coeffs = np.array([
    [-0.03157561272382736, 0.03607562929391861, 0.0001983262482099235, -0.000041368522943230346, -0.012517155148088932, 0, 0, 0]
], dtype=np.float32)

def get3dPoints(center, shape, up_direct=None):
    '''
    计算平面四个顶点相机坐标系下的3D坐标
    
    :param center:          平面中心点的3D相对坐标 (x, y, z)
    :param shape:           平面的宽高和法向量 (width, height, (nx, ny, nz))
    :param up_direct:       平面纵向向量，即y轴正方形 (可选)
    :return:                4个顶点的3D坐标，按顺时针顺序排列 :type:`np.ndarray` (4x3)
    '''
    w, h, n = shape
    # 平面自身局部坐标（以中心为原点）
    half_w = w / 2.0
    half_h = h / 2.0
    plane_local = np.array([
        [-half_w, -half_h, 0],
        [ half_w, -half_h, 0],
        [ half_w,  half_h, 0],
        [-half_w,  half_h, 0]
    ], dtype=np.float32)
    # 向量 n 归一化
    nx, ny, nz = n
    n = np.array([nx, ny, nz], dtype=np.float32)
    n /= np.linalg.norm(n)

    if up_direct is None:
        # 随便找一个与 n 不平行的向量，求叉积得到平面局部坐标轴
        up = np.array([0, 1, 0], dtype=np.float32)
        if abs(np.dot(up, n)) > 0.9:
            up = np.array([1, 0, 0], dtype=np.float32)
    else:
        up = -up_direct  # 取纵向向量作为up

    x_axis = np.cross(up, n)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(n, x_axis)
    R_plane = np.stack([x_axis, y_axis, n], axis=1)  # 3x3
    points_3d = (R_plane @ plane_local.T).T + np.array(center, dtype=np.float32)
    return points_3d

def trans3DToPlane(points_3d, camera_matrix = camera_matrix, dist_coeffs = dist_coeffs, rvec=None, tvec=None):
    '''
    将3D点投影到图像平面上

    :param points_3d:       Nx3的3D点数组,顺时针 排列 :type:`np.ndarray` (4x3)
    :param camera_matrix:   相机内参矩阵
    :param dist_coeffs:     相机畸变系数
    :param rvec:            旋转向量 (可选)
    :param tvec:            平移向量 (可选)
    :return: 投影后的2D点数组 Nx2
    '''
    camera_matrix = np.array(camera_matrix, dtype=np.float64).reshape(3, 3)
    if rvec is None:
        rvec = np.zeros((3, 1), dtype=np.float32)
    if tvec is None:
        tvec = np.zeros((3, 1), dtype=np.float32)
    points_2d, _ = cv2.projectPoints(points_3d, rvec, tvec, camera_matrix, dist_coeffs)
    return points_2d.reshape(-1, 2)

def ROIRestore(img, points_2d, image_shape = [500,500]):
    '''
    根据投影的2D点计算图像的边界框,展开到原图像

    :param img:         输入图像 :cpp:type:`sensor_msgs::Image`
    :param points_2d:   Nx2的2D点数组 :type:`np.ndarray`
    :param image_shape: 需要图像的形状 (height, width)
    :return: 还原展开后图像 :type:`np.ndarray`
    '''
    w_out = image_shape[1]
    h_out = image_shape[0]
    pts_dst = np.array([
        [0, 0],
        [w_out-1, 0],
        [w_out-1, h_out-1],
        [0, h_out-1]
    ], dtype=np.float32)
    Hmat, _ = cv2.findHomography(points_2d, pts_dst)
    warped = cv2.warpPerspective(img, Hmat, (w_out, h_out))
    return warped

def deformRestore(img, point, shape, up_direct=None, camera_matrix = camera_matrix, dist_coeffs = dist_coeffs, rvec=None, tvec=None, image_shape = [500,500]):
    ''' 
    根据3D点和相机参数还原图像

    :param img:             输入图像 :cpp:type:`sensor_msgs::Image`
    :param point:           平面中心点的3D相对坐标 (x, y, z)
    :param shape:           平面的宽高和法向量 (width, height, (nx, ny, nz))
    :param up_direct:       平面纵向向量，即y轴正方形 (可选)
    :param camera_matrix:   相机内参矩阵
    :param dist_coeffs:     相机畸变系数
    :param rvec:            旋转向量 (外参可选)
    :param tvec:            平移向量 (外参可选)
    :param image_shape:     需要图像的形状 (height, width)
    :return: 还原展开后图像
    '''
    camera_matrix = np.array(camera_matrix, dtype=np.float64).reshape(3, 3)
    points_3d = get3dPoints(point, shape, up_direct=up_direct)
    points_2d = trans3DToPlane(points_3d, camera_matrix, dist_coeffs, rvec=rvec, tvec=tvec)
    return ROIRestore(img, points_2d, image_shape=image_shape), points_2d

if __name__ == "__main__":
    # 测试代码
    img = cv2.imread("cv_lib/color_image/retrieved_image.jpg")
    img = cv2.resize(img, (1280, 800))
    center = (0.457, 0.017, 0.6)  # 平面中心点的3D坐标
    shape = (0.35, 0.35, (0, 0, 1))  # 平面的宽高和法向量
    restored_img = deformRestore(img, center, shape, up_direct=None)
    cv2.imshow("Restored Image", restored_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()