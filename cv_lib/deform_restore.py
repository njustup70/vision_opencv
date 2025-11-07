import cv2
import numpy as np

camera_matrix = np.array([
    [714.9199755874305, 0.0, 354.4380105887537], 
    [0.0, 715.2231562188343, 715.2231562188343],   
    [0.0, 0.0, 1.0]          
], dtype=np.float32)

dist_coeffs = np.array([
    [-0.15781026869102965,  1.583778819087774, -0.011256244004513636, 0.011363172123126099, -4.632718536609601]
], dtype=np.float32)

def get3dPoints(center, shape):
    '''
    计算平面四个顶点的3D坐标
    center: 平面中心点的3D相对坐标 (x, y, z)
    shape: 平面的宽高和法向量 (width, height, (nx, ny, nz))
    return: 4个顶点的3D坐标，按顺时针顺序排列
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

    # 随便找一个与 n 不平行的向量，求叉积得到平面局部坐标轴
    up = np.array([0, 1, 0], dtype=np.float32)
    if abs(np.dot(up, n)) > 0.9:
        up = np.array([1, 0, 0], dtype=np.float32)

    x_axis = np.cross(up, n)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(n, x_axis)
    R_plane = np.stack([x_axis, y_axis, n], axis=1)  # 3x3
    points_3d = (R_plane @ plane_local.T).T + np.array(center, dtype=np.float32)
    return points_3d

def Let3DToPlane(points_3d, camera_matrix, dist_coeffs, rvec=None, tvec=None):
    '''
    将3D点投影到图像平面上
    points_3d: Nx3的3D点数组,顺时针
    camera_matrix: 相机内参矩阵
    dist_coeffs: 相机畸变系数
    rvec: 旋转向量 (可选)
    tvec: 平移向量 (可选)
    return: 投影后的2D点数组 Nx2
    '''
    if rvec is None:
        rvec = np.zeros((3, 1), dtype=np.float32)
    if tvec is None:
        tvec = np.zeros((3, 1), dtype=np.float32)
    points_2d, _ = cv2.projectPoints(points_3d, rvec, tvec, camera_matrix, dist_coeffs)
    return points_2d.reshape(-1, 2)

def ROIRestore(img, points_2d, image_shape = [500,500]):
    '''
    根据投影的2D点计算图像的边界框,展开到原图像
    points_2d: Nx2的2D点数组
    image_shape: 需要图像的形状 (height, width)
    return: 还原展开后图像
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

def DeformRestore(img, point, shape, camera_matrix = camera_matrix, dist_coeffs = dist_coeffs, rvec=None, tvec=None, image_shape = [500,500]):
    ''' 
    根据3D点和相机参数还原图像
    img: 输入图像
    point: 平面中心点的3D相对坐标 (x, y, z)
    shape: 平面的宽高和法向量 (width, height, (nx, ny, nz))
    camera_matrix: 相机内参矩阵
    dist_coeffs: 相机畸变系数
    rvec: 旋转向量 (外参可选)
    tvec: 平移向量 (外参可选)
    image_shape: 需要图像的形状 (height, width)
    return: 还原展开后图像
    '''
    points_3d = get3dPoints(point, shape)
    points_2d = Let3DToPlane(points_3d, camera_matrix, dist_coeffs, rvec=rvec, tvec=tvec)
    return ROIRestore(img, points_2d, image_shape=image_shape)

if __name__ == "__main__":
    # 测试代码
    img = cv2.imread("test_deform.jpg")
    center = (0, 0, 0.5)  # 平面中心点的3D坐标
    shape = (0.3, 0.2, (0, 0, 1))  # 平面的宽高和法向量
    restored_img = DeformRestore(img, center, shape)
    cv2.imshow("Restored Image", restored_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()