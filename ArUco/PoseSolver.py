import cv2
import numpy as np

class PoseSolver:
    def __init__(self, camera_matrix, dist_coeffs, marker_length):
        """
        :param camera_matrix: 相机内参矩阵 (3x3)
        :param dist_coeffs: 畸变系数 (1x5)
        :param marker_length: ArUco码实际物理边长 (单位: 米)
        """
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.marker_length = marker_length
        
        # 定义ArUco码的3D角点 (以中心为原点，Z=0平面)
        self.obj_points = np.array([
            [-marker_length/2, marker_length/2, 0],
            [marker_length/2, marker_length/2, 0],
            [marker_length/2, -marker_length/2, 0],
            [-marker_length/2, -marker_length/2, 0]
        ], dtype=np.float32)

    def solve_pose(self, corners):
        """
        输入ArUco码的角点,解算位姿
        :param corners: ArUco码的4个角点坐标 (格式: np.array, shape=(4,2))
        :return: rvec, tvec (旋转向量, 平移向量)
        """
        success, rvec, tvec = cv2.solvePnP(
            self.obj_points,
            corners.astype(np.float32),
            self.camera_matrix,
            self.dist_coeffs
        )
        if not success:
            raise ValueError("PnP解算失败")
        return rvec, tvec

    def draw_axis(self, image, rvec, tvec, axis_length=0.05):
        """
        在图像上绘制3D坐标轴
        :param axis_length: 坐标轴长度 (单位: 米)
        """
        axis_points = np.array([
            [0, 0, 0],
            [axis_length, 0, 0],
            [0, axis_length, 0],
            [0, 0, axis_length]
        ], dtype=np.float32)

        img_points, _ = cv2.projectPoints(
            axis_points, rvec, tvec,
            self.camera_matrix, self.dist_coeffs
        )

        origin = tuple(map(int, img_points[0].ravel()))
        x_end = tuple(map(int, img_points[1].ravel()))
        y_end = tuple(map(int, img_points[2].ravel()))
        z_end = tuple(map(int, img_points[3].ravel()))

        cv2.line(image, origin, x_end, (0, 0, 255), 2)  # X轴 (红色)
        cv2.line(image, origin, y_end, (0, 255, 0), 2)  # Y轴 (绿色)
        cv2.line(image, origin, z_end, (255, 0, 0), 2)  # Z轴 (蓝色)