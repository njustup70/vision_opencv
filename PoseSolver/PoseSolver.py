import cv2
import numpy as np

class PoseSolver:
    def __init__(self, camera_matrix, dist_coeffs:np.ndarray, marker_length:np.ndarray, marker_width=None,print_result=False):
        """
        :param camera_matrix: 相机内参矩阵 (3x3)
        :param dist_coeffs: 畸变系数 (1x5)
        :param marker_length: 矩形标记的实际物理长度(X轴方向,单位:米)
        :param marker_width: 矩形标记的实际物理宽度(Y轴方向,单位:米。默认为None,表示使用正方形)
        """
        if type(camera_matrix) is not np.ndarray or type(dist_coeffs) is not np.ndarray:
            raise TypeError("camera_matrix和dist_coeffs必须是numpy数组")
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.marker_length = marker_length
        self.marker_width = marker_width if marker_width is not None else marker_length
        self.print_result=print_result
        # 定义矩形的3D角点（以中心为原点，Z=0平面）
        self.obj_points = np.array([
            [-self.marker_length/2,  self.marker_width/2, 0],  # 左上角
            [ self.marker_length/2,  self.marker_width/2, 0],  # 右上角
            [ self.marker_length/2, -self.marker_width/2, 0],  # 右下角
            [-self.marker_length/2, -self.marker_width/2, 0]   # 左下角
        ], dtype=np.float32)

    def solve_pose(self, corners):
        """
        输入矩形标记的角点,解算位姿,要求角点顺序与obj_points定义一致,
        :param corners: 矩形标记的4个角点坐标(格式:np.array,shape=(4,2))
        :return: rvec, tvec (旋转向量, 平移向量)
        """
        # 确保输入角点顺序与obj_points一致（左上、右上、右下、左下）
        assert corners.shape == (4, 2), "角点格式应为(4,2)的np数组"
        
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
        :param axis_length: 坐标轴长度（单位：米）
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

        cv2.line(image, origin, x_end, (0, 0, 255), 2)  # X轴（红色）
        cv2.line(image, origin, y_end, (0, 255, 0), 2)  # Y轴（绿色）
        cv2.line(image, origin, z_end, (255, 0, 0), 2)  # Z轴（蓝色）
    def update(self,image:np.ndarray,content:dict):
        """_summary_

        Args:
            image (np.ndarray): _description_
            content (dict): 需要有corners字段, corners字段中需要有corners
        brief:
            会读取corners字段中的第一个框的corners,然后解算位姿
        """
        corners=[]
        #创建pnp字典
        content["pnp"]={}
        if "corners" in content:
            if len(content["corners"])==0:
                return
            #读取第一个框
            corners=content["corners"][0]["corners"] 
            # corners.append(content["corners"][0]["cx"],content["corners"][0]["cy"])

        else :
            return
        rvec,tvec=self.solve_pose(corners)
        self.draw_axis(image,rvec,tvec)
        content["pnp"]["rvec"]=rvec
        content["pnp"]["tvec"]=tvec
        if self.print_result:
            rotation_matrix, _ = cv2.Rodrigues(rvec)
            sy = np.sqrt(rotation_matrix[0,0] ** 2 + rotation_matrix[1,0] ** 2)
            pitch = np.arctan2(rotation_matrix[2,1], rotation_matrix[2,2])  # X轴旋转角
            yaw = np.arctan2(-rotation_matrix[2,0], sy)                     # Y轴旋转角
            roll = np.arctan2(rotation_matrix[1,0], rotation_matrix[0,0])   # Z轴旋转角
            print(f"Pitch: {np.degrees(pitch):.1f}°, Yaw: {np.degrees(yaw):.1f}°, Roll: {np.degrees(roll):.1f}°")