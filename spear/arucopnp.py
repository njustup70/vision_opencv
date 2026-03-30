import cv2
import numpy as np

class HighPrecisionPoseEstimator:
    def __init__(self,K=None):
        # OpenCV 4.10.0 是 Aruco 模块 API 彻底标准化的重要节点
        cv_version = cv2.__version__.split('.')
        cv_major = int(cv_version[0])
        cv_minor = int(cv_version[1])
        
        assert cv_major > 4 or (cv_major == 4 and cv_minor >= 10), \
            f"检测到 OpenCV 版本为 {cv2.__version__}，本解算器要求版本 >= 4.10.0 以支持新的 ArucoDetector 接口。"
        # 1. 配置字典和板子 (请根据你的物理板子实际尺寸修改)
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        # 参数: (列数, 行数, 棋盘格方块边长, 标记边长, 字典)
        # 单位建议用米(m)，例如 0.04 表示 4cm
        self.board = cv2.aruco.CharucoBoard((5, 5), 0.01, 0.007, self.dictionary)
        
        # 2. 配置高精度检测参数
        self.detector_params = cv2.aruco.DetectorParameters()
        # 核心：使用 AprilTag 算法进行亚像素精修，这是目前最稳的方法
        self.detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
        # 搜索窗口：如果你的 Marker 在图里很大，用 10-20；如果很小，建议保持默认 5
        self.detector_params.cornerRefinementWinSize = 4 
        self.detector_params.cornerRefinementMaxIterations = 50
        self.detector_params.cornerRefinementMinAccuracy = 0.02
        # 3. 初始化检测器
        self.charuco_detector = cv2.aruco.CharucoDetector(
            self.board, 
            detectorParams=self.detector_params,
            # charucoParams=charuco_params,  # 确保这个参数被传入
        )

        if K is not None:
            self.K = K
        else:
            # 4. 相机内参 (请替换为你标定后的真实数据)
            self.K = np.array([[800.0, 0, 320.0], 
                            [0, 800.0, 240.0], 
                            [0, 0, 1.0]], dtype=np.float32)
        self.D = np.zeros((5, 1), dtype=np.float32) # 畸变系数

        # 5. 帧间平滑记录（用于防止跳变）
        self.last_rvec = None
        self.last_tvec = None

    def on_image(self, frame):
        """数据回调函数"""
        if frame is None:
            return None, None

        # 执行检测：得到精修后的棋盘格角点
        # charuco_corners: 2D 坐标, charuco_ids: 角点 ID
        c_corners, c_ids, m_corners, m_ids = self.charuco_detector.detectBoard(frame)
        print('{m_ids} markers and {c_ids} charuco corners detected.'.format(m_ids=0 if m_ids is None else len(m_ids), c_ids=0 if c_ids is None else len(c_ids)))
        best_rvec, best_tvec = None, None
        # 必须至少有 4 个点才能进行 PnP
        if c_ids is not None and len(c_ids) >= 4:
            # 获取对应的 3D 物理坐标
            obj_points = self.board.getChessboardCorners()[c_ids.ravel()]
            img_points = c_corners

            # 使用 solvePnPGeneric 获取所有可能的解（处理翻转歧义性）
            # 对于平面物体，SOLVEPNP_IPPE 是目前数学上最严谨的解法
            retval, rvecs, tvecs, errors = cv2.solvePnPGeneric(
                obj_points, img_points, self.K, self.D, 
                flags=cv2.SOLVEPNP_IPPE
            )

            # 选择最佳解逻辑
            if retval > 0:
                best_rvec, best_tvec = self._select_best_pose(rvecs, tvecs)

        # 绘制结果预览：有什么数据就画什么数据
        self._draw_result(frame, c_corners, c_ids, m_corners, m_ids, best_rvec, best_tvec)
        return best_rvec, best_tvec

    def _select_best_pose(self, rvecs, tvecs):
        """多解选择逻辑：优先选误差小的，结合上一帧平滑"""
        if len(rvecs) == 1:
            return rvecs[0], tvecs[0]

        # 如果有多个解（通常是 2 个），且有历史帧
        if self.last_rvec is not None:
            # 计算当前两个解与上一帧的欧氏距离，选变动最小的那个
            dist0 = np.linalg.norm(rvecs[0] - self.last_rvec)
            dist1 = np.linalg.norm(rvecs[1] - self.last_rvec)
            idx = 0 if dist0 < dist1 else 1
        else:
            # 第一帧直接选误差最小的（rvecs[0] 通常是最小重投影误差）
            idx = 0
        
        self.last_rvec = rvecs[idx]
        self.last_tvec = tvecs[idx]
        return rvecs[idx], tvecs[idx]

    def _draw_result(self, frame, corners, ids, m_corners, m_ids, rvec, tvec):
        # 检查是否有有效的位姿
        # marker 有效时画出来
        if m_corners is not None and m_ids is not None and len(m_ids) > 0:
            cv2.aruco.drawDetectedMarkers(frame, m_corners, None)
        # 角点有效时再画，避免 None 触发 OpenCV 断言
        if corners is not None and ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedCornersCharuco(frame, corners, None)

        # 有位姿才画 3D 坐标轴
        if rvec is not None and tvec is not None:
            cv2.drawFrameAxes(frame, self.K, self.D, rvec, tvec, 0.1)

        # 有什么数据标什么数据
        lines = []
        if m_ids is not None:
            lines.append(f"markers: {len(m_ids)}")
        if ids is not None:
            lines.append(f"charuco: {len(ids)}")
        if rvec is not None:
            r = np.asarray(rvec).reshape(-1)
            if r.size >= 3:
                lines.append(f"rvec: [{r[0]:.3f}, {r[1]:.3f}, {r[2]:.3f}]")
        if tvec is not None:
            t = np.asarray(tvec).reshape(-1)
            if t.size >= 3:
                lines.append(f"tvec(m): [{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}]")
        return frame

# --- 使用示例 ---
def main():
    cap = cv2.VideoCapture(0)
    estimator = HighPrecisionPoseEstimator()
    while True:
        _,img= cap.read()
        if _ is False:
            continue
        rvec, tvec = estimator.on_image(img)
        # on_image 内部已完成绘制，这里直接显示
        cv2.imshow("Pose Estimation", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()   