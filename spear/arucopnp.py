import cv2
import numpy as np

class HighPrecisionPoseEstimator:
    def __init__(self,K=None,D=None):
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
        self.board = cv2.aruco.CharucoBoard((3, 3), 0.025, 0.018, self.dictionary)
        
        # 2. 配置高精度检测参数
        self.detector_params = cv2.aruco.DetectorParameters()
        # 核心：使用 AprilTag 算法进行亚像素精修，这是目前最稳的方法
        self.detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        # 搜索窗口：如果你的 Marker 在图里很大，用 10-20；如果很小，建议保持默认 5
        self.detector_params.cornerRefinementWinSize = 4 
        self.detector_params.cornerRefinementMaxIterations = 20
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
        if D is not None:
            self.D = D
        else:
            self.D = np.zeros((5, 1), dtype=np.float32) # 畸变系数

        # 5. 帧间平滑记录（用于防止跳变）
        self.last_rvec = None
        self.last_tvec = None

    def on_image(self, frame):
        """数据回调函数"""
        if frame is None: return None, None

        # 1. 执行检测
        c_corners, c_ids, m_corners, m_ids = self.charuco_detector.detectBoard(frame)
        
        obj_points, img_points = None, None
        best_rvec, best_tvec = None, None

        # 2. 策略 A：优先使用 Charuco 角点（高精度）
        if c_ids is not None and len(c_ids) >= 4:
            # 这里的 getChessboardCorners 会根据 c_ids 自动匹配 3D 坐标
            obj_points,img_points = self.board.matchImagePoints(c_corners, c_ids)
            #  = c_corners
        # 3. 策略 B：退而求其次，使用 ArUco 标记角点（防止 c_ids 丢失）
        elif m_ids is not None and len(m_ids) >= 1:
            obj_list = []
            img_list = []

            # 获取该 Board 中所有 Marker 的 ID 列表及其对应的 3D 坐标
            board_ids = self.board.getIds()
            # getObjPoints 返回的是所有 Marker 的 4 个角坐标，形状为 (N, 4, 3)
            board_obj_points = self.board.getObjPoints()

            for i, m_id in enumerate(m_ids.flatten()):
                # 寻找当前检测到的 ID 在 Board 定义中的索引
                idx = np.where(board_ids == m_id)[0]
                if len(idx) > 0:
                    # 提取该索引对应的 4 个 3D 点
                    obj_list.append(board_obj_points[idx[0]])
                    # 提取对应的 2D 检测点，并确保形状为 (4, 2)
                    img_list.append(m_corners[i].reshape(4, 2))

            else:
                # 如果一个匹配的都没有，跳过
                return None, None
        # 4. 执行 PnP 解算
        if obj_points is not None and len(obj_points) >= 4:
            # 与 spear_vision 对齐：优先 SOLVEPNP_IPPE，失败回退 SOLVEPNP_ITERATIVE
            assert isinstance(img_points, np.ndarray) and isinstance(obj_points, np.ndarray), "输入点必须是 numpy 数组"
            retval, rvecs, tvecs = 0, None, None
            # try:
            #     retval, rvecs, tvecs, _ = cv2.solvePnPGeneric(
            #         obj_points, img_points, self.K, self.D,
            #         flags=cv2.SOLVEPNP_IPPE
            #     )
            # except cv2.error:
            #     retval = 0

            if retval <= 0:
                try:
                    retval, rvecs, tvecs, _ = cv2.solvePnPGeneric(
                        obj_points, img_points, self.K, self.D,
                        flags=cv2.SOLVEPNP_ITERATIVE
                    )
                except cv2.error:
                    retval = 0

            if retval > 0:
                best_rvec, best_tvec = self._select_best_pose(rvecs, tvecs)
                # print(f"Pose estimated using {method_name}")
        # 5. 绘制与返回
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