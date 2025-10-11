from Aruco import Aruco
from PoseSolver import PoseSolver
from imutils.video import VideoStream
import cv2
import numpy as np
import time

# ================== 手动输入相机参数 ==================
camera_matrix = np.array([
    [714.9199755874305, 0.0, 354.4380105887537], 
    [0.0, 715.2231562188343, 715.2231562188343],   
    [0.0, 0.0, 1.0]          
], dtype=np.float32)

dist_coeffs = np.array([
    [-0.15781026869102965,  1.583778819087774, -0.011256244004513636, 0.011363172123126099, -4.632718536609601]
], dtype=np.float32)

marker_length = 0.047  # ArUco码实际边长 (单位: 米)

# ================== 初始化模块 ==================
aruco_detector = Aruco()
pose_solver = PoseSolver(
    camera_matrix=camera_matrix,
    dist_coeffs=dist_coeffs,
    marker_length=marker_length
)

# ================== 实时视频模式 ==================
vs = VideoStream(src=0).start()  # 使用默认摄像头
time.sleep(2.0)

while True:
    frame = vs.read()
    if frame is None:
        break

    # Step 1: 检测ArUco码 (调用原有封装方法)
    results = aruco_detector.detect_image(
        frame,  
        aruco_type="DICT_5X5_100",
        if_draw=False
    )

    # Step 2: 对每个检测到的ArUco码解算位姿
    if results:
        for detection in results:
            corners = detection["corners"]  # 假设返回结果包含角点数据
            try:
                rvec, tvec = pose_solver.solve_pose(corners)
                pose_solver.draw_axis(frame, rvec, tvec)

                # 打印位姿信息
                print(f"ID: {detection['id']} | Tvec: {tvec.flatten().round(3)} | Rvec: {rvec.flatten().round(3)}")
                rotation_matrix, _ = cv2.Rodrigues(rvec)
                sy = np.sqrt(rotation_matrix[0,0] ** 2 + rotation_matrix[1,0] ** 2)
                pitch = np.arctan2(rotation_matrix[2,1], rotation_matrix[2,2])  # X轴旋转角
                yaw = np.arctan2(-rotation_matrix[2,0], sy)                     # Y轴旋转角
                roll = np.arctan2(rotation_matrix[1,0], rotation_matrix[0,0])   # Z轴旋转角
                print(f"Pitch: {np.degrees(pitch):.1f}°, Yaw: {np.degrees(yaw):.1f}°, Roll: {np.degrees(roll):.1f}°")
                print("")

            except Exception as e:
                print(f"ID {detection['id']} 位姿解算失败: {e}")

    # 显示画面
    cv2.imshow("Real-time Pose Estimation", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vs.stop()
cv2.destroyAllWindows()

