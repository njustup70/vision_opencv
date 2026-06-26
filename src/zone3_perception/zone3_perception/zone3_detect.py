#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int8MultiArray

from ultralytics import YOLO
import cv2
import subprocess
import os
import time
import yaml
import numpy as np

CALIBRATION_YAML = "/home/Elaina/yolo/CameraCalibration/cal_yaml/camera_calibration_bl.yaml"

DEVICE_INDEX = 4 
DEVICE_PATH = f"/dev/video{DEVICE_INDEX}"
WINDOW_NAME = "Detection"  

CONF_KFS = 0.8       # 物块检测置信度阈值
WINDOW_SEC = 5       # 终端周期打印时间(秒)

RED_CLASS = 16
BLUE_CLASS = 17

ROI_CORNERS = [
    [0, 0],  # 左上角
    [640, 0], # 右上角
    [640, 480], # 右下角
    [0, 480]   # 左下角
]

VIRTUAL_SIZE = 300 

def load_camera_intrinsics(yaml_path):
    """读取相机内参和畸变系数"""
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"找不到标定文件: {yaml_path}")
    with open(yaml_path, 'r') as f:
        calib_data = yaml.safe_load(f)
        
    cam_mat = calib_data['camera_matrix']
    mtx = np.array(cam_mat['data']).reshape(3, 3) if isinstance(cam_mat, dict) else np.array(cam_mat).reshape(3, 3)
        
    dist_coeffs = calib_data['dist_coeffs']
    dist = np.array(dist_coeffs['data']) if isinstance(dist_coeffs, dict) else np.array(dist_coeffs)
        
    return mtx, dist

def apply_camera_controls():
    """初始化底层摄像头参数"""
    cmds = [
        ["v4l2-ctl", "-d", DEVICE_PATH, "--set-ctrl=brightness=30"],
        ["v4l2-ctl", "-d", DEVICE_PATH, "--set-ctrl=contrast=4"],
        ["v4l2-ctl", "-d", DEVICE_PATH, "--set-ctrl=saturation=50"],
        ["v4l2-ctl", "-d", DEVICE_PATH, "--set-ctrl=exposure_time_absolute=300"],
        ["v4l2-ctl", "-d", DEVICE_PATH, "--set-ctrl=white_balance_automatic=1"],
        ["v4l2-ctl", "-d", DEVICE_PATH, "--set-ctrl=auto_exposure=3"],
    ]
    for cmd in cmds:
        subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def init_perspective_matrices():
    """初始化透视变换矩阵"""
    pts_src = np.array(ROI_CORNERS, dtype=np.float32)
    pts_dst = np.array([
        [0, 0], 
        [VIRTUAL_SIZE, 0], 
        [VIRTUAL_SIZE, VIRTUAL_SIZE], 
        [0, VIRTUAL_SIZE]
    ], dtype=np.float32)
    
    M = cv2.getPerspectiveTransform(pts_src, pts_dst)
    M_inv = cv2.getPerspectiveTransform(pts_dst, pts_src)
    return M, M_inv

def draw_perspective_grid(frame, M_inv):
    """在真实画面上绘制形变后的九宫格"""
    grid_color = (0, 200, 200)
    thickness = 2
    step = VIRTUAL_SIZE // 3
    
    lines_virtual = [
        [[0, step], [VIRTUAL_SIZE, step]],
        [[0, step*2], [VIRTUAL_SIZE, step*2]],
        [[step, 0], [step, VIRTUAL_SIZE]],
        [[step*2, 0], [step*2, VIRTUAL_SIZE]]
    ]
    
    for line in lines_virtual:
        pts = np.array([line], dtype=np.float32)
        pts_real = cv2.perspectiveTransform(pts, M_inv)[0]
        p1 = (int(pts_real[0][0]), int(pts_real[0][1]))
        p2 = (int(pts_real[1][0]), int(pts_real[1][1]))
        cv2.line(frame, p1, p2, grid_color, thickness)
        
    pts_src = np.array(ROI_CORNERS, np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [pts_src], True, grid_color, thickness)

def detect_and_map_kfs(result, frame, M):
    """解析 YOLO 结果，框出物块，并通过透视矩阵计算其所在的九宫格坐标"""
    detected_blocks = []

    if not result.boxes:
        return detected_blocks

    for box in result.boxes:
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        
        if conf < CONF_KFS:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        
        pt_real = np.array([[[cx, cy]]], dtype=np.float32)
        pt_virtual = cv2.perspectiveTransform(pt_real, M)[0][0]
        vx, vy = pt_virtual[0], pt_virtual[1]
        
        if 0 <= vx <= VIRTUAL_SIZE and 0 <= vy <= VIRTUAL_SIZE:
            grid_i = int(vy // (VIRTUAL_SIZE / 3))
            grid_j = int(vx // (VIRTUAL_SIZE / 3))
            grid_i = min(2, max(0, grid_i))
            grid_j = min(2, max(0, grid_j))
            
            if cls == RED_CLASS:
                detected_blocks.append(("红", grid_i, grid_j))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"R ({grid_i},{grid_j})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
            elif cls == BLUE_CLASS:
                detected_blocks.append(("蓝", grid_i, grid_j))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"B ({grid_i},{grid_j})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    return detected_blocks

def main():
    rclpy.init()
    node = Node('yolo_grid_publisher')
    pub = node.create_publisher(Int8MultiArray, 'grid_state', 10)

    apply_camera_controls()
    
    print("正在加载相机标定参数...")
    mtx, dist = load_camera_intrinsics(CALIBRATION_YAML)
    
    M, M_inv = init_perspective_matrices()
    model = YOLO("best.pt") 

    cap = cv2.VideoCapture(DEVICE_INDEX)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    ret, sample_frame = cap.read()
    if not ret:
        print("初始读取摄像头失败，请检查连接。")
        node.destroy_node()
        rclpy.shutdown()
        return
        
    h, w = sample_frame.shape[:2]
    new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, new_mtx, (w, h), cv2.CV_16SC2)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)

    window_start = time.time()
    acc_red = 0
    acc_blue = 0
    frame_cnt = 0
    last_print = ""
    retry_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            retry_count += 1
            if retry_count > 5:
                print("连续 5 次读取失败，退出。")
                break
            time.sleep(0.1)
            continue
            
        retry_count = 0  

        frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
        draw_perspective_grid(frame, M_inv)
        
        results = model(frame, conf=CONF_KFS, verbose=False)
        detected_blocks = detect_and_map_kfs(results[0], frame, M)

        grid = [0] * 9  # 0 空, 1 红, 2 蓝
        for color, i, j in detected_blocks:
            idx = i * 3 + j
            if color == "红":
                grid[idx] = 1
            elif color == "蓝":
                grid[idx] = 2
        msg = Int8MultiArray()
        msg.data = grid
        pub.publish(msg)-

        if detected_blocks:
            frame_cnt += 1
            
            block_details = []
            cur_red, cur_blue = 0, 0
            for color, i, j in detected_blocks:
                block_details.append(f"{color}({i},{j})")
                if color == "红": cur_red += 1
                if color == "蓝": cur_blue += 1
                
            acc_red += cur_red
            acc_blue += cur_blue

            current_print = f"检出 {len(detected_blocks)} 块 | 详细位置: " + " ".join(block_details)

            if current_print != last_print:
                print(f"[即时状态] {current_print}")
                last_print = current_print

        if time.time() - window_start >= WINDOW_SEC:
            # 周期统计打印（可根据需要启用）
            # if frame_cnt > 0:
            #     print(f"\n=== 过去 {WINDOW_SEC} 秒综合结果（基于 {frame_cnt} 帧） ===")
            #     print(f"红色总累计抓取人次: {acc_red}")
            #     print(f"蓝色总累计抓取人次: {acc_blue}")
            #     print("===================================================\n")
            acc_red = 0
            acc_blue = 0
            frame_cnt = 0
            window_start = time.time()

        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # 处理ROS2回调
        rclpy.spin_once(node, timeout_sec=0.001)

    cap.release()
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()