from ultralytics import YOLO
import cv2
import subprocess
import os
import time
import yaml
import numpy as np

# ==========================================
# 标定文件路径配置
# ==========================================
CALIBRATION_YAML = "/home/Elaina/yolo/CameraCalibration/cal_yaml/camera_calibration_bl.yaml"

DEVICE_INDEX = 4 
DEVICE_PATH = f"/dev/video{DEVICE_INDEX}"
SAVE_DIR = "saves"
WINDOW_NAME = "Detection"  

CONF_SHELF = 0.95  
CONF_KFS = 0.8   
WINDOW_SEC = 5    

SHELF_CLASS = 15
RED_CLASS = 16
BLUE_CLASS = 17

def load_camera_intrinsics(yaml_path):
    """读取相机内参和畸变系数（自动兼容嵌套字典或纯列表格式）"""
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"找不到标定文件: {yaml_path}")
        
    with open(yaml_path, 'r') as f:
        calib_data = yaml.safe_load(f)
        
    # 解析内参矩阵
    cam_mat = calib_data['camera_matrix']
    if isinstance(cam_mat, dict) and 'data' in cam_mat:
        mtx = np.array(cam_mat['data']).reshape(3, 3)
    else:
        # 如果直接是列表格式，直接转为 numpy 数组并 reshape
        mtx = np.array(cam_mat).reshape(3, 3)
        
    # 解析畸变系数
    dist_coeffs = calib_data['dist_coeffs']
    if isinstance(dist_coeffs, dict) and 'data' in dist_coeffs:
        dist = np.array(dist_coeffs['data'])
    else:
        # 如果直接是列表格式，直接转为 numpy 数组
        dist = np.array(dist_coeffs)
        
    return mtx, dist

def apply_camera_controls():
    cmds = [
        ["v4l2-ctl", "-d", DEVICE_PATH, "--set-ctrl=brightness=50"],
        ["v4l2-ctl", "-d", DEVICE_PATH, "--set-ctrl=contrast=4"],
        ["v4l2-ctl", "-d", DEVICE_PATH, "--set-ctrl=saturation=50"],
        ["v4l2-ctl", "-d", DEVICE_PATH, "--set-ctrl=exposure_time_absolute=300"],
        ["v4l2-ctl", "-d", DEVICE_PATH, "--set-ctrl=white_balance_automatic=1"],
        ["v4l2-ctl", "-d", DEVICE_PATH, "--set-ctrl=auto_exposure=3"],
    ]
    
    for cmd in cmds:
        subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def detect_shelf_and_kfs(result, frame):
    shelf = None
    kfs = []

    if not result.boxes:
        return shelf, kfs

    for box in result.boxes:
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])

        if cls == SHELF_CLASS and conf >= CONF_SHELF:
            shelf = (x1, y1, x2, y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
            cv2.putText(frame, f"Shelf {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
        elif cls == RED_CLASS and conf >= CONF_KFS:
            kfs.append(("红", (x1, y1, x2, y2)))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
        elif cls == BLUE_CLASS and conf >= CONF_KFS:
            kfs.append(("蓝", (x1, y1, x2, y2)))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    return shelf, kfs

def build_grid(shelf, kfs):
    """基于架子坐标构建 3x3 空间网格，并将物块映射进去"""
    x1, y1, x2, y2 = shelf
    grid = [[[] for _ in range(3)] for _ in range(3)]
    
    width, height = x2 - x1, y2 - y1
    if width <= 0 or height <= 0:
        return grid

    for color, (kx1, ky1, kx2, ky2) in kfs:
        cx = (kx1 + kx2) // 2
        cy = (ky1 + ky2) // 2
        
        i = max(0, min(2, (cy - y1) * 3 // height))
        j = max(0, min(2, (cx - x1) * 3 // width))
        grid[i][j].append(color)
        
    return grid

def draw_grid(frame, shelf):
    "九宫格"
    x1, y1, x2, y2 = shelf
    for i in range(1, 3):
        y = y1 + i * (y2 - y1) // 3
        x = x1 + i * (x2 - x1) // 3
        cv2.line(frame, (x1, y), (x2, y), (0, 200, 200), 2)
        cv2.line(frame, (x, y1), (x, y2), (0, 200, 200), 2)

def summarize_grid(grid, red_acc, blue_acc):
    total_kfs = 0
    current_print_parts = []
    
    for i in range(3):
        for j in range(3):
            if grid[i][j]:
                red_cnt = grid[i][j].count("红")
                blue_cnt = grid[i][j].count("蓝")
                
                red_acc[i][j] += red_cnt
                blue_acc[i][j] += blue_cnt
                total_kfs += red_cnt + blue_cnt
                
                current_print_parts.append(f"({i},{j}): {red_cnt}红{blue_cnt}蓝")
                
    return total_kfs, "  ".join(current_print_parts)

def main():
    apply_camera_controls()
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # -----------------------------
    # 1. 加载标定参数并预计算映射矩阵
    # -----------------------------
    print("正在加载相机标定参数...")
    mtx, dist = load_camera_intrinsics(CALIBRATION_YAML)

    saved_img_count = len([name for name in os.listdir(SAVE_DIR) if name.endswith('.jpg')])
    model = YOLO("best.pt") 

    cap = cv2.VideoCapture(DEVICE_INDEX)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # 获取相机分辨率以生成映射表
    ret, sample_frame = cap.read()
    if not ret:
        print("初始读取摄像头失败，请检查连接。")
        return
        
    h, w = sample_frame.shape[:2]
    # alpha=0 表示裁剪掉畸变校正后产生的黑边，保证画面无畸变且填满窗口
    new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))
    # 预先生成 X, Y 像素映射表（极大提升运行帧率）
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, new_mtx, (w, h), cv2.CV_16SC2)
    print("映射矩阵初始化完成。")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)

    window_start = time.time()
    red_acc = [[0] * 3 for _ in range(3)]
    blue_acc = [[0] * 3 for _ in range(3)]
    frame_cnt = 0
    last_print = ""
    
    cached_shelf = None
    retry_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            retry_count += 1
            if retry_count > 5:
                print("连续 5 次无法读取摄像头画面，程序退出。")
                break
            time.sleep(0.1)
            continue
            
        retry_count = 0  

        # -----------------------------
        # 2. 对当前帧执行极速去畸变映射
        # -----------------------------
        frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
        
        # 将去畸变后的规整画面喂给 YOLO
        results = model(frame, conf=min(CONF_SHELF, CONF_KFS), verbose=False)
        shelf, kfs = detect_shelf_and_kfs(results[0], frame)

        # 缓存架子坐标
        if shelf:
            cached_shelf = shelf
        else:
            shelf = cached_shelf 

        # 处理网格和物块
        if shelf and kfs:
            grid = build_grid(shelf, kfs)
            draw_grid(frame, shelf)
            total_kfs, grid_text = summarize_grid(grid, red_acc, blue_acc)
            frame_cnt += 1

            current_print = f"检测到: {total_kfs} 个 KFS | {grid_text}"

            # 状态发生变化时打印并保存图片
            if current_print != last_print:
                print(f"[即时状态] {current_print}")
                last_print = current_print
                
                img_path = os.path.join(SAVE_DIR, f"detect_{saved_img_count:04d}.jpg")
                cv2.imwrite(img_path, frame)
                saved_img_count += 1

        # 时间窗口周期结算
        if time.time() - window_start >= WINDOW_SEC:
            if frame_cnt > 0:
                print(f"\n=== 过去 {WINDOW_SEC} 秒综合结果（基于 {frame_cnt} 帧有效检测） ===")
                for i in range(3):
                    for j in range(3):
                        if red_acc[i][j] > 0 or blue_acc[i][j] > 0:
                            print(f"格位 ({i},{j}) -> 红累计: {red_acc[i][j]} 次，蓝累计: {blue_acc[i][j]} 次")
                print("===================================================\n")
            
            red_acc = [[0] * 3 for _ in range(3)]
            blue_acc = [[0] * 3 for _ in range(3)]
            frame_cnt = 0
            window_start = time.time()

        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()