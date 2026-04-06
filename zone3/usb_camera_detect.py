from ultralytics import YOLO
import cv2
import subprocess
import os
import time

DEVICE_INDEX = 0
DEVICE_PATH = f"/dev/video{DEVICE_INDEX}"
SAVE_DIR = "saves"
WINDOW_NAME = "检测"

CONF_SHELF = 0.95  # 架子置信度
CONF_KFS = 0.9  # KFS物块置信度
WINDOW_SEC = 5

SHELF_CLASS = 15
RED_CLASS = 16
BLUE_CLASS = 17


def apply_camera_controls():
    cmds = [
        ["v4l2-ctl", "-d", DEVICE_PATH, "--set-ctrl=brightness=0"],
        ["v4l2-ctl", "-d", DEVICE_PATH, "--set-ctrl=contrast=4"],
        ["v4l2-ctl", "-d", DEVICE_PATH, "--set-ctrl=saturation=50"],
        ["v4l2-ctl", "-d", DEVICE_PATH, "--set-ctrl=exposure_time_absolute=300"],
        ["v4l2-ctl", "-d", DEVICE_PATH, "--set-ctrl=white_balance_automatic=1"],
        ["v4l2-ctl", "-d", DEVICE_PATH, "--set-ctrl=auto_exposure=3"],
    ]
    for cmd in cmds:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def detect_shelf_and_kfs(result, frame):
    shelf = None
    kfs = []

    if not result.boxes:
        return shelf, kfs

    for box in result.boxes:
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])

        # 根据类别分别过滤置信度
        if cls == SHELF_CLASS and conf >= CONF_SHELF:
            shelf = (x1, y1, x2, y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
        elif cls == RED_CLASS and conf >= CONF_KFS:
            kfs.append(("红", (x1, y1, x2, y2)))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        elif cls == BLUE_CLASS and conf >= CONF_KFS:
            kfs.append(("蓝", (x1, y1, x2, y2)))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    return shelf, kfs


def build_grid(shelf, kfs):
    x1, y1, x2, y2 = shelf
    grid = [[[] for _ in range(3)] for _ in range(3)]
    for color, (kx1, ky1, kx2, ky2) in kfs:
        cx = (kx1 + kx2) // 2
        cy = (ky1 + ky2) // 2
        i = min(2, (cy - y1) * 3 // (y2 - y1))
        j = min(2, (cx - x1) * 3 // (x2 - x1))
        grid[i][j].append(color)
    return grid


def draw_grid(frame, shelf):
    x1, y1, x2, y2 = shelf
    for i in range(1, 3):
        y = y1 + i * (y2 - y1) // 3
        x = x1 + i * (x2 - x1) // 3
        cv2.line(frame, (x1, y), (x2, y), (0, 200, 200), 2)
        cv2.line(frame, (x, y1), (x, y2), (0, 200, 200), 2)


def summarize_grid(grid, red_acc, blue_acc):
    total_kfs = 0
    current_print = ""
    for i in range(3):
        for j in range(3):
            if grid[i][j]:
                red_cnt = grid[i][j].count("红")
                blue_cnt = grid[i][j].count("蓝")
                red_acc[i][j] += red_cnt
                blue_acc[i][j] += blue_cnt
                total_kfs += red_cnt + blue_cnt
                current_print += f"({i},{j}): {red_cnt}红{blue_cnt}蓝 "
    return total_kfs, current_print

apply_camera_controls()
os.makedirs(SAVE_DIR, exist_ok=True)

model = YOLO("best.pt")
cap = cv2.VideoCapture(DEVICE_INDEX)

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 1280, 720)

window_start = time.time()
red_acc = [[0] * 3 for _ in range(3)]
blue_acc = [[0] * 3 for _ in range(3)]
frame_cnt = 0

last_print = ""

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=min(CONF_SHELF, CONF_KFS), verbose=False)  # 全局用较低阈值，后面再过滤
    shelf, kfs = detect_shelf_and_kfs(results[0], frame)

    if shelf and kfs:
        grid = build_grid(shelf, kfs)
        draw_grid(frame, shelf)
        total_kfs, grid_text = summarize_grid(grid, red_acc, blue_acc)
        frame_cnt += 1

        current_print = f"检测到: {total_kfs}个KFS\n" + grid_text

        if current_print != last_print:
            print(current_print)
            last_print = current_print
            cv2.imwrite(f"{SAVE_DIR}/detect_{len(os.listdir(SAVE_DIR))}.jpg", frame)

    # 每5秒输出综合结果
    if time.time() - window_start >= WINDOW_SEC:
        if frame_cnt > 0:  # 窗口内有有效帧才输出
            print(f"\n=== 过去{WINDOW_SEC}秒综合结果（基于{frame_cnt}帧）===")
            for i in range(3):
                for j in range(3):
                    if red_acc[i][j] > 0 or blue_acc[i][j] > 0:
                        print(f"({i},{j}): 红累计{red_acc[i][j]}次 蓝累计{blue_acc[i][j]}次")
            print("=====================================")
        # 重置累积数据
        red_acc = [[0] * 3 for _ in range(3)]
        blue_acc = [[0] * 3 for _ in range(3)]
        frame_cnt = 0
        window_start = time.time()

    cv2.imshow(WINDOW_NAME, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()