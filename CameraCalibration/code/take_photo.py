import cv2
import os
import glob
import re

# -----------------------------
# 初始化摄像头
# -----------------------------
cap = cv2.VideoCapture(1)  # 0 表示默认摄像头，外接摄像头可尝试改为1

# -----------------------------
# 自动计算起始序号（避免覆盖已有文件）
# -----------------------------
existing_files = glob.glob("test_calibration_*.jpg")
count = 1  # 默认从1开始

if existing_files:
    # 提取现有文件中的最大序号
    numbers = []
    for f in existing_files:
        match = re.search(r"test_calibration_(\d+).jpg", f)
        if match:
            numbers.append(int(match.group(1)))
    if numbers:
        count = max(numbers) + 1

# -----------------------------
# 拍照主程序
# -----------------------------
print("摄像头已启动，按 [ENTER] 拍照，按 [ESC] 退出")

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法读取摄像头画面")
        break

    cv2.imshow("Camera Preview", frame)
    key = cv2.waitKey(1) & 0xFF
    
    # 按 ESC 退出
    if key == 27:
        break
    
    # 按 ENTER 拍照（Windows/Linux均兼容）
    if key == 13 or key == 10:
        filename = f"test_calibration_{count}.jpg"
        cv2.imwrite(filename, frame)
        print(f"已保存：{filename}")
        count += 1

cap.release()
cv2.destroyAllWindows()
print("程序已退出")