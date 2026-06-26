import cv2
import os
import glob
import re

# -----------------------------
# 配置存储目录
# -----------------------------
save_dir = "/home/Elaina/yolo/CameraCalibration/calibration_img"  # 你可以修改为你想要的文件夹名称或绝对路径
os.makedirs(save_dir, exist_ok=True) # 如果文件夹不存在则自动创建
print(f"照片将保存在目录: {os.path.abspath(save_dir)}")

# -----------------------------
# 初始化摄像头
# -----------------------------
cap = cv2.VideoCapture(4)  # 保持你指定的摄像头索引 4

# -----------------------------
# 自动计算起始序号（避免覆盖已有文件）
# -----------------------------
# 更新 glob 搜索路径，限定在指定的文件夹内
existing_files = glob.glob(os.path.join(save_dir, "test_calibration_*.jpg"))
count = 1  # 默认从1开始

if existing_files:
    # 提取现有文件中的最大序号
    numbers = []
    for f in existing_files:
        # 考虑到路径中可能包含其他数字，只匹配文件名部分
        basename = os.path.basename(f)
        match = re.search(r"test_calibration_(\d+).jpg", basename)
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
        filepath = os.path.join(save_dir, filename) # 拼接完整保存路径
        cv2.imwrite(filepath, frame)
        print(f"已保存：{filepath}")
        count += 1

cap.release()
cv2.destroyAllWindows()
print("程序已退出")