from ultralytics import YOLO
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

config_path = "yaml/kfs.yaml"

import glob
images = glob.glob("../kfs_data/images/train/*.jpg") + glob.glob("../kfs_data/images/train/*.png")
labels = glob.glob("../kfs_data/labels/train/*.txt")

print(f"找到 {len(images)} 张图片, {len(labels)} 个标注文件")

if len(images) == 0:
    print("错误: 没有训练图片！")
    print("请将架子图片放入 kfs_data/images/train/")
    exit(1)

# 开始训练
print("开始训练...")
model = YOLO("yolo11n.pt")
model.train(
    data=config_path,
    epochs=500,
    imgsz=640,
    batch=8,
    device="0"
)

print("训练完成！")
print(f"模型保存在: runs/detect/train/weights/best.pt")