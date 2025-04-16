import cv2
import numpy as np
import matplotlib.pyplot as plt

def brisk_corner_detection(image_path, show_result=True):
    """
    基于 BRISK 的角点检测算法
    :param image_path: 输入图像路径
    :param show_result: 是否显示检测结果
    :return: 检测到的角点关键点列表
    """
    # 1. 读取图像并转为灰度图
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"图像 {image_path} 未找到！")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. 初始化 BRISK 检测器
    brisk = cv2.BRISK_create()  # 默认参数，也可自定义阈值、金字塔层数等

    # 3. 检测关键点并计算描述子
    keypoints, descriptors = brisk.detectAndCompute(gray, None)

    # 4. 绘制关键点
    image_with_keypoints = cv2.drawKeypoints(
        image, 
        keypoints, 
        None, 
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    # 5. 显示结果
    if show_result:
        plt.figure(figsize=(10, 6))
        plt.imshow(cv2.cvtColor(image_with_keypoints, cv2.COLOR_BGR2RGB))
        plt.title("BRISK Corner Detection")
        plt.axis("off")
        plt.show()

    return keypoints, descriptors

# 示例调用
if __name__ == "__main__":
    image_path = "example.jpg"  # 替换为你的图像路径
    keypoints, descriptors = brisk_corner_detection(image_path)
    print(f"检测到 {len(keypoints)} 个角点")