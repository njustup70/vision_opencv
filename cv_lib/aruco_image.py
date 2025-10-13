import imutils
import cv2
import sys

# 配置参数（图像路径和ArUco类型）
args = {
    "image": "/home/Elaina/yolo/PoseSolver/aruco_image/test_5x5_100.png",  # 输入图像路径
    "type": "DICT_5X5_100"  # ArUco标记类型
}

# 定义OpenCV支持的所有可能的ArUco标记名称及对应常量
ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

# 从磁盘加载输入图像并调整大小
print("[信息] 加载图像中...")
image = cv2.imread(args["image"])
image = imutils.resize(image, width=600)  # 调整图像宽度为600像素

# 验证指定的ArUco标记类型是否被OpenCV支持
if ARUCO_DICT.get(args["type"], None) is None:
    print("[信息] ArUco标记类型 '{}' 不被支持！".format(args["type"]))
    sys.exit(0)

# 加载ArUco字典，获取ArUco参数并检测标记
print("[信息] 正在检测 '{}' 类型的标记...".format(args["type"]))
arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[args["type"]])
arucoParams = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)
(corners, ids, rejected) = detector.detectMarkers(image)  # 检测标记，返回角点、ID和被拒绝的候选标记

# 验证至少检测到一个ArUco标记
if len(corners) > 0:
    # 将ArUco ID列表展平（从二维数组转为一维）
    ids = ids.flatten()
    # 遍历检测到的ArUco角点
    for (markerCorner, markerID) in zip(corners, ids):
        # 提取标记的角点，OpenCV返回的角点顺序固定为：左上、右上、右下、左下
        corners = markerCorner.reshape((4, 2))
        (topLeft, topRight, bottomRight, bottomLeft) = corners
        # 将每个（x, y）坐标对转换为整数
        topRight = (int(topRight[0]), int(topRight[1]))
        bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
        bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
        topLeft = (int(topLeft[0]), int(topLeft[1]))
        # 绘制ArUco检测结果的边界框
        cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
        cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
        cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
        cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)
        # 计算并绘制ArUco标记的中心（x, y）坐标
        cX = int((topLeft[0] + bottomRight[0]) / 2.0)
        cY = int((topLeft[1] + bottomRight[1]) / 2.0)
        cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)  # 绘制红色中心点
        # 在图像上绘制ArUco标记的ID
        cv2.putText(image, str(markerID), (topLeft[0], topLeft[1] - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        print("[信息] ArUco标记ID: {}".format(markerID))
        # 保存输出图像（取消注释即可启用）
        cv2.imwrite("{}_{}.jpg".format(args["type"], markerID), image)
        # 显示输出图像
        cv2.imshow("Image", image)
        cv2.waitKey(0)  # 等待按键输入后关闭窗口
