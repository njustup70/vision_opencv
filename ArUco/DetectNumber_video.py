from imutils.video import VideoStream
import imutils
import time
import cv2

ARUCO_DICT_TYPE = "DICT_5X5_100"  

# 选择输入源（摄像头或视频文件）
USE_CAMERA = True  # 如果为 True，使用摄像头；如果为 False，使用视频文件
CAMERA_INDEX = 1
VIDEO_PATH = "E:/Code/NJUST-UP70/vision_yolo/ArUco/test_video.mp4"  # 视频文件路径（仅在 USE_CAMERA=False 时生效）

# 定义 OpenCV 支持的 ArUco 字典
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

# 检查 ArUco 字典类型是否支持
if ARUCO_DICT.get(ARUCO_DICT_TYPE, None) is None:
    print(f"[ERROR] ArUco 类型 '{ARUCO_DICT_TYPE}' 不支持！")
    exit()

# 加载 ArUco 字典和参数
print(f"[INFO] 检测 '{ARUCO_DICT_TYPE}' 类型的 ArUco 码...")
arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[ARUCO_DICT_TYPE])
arucoParams = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)

print("[INFO] 初始化视频流...")
if USE_CAMERA:
    vs = VideoStream(src=CAMERA_INDEX).start()  
else:
    vs = cv2.VideoCapture(VIDEO_PATH)  
time.sleep(2.0)  

# 循环处理视频帧
while True:
    if USE_CAMERA:
        frame = vs.read()  
    else:
        ret, frame = vs.read()  
        if not ret:
            print("[INFO] 视频播放结束！")
            break

    frame = imutils.resize(frame, width=1000)

    (corners, ids, rejected) = detector.detectMarkers(frame)
    if ids is not None:
        ids = ids.flatten()
        # 遍历检测到的 ArUco 码
        for (markerCorner, markerID) in zip(corners, ids):
            # 提取角点
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners
            # 转换为整数坐标
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))
            # 绘制边界框
            cv2.line(frame, topLeft, topRight, (0, 255, 0), 2)
            cv2.line(frame, topRight, bottomRight, (0, 255, 0), 2)
            cv2.line(frame, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv2.line(frame, bottomLeft, topLeft, (0, 255, 0), 2)
            # 计算并绘制中心点
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            cv2.circle(frame, (cX, cY), 4, (0, 0, 255), -1)
            # 绘制 ArUco 码 ID
            cv2.putText(frame, str(markerID), (topLeft[0], topLeft[1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
if USE_CAMERA:
    vs.stop()  
else:
    vs.release()  