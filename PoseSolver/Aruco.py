import cv2
import imutils
from imutils.video import VideoStream
import time
import numpy as np

class Aruco:
    """
    ArUco 检测器, 支持图像和视频检测以及快速ArUco码生成
    """
    # 预定义的 ArUco 字典
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
    
    def __init__(self,aruco_type="DICT_5X5_1000", if_draw=False):
        # self.detector = None
        """初始化 ArUco 检测器"""
        if aruco_type not in self.ARUCO_DICT:
            raise ValueError(f"不支持的 ArUco 类型: {aruco_type}")
        aruco_dict = cv2.aruco.getPredefinedDictionary(self.ARUCO_DICT[aruco_type])
        aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        self.if_draw = if_draw

    def _initialize_detector(self, aruco_type):
        """初始化 ArUco 检测器"""
        if aruco_type not in self.ARUCO_DICT:
            raise ValueError(f"不支持的 ArUco 类型: {aruco_type}")
        aruco_dict = cv2.aruco.getPredefinedDictionary(self.ARUCO_DICT[aruco_type])
        aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    def aruco_maker(self, aruco_type, ids, pix, path):
        """
        ArUco码生成方法
        :param aruco_type: ArUco 字典类型
        :param ids: 欲被生成ArUco码的十进制译码
        :param pix: 生成图片的像素大小(pix * pix)
        :param path: 图片的保存路径
        """
        aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_type)
        maker = cv2.aruco.generateImageMarker(aruco_dict, ids, pix)
        cv2.imwrite(path,maker)


    def detect_image(self, input_data, aruco_type="DICT_5X5_1000", if_draw=True):
        """
        :param input_data: 可以是文件路径 (str) 或图像帧 (numpy数组)
        """
        self._initialize_detector(aruco_type)
        
        # 判断输入是路径还是图像帧
        if isinstance(input_data, str):
            image = cv2.imread(input_data)
            if image is None:
                raise FileNotFoundError(f"Image not found: {input_data}")
        elif isinstance(input_data, np.ndarray):
            image = input_data.copy()
        else:
            raise ValueError("input_data 必须是文件路径 (str) 或图像帧 (numpy数组)")
        

        image = imutils.resize(image, width=600)
        corners, ids, _ = self.detector.detectMarkers(image)

        results = []
        if ids is not None:

            for i in range(len(ids)):
                marker_id = int(ids[i][0])
                corner = corners[i][0]  # 直接获取角点数据 (shape: (4,2))
                (topLeft, _, _, bottomRight) = corner.reshape((4, 2))
                cX = int((topLeft[0] + bottomRight[0]) / 2)
                cY = int((topLeft[1] + bottomRight[1]) / 2)
                results.append({
                    "id": marker_id,
                    "cx": cX,
                    "cy": cY,
                    "corners": corner  # 新增角点数据
                })

                if if_draw:
                    self._draw_marker(image, corner.reshape((4, 2)), marker_id)

        return image if if_draw else results
    def update(self, image:np.ndarray,content:dict=None):
        """
        更新图像并检测 ArUco 标记
        :param image: OpenCV 图像对象 (np.ndarray)
        :param content: 附加的内容，如时间戳、坐标系等
        """
        corners, ids, _ = self.detector.detectMarkers(image)

        results = []
        if ids is not None:
            # 保留原始 ids 的二维结构 (n, 1)
            for i in range(len(ids)):
                marker_id = int(ids[i][0])  # 提取原始数值
                corner = corners[i].reshape((4, 2))
                (topLeft, _, _, bottomRight) = corner
                cX = int((topLeft[0] + bottomRight[0]) / 2)
                cY = int((topLeft[1] + bottomRight[1]) / 2)
                results.append({"id": marker_id, "cx": cX, "cy": cY, "corners": corner})

                if self.if_draw:
                    self._draw_marker(image, corner, marker_id)
        content['corners'] = results

    def detect_video( self, 
                      use_camera=True, 
                      camera_index=0, 
                      video_path=None, 
                      aruco_type="DICT_5X5_100", 
                      if_draw=True
                     ):
        """
        视频检测方法
        :param use_camera: 是否使用摄像头
        :param camera_index: 摄像头设备索引
        :param video_path: 视频文件路径
        :param aruco_type: ArUco 字典类型
        :param if_draw: 是否绘制标记
        :return: 逐帧输出检测结果
        """
        self._initialize_detector(aruco_type)
        
        if use_camera:
            vs = VideoStream(src=camera_index).start()
        else:
            if not video_path:
                raise ValueError("Video path is required when use_camera=False")
            vs = cv2.VideoCapture(video_path)

        time.sleep(2.0)  # 摄像头预热

        frame_count = 0  # 帧序号
        while True:
            frame_count += 1
            frame = vs.read() if use_camera else vs.read()[1]
            if frame is None:
                break

            frame = imutils.resize(frame, width=1000)
            corners, ids, _ = self.detector.detectMarkers(frame)

            if ids is not None:
                results = []
                for i in range(len(ids)):
                    marker_id = int(ids[i][0])  # 提取原始数值
                    corner = corners[i].reshape((4, 2))
                    (topLeft, _, _, bottomRight) = corner
                    cX = int((topLeft[0] + bottomRight[0]) / 2)
                    cY = int((topLeft[1] + bottomRight[1]) / 2)
                    results.append({"id": marker_id, "cx": cX, "cy": cY,"corners": corners[i][0]})


                    if if_draw:
                        self._draw_marker(frame, corner, marker_id)

                # 输出检测结果
                print(f"Frame {frame_count}:")
                for detection in results:
                    print(f"  ids: {detection['id']}  cx: {detection['cx']}  cy: {detection['cy']}")
            else:
                # 输出未检测到标记
                print(f"Frame {frame_count}: No Detection.")

            if if_draw:
                cv2.imshow("Frame", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cv2.destroyAllWindows()
        vs.stop() if use_camera else vs.release()

    def _draw_marker(self, image, corners, markerID):
        """绘制标记边界框和ID (私有方法)"""
        (topLeft, topRight, bottomRight, bottomLeft) = corners.astype("int")
        # 绘制边界框
        cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
        cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
        cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
        cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)
        # 计算中心点
        cX = int((topLeft[0] + bottomRight[0]) / 2)
        cY = int((topLeft[1] + bottomRight[1]) / 2)
        cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
        # 绘制ID
        cv2.putText(image, str(markerID), (topLeft[0], topLeft[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
