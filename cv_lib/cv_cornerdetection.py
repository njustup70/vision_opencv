import cv2
import time

class BRISKCornerDetector:
    def __init__(self, show_result=True):
        """
        BRISK角点检测器
        :param show_result: 是否实时显示角点检测结果
        """
        self.brisk = cv2.BRISK_create()
        self.show_result = show_result

    def detect(self, image):
        """
        执行角点检测
        :param image: 输入图像（BGR格式）
        :return: (keypoints, descriptors)
        """
        if image is None or image.size == 0:
            return None, None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.brisk.detectAndCompute(gray, None)

        if self.show_result:
            img_kp = cv2.drawKeypoints(
                image, keypoints, None, 
                #flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
                flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT  # 只显示简单点
            )
            cv2.imshow("BRISK Corners", img_kp)
            cv2.waitKey(1)

        return keypoints, descriptors

    def update(self, image, content):
        """
        管道模式要求的更新方法
        :param image: 输入图像
        :param content: 共享数据字典
        :return: 修改后的图像和内容
        """
        keypoints, descriptors = self.detect(image)
        
        # 将结果存入content字典，供后续模块使用
        content["keypoints"] = keypoints
        content["descriptors"] = descriptors
        
        return image, content