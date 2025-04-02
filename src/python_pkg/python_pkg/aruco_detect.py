import cv2
import numpy as np
import imutils
import sys
import cv_lib.cv_ros_lib
class aruco_detect_node():
    def __init__(self):
        super().__init__('aruco_detect_node')
        # self._bridge = CvBridge()
        self._handle_pipe=[]
        self._global_cotent = {}    
        self._handle_pipe.append(cv_lib.cv_ros_lib.ImagePublish_t(self, "/camera/image", 10))
   
def main ():
    # 创建 ArUco 检测节点
    node = aruco_detect_node()
    
    
if __name__ == '__main__':
    main()