#导入所需的库
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 
print(sys.path)

import cv2
import numpy as np
from cv_lib.cv_bridge import ImagePublish_t,ImageReceive_t
from ArUco.Aruco import Aruco

def main():
    
    pipe=[]
    pipe.append(ImageReceive_t())
    pipe.append(Aruco("DICT_4X4_50"))
    pipe.append(ImagePublish_t("aruco"))
    content={}
    while True:
        # 创建一个空的图像对象，这里用一个全黑的图像作为示例
        image = np.zeros((1, 1, 3), dtype=np.uint8)
        #启动处理
        for p in pipe:
            p.update(image,content)
        print(content)
if __name__ == "__main__":
    main()