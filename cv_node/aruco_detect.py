#导入所需的库
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 
print(sys.path)
import time
import cv2
import numpy as np
from cv_lib.cv_bridge import ImagePublish_t,ImageReceive_t
from ArUco.Aruco import Aruco

def main():
    
    pipe=[]
    pipe.append(ImageReceive_t(print_latency=True))
    pipe.append(Aruco("DICT_5X5_100",if_draw=True))
    pipe.append(ImagePublish_t("aruco"))
    content={}
    print_time=True
    while True:
        # 创建一个空的图像对象，这里用一个全黑的图像作为示例
        image = np.zeros((1, 1, 3), dtype=np.uint8)
        #启动处理
        for p in pipe:
            if print_time:
                start_time = time.time()
            p.update(image,content)
            if print_time:
                end_time = time.time()
                print(f"name:{type(p).__name__}: {(end_time - start_time)*1000:2f} ms")
                print(content)
if __name__ == "__main__":
    main()