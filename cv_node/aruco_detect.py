#导入所需的库
from cv_lib import *
import time
import cv2
import numpy as np

def main():
    # camera_matix=[600.574780 ,0.000000 ,440.893136,0.000000 ,600.705625 ,235.248930,0.000000 ,0.000000 ,1.000000]
    camera_martix=np.array([[606.634521484375, 0, 433.2264404296875,],
                            [0, 606.5910034179688, 247.10369873046875],
                            [0.000000, 0.000000, 1.000000]],dtype=np.float32)
    # dist_coeffs=[0.077177 ,-0.119285 ,-0.006264 ,0.005271 ,0.000000]
    dist_coeffs=np.array([[0, 0,0, 0, 0]],dtype=np.float32)
    pipe=[]
    pipe.append(ImageReceive_t(print_latency=True))
    pipe.append(Aruco("DICT_5X5_100",if_draw=True))
    pipe.append(PoseSolver(camera_martix,dist_coeffs,0.1,print_result=True))
    pipe.append(ImagePublish_t("aruco"))
    content={}
    print_time=False
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