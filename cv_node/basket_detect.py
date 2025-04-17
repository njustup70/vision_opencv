#导入所需的库
from cv_lib import *
import numpy as np
import time
from os.path import expanduser
def main():
    camera_martix=np.array([[606.634521484375, 0, 433.2264404296875,],
                            [0, 606.5910034179688, 247.10369873046875],
                            [0.000000, 0.000000, 1.000000]],dtype=np.float32)
    dist_coeffs=np.array([[0, 0,0, 0, 0]],dtype=np.float32)
    model_path=expanduser("~/yolo/YOLOv11/models/yolo11n-seg_int8_openvino_model/yolo11n-seg.xml")
    pipe=[]
    pipe.append(ImageReceive_t(print_latency=True))
    pipe.append(MyYOLO(model_path,show=True,use_intel=True))
    pipe.append(ImagePublish_t("yolo"))
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
if __name__ == "__main__":
    main()