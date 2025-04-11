from ultralytics import YOLO
import numpy as np
import cv2
class MyYOLO():
    def __init__(self, model_path,show=False):
        self.model = YOLO(model_path)
        self.show=show
    def update(self,image:np.ndarray,content:dict):
        results=self.model(image)
        # 把结果显示到图片上
        #创建一个空的图像对象
        segmentation_mask = np.zeros_like(image, dtype=np.uint8)
        for i,result in enumerate(results):
            if result.masks is None:
                continue
            for j,mask in enumerate(result.masks.xy):
                mask=np.array(mask,dtype=np.int32)
                segmentation_mask=cv2.fillPoly(segmentation_mask, [mask], (0, 255, 0))
        if self.show:
            # cv2.addWeighted(image, 1, segmentation_mask, 0.7, 0, image)  # 将修改写回原图像
            result = results[0]
            image_result= result.plot()
            #将image_result内容给image
            image[:]=image_result
            