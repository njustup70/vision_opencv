from ultralytics import YOLO
import numpy as np
import cv2
from pathlib import Path

class MyYOLO():
    def __init__(self, model_path,show=False,use_intel=False):
        self.model = YOLO(model_path)
        self.show=show
        if use_intel:
            import openvino.runtime as ov
            from openvino.runtime import Core
            import openvino.properties.hint as hints
            self.model = YOLO(Path(model_path).parent)
            config = {hints.performance_mode: hints.PerformanceMode.THROUGHPUT}
            core = Core()
            model = core.read_model(model_path)
            quantized_seg_compiled_model  = core.compile_model(model,config=config)
            if self.model.predictor is None:
                custom = {"conf": 0.25, "batch": 1, "save": False, "mode": "predict"}  # method defaults
                args = {**self.model.overrides, **custom}
                self.model.predictor = self.model._smart_load("predictor")(overrides=args, _callbacks=self.model.callbacks)
                self.model.predictor.setup_model(model=self.model.model)
            self.model.predictor.model.ov_compiled_model = quantized_seg_compiled_model
            
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
