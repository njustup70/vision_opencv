from ultralytics import YOLO
import numpy as np
import cv2
from pathlib import Path

class MyYOLO():
    def __init__(self, model_path, show=False, use_intel=False):
        self.model = YOLO(model_path)
        self.show = show
        if use_intel:
            import openvino.runtime as ov
            from openvino.runtime import Core
            import openvino.properties.hint as hints
            self.model = YOLO(Path(model_path).parent)
            config = {hints.performance_mode: hints.PerformanceMode.THROUGHPUT}
            core = Core()
            model = core.read_model(model_path)
            quantized_seg_compiled_model = core.compile_model(model, config=config)
            if self.model.predictor is None:
                custom = {"conf": 0.25, "batch": 1, "save": False, "mode": "predict"}
                args = {**self.model.overrides, **custom}
                self.model.predictor = self.model._smart_load("predictor")(overrides=args, _callbacks=self.model.callbacks)
                self.model.predictor.setup_model(model=self.model.model)
            self.model.predictor.model.ov_compiled_model = quantized_seg_compiled_model
            
    def update(self, image: np.ndarray):
        """
        更新图像并执行YOLO检测
        :param image: 输入图像 (numpy数组)
        :return: 包含检测结果的元组 (annotated_image, segmentation_mask, results)
        """
        results = self.model(image)
        
        # 创建分割掩码
        segmentation_mask = np.zeros_like(image, dtype=np.uint8)
        for result in results:
            if result.masks is None:
                continue
            for mask in result.masks.xy:
                mask = np.array(mask, dtype=np.int32)
                segmentation_mask = cv2.fillPoly(segmentation_mask, [mask], (0, 255, 0))
        
        annotated_image = image.copy()
        if self.show and results:
            # 绘制检测结果
            annotated_image = results[0].plot()
        
        return annotated_image, segmentation_mask, results