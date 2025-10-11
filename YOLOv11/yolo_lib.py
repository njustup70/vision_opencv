from ultralytics import YOLO
import numpy as np
import cv2
from pathlib import Path
from cv_mask.cv_mask import MaskProcessor

class MyYOLO:
    def __init__(self, model_path, show=False, use_intel=False):
        self.model = YOLO(model_path)
        self.show = show
        self.mask_processor = MaskProcessor()  # 实例化掩膜处理器
        
        if use_intel:
            # OpenVINO 加速部分（若无需加速可忽略）
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

    def update(self, image: np.ndarray, confidence_threshold=0.8, epsilon=0.05):
        """
        处理图像并输出兼容 PoseSolver 的角点格式
        :param image: 输入图像(BGR格式)
        :param confidence_threshold: 置信度阈值
        :param epsilon: 多边形拟合参数
        :return: 角点列表,每个元素的形状巍(4,2)的numpy的数组
        """
        results = self.model(image)
        corners = []

        # 筛选有效掩膜
        valid_masks = self._filter_valid_masks(results, confidence_threshold)

        # 处理每个有效掩膜
        for mask_info in valid_masks:
            # 使用独立的掩膜处理器进行后处理
            processed_points = self.mask_processor.postprocess_mask(mask_info["mask"], image.shape[:2])
            contour = processed_points.astype(np.int32).reshape(-1, 1, 2)
            approx_polygon = self.mask_processor.fit_polygon(contour, epsilon)
            
            # 提取并筛选角点
            corner_points = self.mask_processor.extract_corner_points(approx_polygon, mask_info["confidence"])
            if corner_points is not None and len(corner_points) == 4:
                corners.append(corner_points)

        # 可视化结果
        if self.show and corners:
            self._visualize_results(corners, image, confidence_threshold)

        return corners if corners else None

    def _filter_valid_masks(self, results, confidence_threshold):
        """筛选出符合置信度阈值的掩膜"""
        valid_masks = []
        for result in results:
            if result.masks is None:
                continue
            confidences = result.boxes.conf.cpu().numpy() if hasattr(result.boxes, 'conf') else np.array([0.0])
            for i in range(min(len(result.masks.xy), len(confidences))):
                try:
                    mask = result.masks.xy[i]
                    conf = float(confidences[i])
                    if conf < confidence_threshold:
                        continue
                    mask_points = np.array(mask, dtype=np.float32).reshape(-1, 2)
                    if len(mask_points) < 4:
                        continue
                    valid_masks.append({"mask": mask_points, "confidence": conf})
                except Exception as e:
                    print(f"掩膜筛选错误: {str(e)}")
        return valid_masks

    def _visualize_results(self, corners, image, confidence_threshold):
        """可视化多边形拟合结果"""
        image_result = np.zeros_like(image)
        for corner_points in corners:
            polygon = corner_points.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(image_result, [polygon], True, (0, 0, 255), 2)
            for point in polygon:
                cv2.circle(image_result, tuple(point[0]), 3, (255, 0, 0), -1)
            # 这里假设置信度为 1.0，可根据实际情况修改
            conf = 1.0
            cv2.putText(image_result, f"Conf: {conf:.2f}, Vertices: {len(corner_points)}",
                        (polygon[0][0][0], polygon[0][0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        image[:] = cv2.addWeighted(image, 1, image_result, 0.5, 0)
