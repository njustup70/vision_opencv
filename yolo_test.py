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

    def update(self, image: np.ndarray, content: dict, confidence_threshold=0.5):
        results = self.model(image)
        content["corners"] = []
        content["masks"] = []  # 存储掩膜信息

        # 存储所有符合阈值条件的候选掩膜
        valid_masks = []

        for result in results:
            if result.masks is None:
                continue

            # 获取当前结果中的所有检测框置信度
            if hasattr(result, 'boxes') and hasattr(result.boxes, 'conf'):
                confidences = result.boxes.conf.cpu().numpy()
            else:
                confidences = np.array([0.0])  # 默认值

            # 确保掩膜和置信度数量匹配
            num_masks = len(result.masks.xy)
            num_confs = len(confidences)
            valid_range = min(num_masks, num_confs)

            for i in range(valid_range):
                try:
                    mask = result.masks.xy[i]
                    conf = float(confidences[i])
                
                    # 跳过置信度低于阈值的掩膜
                    if conf < confidence_threshold:
                        continue
                    
                    mask_points = np.array(mask, dtype=np.float32).reshape(-1, 2)
                    if len(mask_points) < 4:
                        continue

                    # 只存储基本信息，延迟处理
                    valid_masks.append({
                        "mask": mask_points,
                        "confidence": conf,
                        "raw_result": result,  # 保存原始结果用于后续处理
                        "mask_index": i        # 保存掩膜索引
                    })
                except Exception as e:
                    print(f"掩膜筛选出错: {str(e)}")
                    continue

        # 存储所有有效掩膜信息
        for mask_info in valid_masks:
            # 对掩膜进行后处理
            processed_mask = self._postprocess_mask(mask_info["mask"], image.shape[:2])
            
            # 保存处理结果
            content["masks"].append({
                "mask": processed_mask,
                "confidence": mask_info["confidence"],
            })

        # 可视化结果
        if self.show and len(results) > 0:
            self._visualize_results(results[0], image, content, confidence_threshold)

    def _postprocess_mask(self, mask_points, image_shape):
        """对预测掩膜进行后处理，提升质量"""
        # 创建掩膜
        mask_img = np.zeros(image_shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask_img, [mask_points.astype(np.int32)], 255)
        
        # 平滑处理 - 可调整高斯核大小
        mask_img = cv2.GaussianBlur(mask_img, (5, 5), 0)
        
        # 形态学操作 - 可调整核大小和迭代次数
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_img = cv2.morphologyEx(mask_img, cv2.MORPH_OPEN, kernel, iterations=1)
        mask_img = cv2.morphologyEx(mask_img, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # 查找轮廓并优化
        contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return mask_points  # 返回原始点作为后备
        
        # 获取最大轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 多边形逼近 - 可调整epsilon值
        epsilon = 0.01 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # 返回优化后的轮廓点
        return approx.reshape(-1, 2).astype(np.float32)

    def _visualize_results(self, result, image, content, confidence_threshold=0.5):
        """可视化检测结果"""
        try:
            # 创建空白图像用于绘制
            image_result = np.zeros_like(image)
        
            # 绘制筛选后的掩膜
            for mask_data in content["masks"]:
                confidence = mask_data["confidence"]
                if confidence >= confidence_threshold:
                    mask_points = mask_data["mask"].astype(np.int32)
                
                    # 绘制掩膜区域
                    cv2.fillPoly(image_result, [mask_points], (0, 255, 0, 50))  # 半透明绿色
                
                    # 添加置信度文本
                    # 计算掩膜的中心点用于放置文本
                    M = cv2.moments(mask_points)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        cv2.putText(image_result, f"Conf: {confidence:.2f}",
                                    (cX, cY),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
            # 将绘制结果合并到原图上
            if len(content["masks"]) > 0:
                image[:] = cv2.addWeighted(image, 1, image_result, 0.7, 0)
            else:
                image[:] = image_result  # 如果没有筛选出掩膜，显示原图
            
        except Exception as e:
            print(f"可视化出错: {str(e)}")
