import numpy as np
import cv2


class MaskProcessor:
    """独立的掩膜处理类，负责掩膜后处理、多边形拟合和角点提取等操作"""
    
    @staticmethod 
    def postprocess_mask(mask_points, image_shape):
        """对掩膜进行后处理（平滑、形态学操作）"""
        mask_img = np.zeros(image_shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask_img, [mask_points.astype(np.int32)], 255)
        mask_img = cv2.GaussianBlur(mask_img, (5, 5), 0)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_img = cv2.morphologyEx(mask_img, cv2.MORPH_OPEN, kernel, iterations=1)
        mask_img = cv2.morphologyEx(mask_img, cv2.MORPH_CLOSE, kernel, iterations=1)
        contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float32) if contours else mask_points

    @staticmethod 
    def fit_polygon(contour, epsilon):
        """多边形拟合"""
        perimeter = cv2.arcLength(contour, True)
        return cv2.approxPolyDP(contour, epsilon * perimeter, True).reshape(-1, 2)

    @staticmethod 
    def extract_corner_points(approx_polygon, confidence):
        """从多边形中提取4个角点（处理4或5个顶点的情况）"""
        vertex_count = len(approx_polygon)
        if vertex_count not in [4, 5]:
            return None  # 只处理4或5个顶点的情况
        
        corner_points = approx_polygon.tolist()
        if vertex_count == 5:
            try:
                selected_points = MaskProcessor.select_four_corners(corner_points)
            except Exception as e:
                print(f"角点筛选错误: {str(e)}")
                return None
        else:
            selected_points = corner_points
        
        # 转换为 (4,2) 的 numpy 数组，并确保顺序正确
        try:
            ordered_points = MaskProcessor.order_corners(selected_points)
            return np.array(ordered_points, dtype=np.float32).reshape(-1, 2)
        except Exception as e:
            print(f"角点排序错误: {str(e)}")
            return None

    @staticmethod 
    def select_four_corners(points):
        """从5个点中筛选出4个角点（基于角度差最大间隔法）"""
        center = np.mean(points, axis=0)
        angles = [(np.arctan2(p[1]-center[1], p[0]-center[0]), p) for p in points]
        angles.sort(key=lambda x: x[0])
        angle_diffs = [( (angles[(i+1)%5][0] - angles[i][0]) % (2*np.pi), i) for i in range(5)]
        max_diff_idx = max(angle_diffs, key=lambda x: x[0])[1]
        remaining_indices = [(max_diff_idx + i) % 5 for i in range(1, 5)]
        return [angles[i][1] for i in remaining_indices]

    @staticmethod 
    def order_corners(points):
        """将4个角点排序为左上、右上、右下、左下"""
        points = np.array(points)
        sum_xy = points[:, 0] + points[:, 1]
        diff_xy = points[:, 0] - points[:, 1]
        top_left = points[np.argmin(sum_xy)]
        top_right = points[np.argmax(diff_xy)]
        bottom_right = points[np.argmax(sum_xy)]
        bottom_left = points[np.argmin(diff_xy)]
        return [top_left, top_right, bottom_right, bottom_left]