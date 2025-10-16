import numpy as np
import cv2


class MaskProcessor:
    
    # 掩膜处理
    def process_mask(mask_points, image_shape,  #mask_points可以更换成外部输入的数据
                    blur_ksize=(5, 5),  # 高斯模糊核（控制平滑程度）
                    morph_ksize=(5, 5), # 形态学核（控制噪点/空洞处理强度）
                    open_iter=1,        # 开运算次数（去外噪）
                    close_iter=1):      # 闭运算次数（填内洞）

        # 1. 用顶点生成初始掩膜
        mask_img = np.zeros(image_shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask_img, [mask_points.astype(np.int32)], 255)
        
        # 2. 高斯模糊平滑边缘
        if blur_ksize[0] > 1 and blur_ksize[1] > 1:
            mask_img = cv2.GaussianBlur(mask_img, blur_ksize, 0)
        
        # 3. 形态学处理（去噪+填充）
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, morph_ksize)
        if open_iter > 0:
            mask_img = cv2.morphologyEx(mask_img, cv2.MORPH_OPEN, kernel, iterations=open_iter)
        if close_iter > 0:
            mask_img = cv2.morphologyEx(mask_img, cv2.MORPH_CLOSE, kernel, iterations=close_iter)
        
        return mask_img

    # 可视化
    def visualize_mask(original_img, processed_mask):

        mask_rgb = cv2.cvtColor(processed_mask, cv2.COLOR_GRAY2BGR)
        mask_rgb[processed_mask == 255] = [0, 255, 0]  
        
        overlay = cv2.addWeighted(original_img, 0.7, mask_rgb, 0.3, 0)
        
        combined = np.vstack([mask_rgb, overlay])
        
        cv2.namedWindow("Processed Mask", cv2.WINDOW_NORMAL)
        cv2.imshow("Processed Mask", combined)
        cv2.waitKey(0)  
        cv2.destroyAllWindows()

    def Maskprocessor(original_img, mask_points, **kwargs):
        processed_mask = MaskProcessor.process_mask(mask_points, original_img.shape,** kwargs)
        MaskProcessor.visualize_final_mask(original_img, processed_mask)


