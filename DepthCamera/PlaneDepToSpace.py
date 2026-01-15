# 找到框内深度图的主要深度峰值，并计算质心像素坐标
import numpy as np
from DepthCamera import DepthCamera, pix_to_cam

import os
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"


class DepthCamera_change(DepthCamera):
    def __init__(self):
        super().__init__()
        self.bin_width = 0.01  # Histogram bin width in meters
        self.d2c_r = None
        self.d2c_t = None
        self.peak_width = 3 * self.bin_width
        self.bin_min_percent = 0.005


    def depthPixelToColor(self, u, v, depth):
        point_d = pix_to_cam(u, v, depth, self.model_d)
        point_c = self.d2c_r @ point_d + self.d2c_t
        pix_c = self.model_c.project3dToPixel((point_c[0], point_c[1], point_c[2]))
        return pix_c

    def depthImageFindCenter(self, box_range, img):
        depth_img = self.bridge.imgmsg_to_cv2(img, desired_encoding='passthrough').astype(np.float32) / 1000.0  # Convert mm to meters
        depth_need = depth_img[box_range[0]:box_range[2], box_range[1]:box_range[3]]
        depth_valid = depth_need[~np.isnan(depth_need) & (depth_need != 0)]
        if depth_valid.size == 0:
            print("No valid depth data in the specified range.")
            return None
        # 建立深度直方图
        depth_min, depth_max = np.min(depth_valid), np.max(depth_valid)
        bins = max(1, int((depth_max - depth_min) / self.bin_width))
        hist, bin_edges = np.histogram(depth_valid, bins=bins, range=(depth_min, depth_max))
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # 每个bin中心
        # 寻找直方图中大于bin_min_percent的数据作为峰值
        #peaks, props = find_peaks(hist, height=0.1*depth_valid.size) # 最小高度为总点数的10%
        peaks = np.where(hist > self.bin_min_percent * depth_valid.size)[0]
        if len(peaks) == 0:
            print("No significant peaks found in depth histogram.")
            return
        peak_positions = centers[peaks]
        #peak_heights = props["peak_heights"]
        # 合并接近的峰值
        merged = []
        cur_group = [0]
        for i in range(1, len(peak_positions)):
            if abs(peak_positions[i] - peak_positions[cur_group[-1]]) < self.peak_width:
                cur_group.append(i)
            else:
                merged.append(cur_group)
                cur_group = [i]
        merged.append(cur_group)
        # 计算每个大峰的中心和范围
        peaks_merged = []
        major_peak = None
        for group in merged:
            idx = peaks[group]
            total_count = np.sum(hist[idx])
            depth_range = (peak_positions[group[0]], peak_positions[group[-1]])
            center_depth = np.mean(peak_positions[group])
            peaks_merged.append({
                'center': center_depth,
                'count': int(total_count),
                'range': depth_range
            })
            if total_count > 0.4*depth_valid.size: # 如果某个峰值占比超过40%，则认为是主要峰值
                major_peak = peaks_merged[-1]
                break
            if total_count > 0.2*depth_valid.size and major_peak is None: # 如果某个峰值占比超过20%且没有超过40%的峰，则取深度最小的峰
                major_peak = peaks_merged[-1]
        if major_peak is None:
            major_peak = peaks_merged[0] # 否则取第一个峰
        #print(f"Major peak found at depth {major_peak['center']:.3f} m , range [{major_peak['range'][0]:.3f}, {major_peak['range'][1]:.3f}] m, with count {major_peak['count']}, {major_peak['count']/depth_valid.size*100:.1f}% of valid points.") 
        valid_pixels = self.findValidPix(major_peak['range'][0], major_peak['range'][1], depth_need)
        center = self.findMassCenter(valid_pixels)
        if center is None:
            print("No valid pixels found in the major peak depth range.")
            return None
        center_u, center_v, valid_points= center
        return center_u + box_range[1], center_v + box_range[0], major_peak['center'], valid_points

    def findValidPix(self, depth_floor, depth_ceiling, img):
        depth_img = img.astype(np.float32)
        valid_pixels = (depth_img >= depth_floor) & (depth_img <= depth_ceiling) & (~np.isnan(depth_img))
        return valid_pixels
    
    def findMassCenter(self, valid_pixels):
        ys, xs = np.nonzero(valid_pixels)
        if len(xs) == 0:
            return None
        center_u = np.mean(xs)
        center_v = np.mean(ys)
        count = len(xs)
        return center_u, center_v, count
