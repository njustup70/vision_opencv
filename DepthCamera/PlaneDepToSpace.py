from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import numpy as np
from image_geometry import PinholeCameraModel
import yaml

import os
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"

def load_camera_info(yaml_path: str) -> CameraInfo:
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    msg = CameraInfo()
    msg.width  = data['image_width']
    msg.height = data['image_height']
    msg.distortion_model = data['distortion_model']
    msg.header.frame_id = data.get('header', {}).get('frame_id', '')
    def to_float_list(x):
        if isinstance(x, str):
            x = x.strip('[]').replace(',', ' ').split()
        return [float(i) for i in x]
    msg.d = to_float_list(data['distortion_coefficients']['data'])
    msg.k = to_float_list(data['camera_matrix']['data'])
    msg.r = to_float_list(data['rectification_matrix']['data'])
    msg.p = to_float_list(data['projection_matrix']['data'])
    msg.binning_x = data.get('binning_x', 0)
    msg.binning_y = data.get('binning_y', 0)
    roi = data.get('roi', {})
    msg.roi.x_offset = roi.get('x_offset', 0)
    msg.roi.y_offset = roi.get('y_offset', 0)
    msg.roi.height = roi.get('height', 0)
    msg.roi.width = roi.get('width', 0)
    msg.roi.do_rectify = roi.get('do_rectify', False)
    return msg

def pix_to_cam(u, v, depth, model):
    ray = model.projectPixelTo3dRay((u, v))
    muit = 1.0 / ray[2]
    X = ray[0] * muit * depth
    Y = ray[1] * muit * depth
    Z = ray[2] * muit * depth # Z = depth
    return X, Y, Z

class DepthCamera:
    def __init__(self):
        self.bridge = CvBridge()
        self.model_d = PinholeCameraModel()
        self.model_c = PinholeCameraModel()
        self.bin_width = 0.01  # Histogram bin width in meters
        self.d2c_r = None
        self.d2c_t = None
        self.peak_width = 3 * self.bin_width
        self.bin_min_percent = 0.005

    def loadCameraInfo(self, info_d = None, info_c = None, info_d2c = None):
        if info_d is None:
            self.model_d.fromCameraInfo(load_camera_info('DepthCamera/depth_camera_info.yaml'))
        else:
            self.model_d.fromCameraInfo(info_d)

        if info_c is None:
            self.model_c.fromCameraInfo(load_camera_info('DepthCamera/color_camera_info.yaml'))
        else:
            self.model_c.fromCameraInfo(info_c)

        if info_d2c is None:
            with open('DepthCamera/depth_to_color_info.yaml', 'r') as f:
                data = yaml.safe_load(f)
            rot = data['depth_to_color_extrinsics']['rotation']['data']
            trans = data['depth_to_color_extrinsics']['translation']['data']
        else:
            rot = info_d2c['rotation']
            trans = info_d2c['translation']
        self.d2c_r = np.array(rot).reshape(3, 3)
        self.d2c_t = np.array(trans).reshape(3, 1)

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

if __name__ == '__main__':
    import rclpy
    from rclpy.node import Node
    import time
    import cv2

class PixelToCamera(Node):
    def __init__(self):
        super().__init__('pixel_to_camera')
        self.info_msg = None
        self.get_logger().info('Waiting for /camera/depth/camera_info...')
        try:
            self.info_msg = self.wait_for_camera_info()
            self.get_logger().info('Loaded camera info from topic.')
        except TimeoutError:
            self.get_logger().warn('Timeout waiting for /camera/depth/camera_info, loading from YAML instead.')
        self.depth_camera = DepthCamera()
        self.depth_camera.loadCameraInfo(info_d=self.info_msg)
        self.create_subscription(Image, '/camera/depth/image_raw', self.depth_callback, 10)
        self.create_subscription(Image, '/camera/color/image_raw', self.color_callback, 10)
        self.get_logger().info('Waiting for camera_info and depth frames...')
        self.depth_img = None
        self.color_img = None
        self.range = [103,485,247,535] # test range [左上，右下]

        cv2.namedWindow("Color Image")
        cv2.setMouseCallback("Color Image", self.mouse_callback)
        self.one_point_clicked = False
        self.point_chosen = [(0,0),(0,0)]

        self.timelist = [0] * 10
        self.timeListHead = 0
    '''
    def info_init_callback(self, msg):
        if self.cameraInfoInit:
            return
        self.model_d.fromCameraInfo(msg)
        self.cameraInfoInit = True
    '''
    def wait_for_camera_info(self, timeout_sec=1.0):
        #阻塞等待一次 /camera/depth/camera_info 消息
        future = rclpy.task.Future()
        def callback(msg):
            if not future.done():
                future.set_result(msg)
        self.create_subscription(CameraInfo, '/camera/depth/camera_info', callback, 10)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_sec)
        if not future.done():
            raise TimeoutError("CameraInfo timeout")
        return future.result()

    def timetest(self):
        self.timelist[self.timeListHead] = time.time()
        self.get_logger().info(f"time interval: {self.timelist[self.timeListHead]-self.timelist[(self.timeListHead-1)%10]:.3f} s")
        self.get_logger().info(f"10 time average interval: {(self.timelist[self.timeListHead]-self.timelist[(self.timeListHead+1)%10])/10:.3f} s")
        self.timeListHead = (self.timeListHead + 1) % 10

    def color_callback(self, msg):
        self.color_img = msg
        #self.timetest()

    def depth_callback(self, msg):
        if self.color_img is None:
            return
        #self.timetest()
        self.depth_img = msg
        cv2_color_img = self.depth_camera.bridge.imgmsg_to_cv2(self.color_img, desired_encoding='passthrough')
        cv2_depth_img = self.depth_camera.bridge.imgmsg_to_cv2(self.depth_img, desired_encoding='passthrough').astype(np.uint16)
        color_resized = cv2.resize(cv2_color_img, (cv2_depth_img.shape[1], cv2_depth_img.shape[0]), interpolation=cv2.INTER_LINEAR)
        # 深度图叠加到彩色图，颜色表示深度
        depth_colored = cv2.applyColorMap(cv2.convertScaleAbs(cv2_depth_img, alpha=0.03), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(color_resized, 0.6, depth_colored, 0.4, 0)
        color_resized = overlay
        center = self.depth_camera.depthImageFindCenter(self.range, self.depth_img)
        if center is not None:
            u, v, depth, valid_points_count = center
            #print(cv2_color_img.shape, color_resized.shape)
            self.get_logger().info(f"Mass center at pixel ({u:.1f}, {v:.1f}) with depth {depth:.3f} m, valid points percent: {valid_points_count / ((self.range[2]-self.range[0])*(self.range[3]-self.range[1]))*100:.1f}%")
            cv2.circle(color_resized, (int(u), int(v)), 5, (65535,65535,0), -1) # 黄色圆点
            # 显示圆点坐标及深度
            x,y,z = pix_to_cam(u, v, depth, self.depth_camera.model_d)
            cv2.putText(color_resized, f"({x:.3f},{y:.3f},{z:.3f})", (int(u)+10, int(v)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (65535,65535,0), 1)
        # 显示选定区域
        cv2.rectangle(color_resized, (self.range[1], self.range[0]), (self.range[3], self.range[2]), (32768,32768,32768), 2)
        cv2.imshow("Color Image", color_resized)
        cv2.waitKey(1)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if not self.one_point_clicked:
                self.point_chosen[0] = (x, y)
                self.one_point_clicked = True
                self.get_logger().info(f"First point chosen at ({x}, {y})")
            else:
                self.point_chosen[1] = (x, y)
                self.one_point_clicked = False
                x1 = min(self.point_chosen[0][0], self.point_chosen[1][0])
                y1 = min(self.point_chosen[0][1], self.point_chosen[1][1])
                x2 = max(self.point_chosen[0][0], self.point_chosen[1][0])
                y2 = max(self.point_chosen[0][1], self.point_chosen[1][1])
                self.range = [y1, x1, y2, x2]
                self.get_logger().info(f"Second point chosen at ({x}, {y}), new range set to [{y1}, {x1}, {y2}, {x2}]")


def main():
    rclpy.init()
    node = PixelToCamera()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
