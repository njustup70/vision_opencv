import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from cv_bridge import CvBridge
import numpy as np
from image_geometry import PinholeCameraModel
import sensor_msgs_py.point_cloud2 as pc2

def cam_to_pix(X, Y, Z, model):
    uv = model.project3dToPixel((X, Y, Z))
    depth = Z
    return uv[0], uv[1], depth

class CameraToPixel(Node):
    def __init__(self):
        super().__init__('camera_to_pixel')
        self.cameraInfoInit = False
        self.bridge = CvBridge()
        self.model = PinholeCameraModel()
        self.PointToDepth = None
        self.depth_data = None
        self.timelist = [0] * 10
        self.timeListHead = 0
        self.create_subscription(CameraInfo, '/camera/depth/camera_info', self.info_init_callback, 10)
        #msg = wait_for_message('/camera/depth/camera_info', CameraInfo, self)
        #self.model.fromCameraInfo(msg)
        self.create_subscription(Image, '/camera/depth/image_raw', self.depth_callback, 10)
        self.create_subscription(PointCloud2, '/camera/depth/points', self.point_callback, 10)
        #width = msg.width
        #height = msg.height
        #self.depth_map = np.full((height, width), np.nan, dtype=np.float32)
        self.width = 0
        self.height = 0
        self.get_logger().info('Waiting for point frames...')

    def info_init_callback(self, msg):
        if self.cameraInfoInit:
            return
        self.model.fromCameraInfo(msg)
        self.cameraInfoInit = True
        self.width = msg.width
        self.height = msg.height
        self.depth_map = np.full((self.height, self.width), np.nan, dtype=np.float32)

    def depth_callback(self, msg):
        self.get_logger().info('Received depth image frame...')
        depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough').astype(np.float32)
        self.depth_data = depth_img / 1000.0  # Convert mm to meters
    
    def dep_errors(self, depth_map, depth_data):
        point_count = 0
        err_average = 0.0
        err_sum = 0.0
        valid_count = 0
        unvalid_count = 0
        deviant20_count = 0
        self.get_logger().info(f'{self.width}x{self.height} depth comparison:')
        for v in range(self.height):
            for u in range(self.width):
                depth_from_points = depth_map[v, u]
                depth_from_image = depth_data[v, u]
                if not (np.isnan(depth_from_points) or depth_from_points == 0) and not (np.isnan(depth_from_image) or depth_from_image == 0):
                    err_value = abs(depth_from_points - depth_from_image) / depth_from_image
                    err_sum += err_value
                    point_count += 1
                    if err_value > 0.2 and err_value < 0.5:
                        deviant20_count += 1
                    elif err_value >= 0.5:
                        unvalid_count += 1
                    else:
                        valid_count += 1
        err_average = err_sum / point_count if point_count > 0 else 0.0
        return point_count, valid_count, deviant20_count, unvalid_count, err_average

    def point_callback(self, msg):
        if self.model.P is None:
            return  # 相机模型未初始化，先跳过
        if self.depth_data is None:
            return  # 深度图像未出现，先跳过
        self.get_logger().info('Processing point cloud frame...')
        points = np.array(list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)))
        for point in points:
            X, Y, Z = point
            u, v, depth = cam_to_pix(X, Y, Z, self.model)
            u = int(round(u))
            v = int(round(v))
            #self.get_logger().info(f"3D Point ({X:.3f}, {Y:.3f}, {Z:.3f}) -> Pixel ({int(u)}, {int(v)}) with Depth {depth:.3f} m")
            if 0 <= u < self.width and 0 <= v < self.height:
                if np.isnan(self.depth_map[v, u]) or depth < self.depth_map[v, u]:
                    self.depth_map[v, u] = depth
        # Now compare self.depth_map with self.depth_data
        point_count, valid_count, deviant20_count, unvalid_count, err_average = self.dep_errors(self.depth_map, self.depth_data)
        self.get_logger().info(f"Valid points: {valid_count},{valid_count/point_count if point_count > 0 else 0:.2f}, Deviant >20%: {deviant20_count}, Invalid(Deviant>50%): {unvalid_count}, Average Error: {err_average*100:.2f}%")

def main():
    rclpy.init()
    node = CameraToPixel()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()