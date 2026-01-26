#读取一帧/camera/color/image_raw话题的图像
import rclpy
from my_cv_bridge import ImageSubscribe_t
from rclpy.qos import qos_profile_sensor_data

def get_a_image():
    rclpy.init()
    image_subscriber = ImageSubscribe_t('/camera/color/image_raw', node_name="image_getter", qos_profile=qos_profile_sensor_data)

    # Spin the node for a short time to ensure we receive an image
    rclpy.spin_once(image_subscriber, timeout_sec=2.0)

    if image_subscriber.latest_image is not None:
        image = image_subscriber.latest_image
        image_subscriber.get_logger().info('Returning the latest image')
    else:
        image_subscriber.get_logger().warn('No image received within the timeout period')
        image = None

    # Destroy the node explicitly
    image_subscriber.destroy_node()
    rclpy.shutdown()
    return image
