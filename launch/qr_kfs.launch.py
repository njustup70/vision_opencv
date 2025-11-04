from launch import LaunchDescription
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    vision_opencv_dir = get_package_share_directory('vision_opencv')
    
    # 摄像头配置文件路径
    usb_cam_config = os.path.join(
        vision_opencv_dir,
        'CameraCalibration',
        'yamls',          
        'usb_cam_d435.yaml'
    )
    # 校验配置文件存在性
    if not os.path.isfile(usb_cam_config):
        raise FileNotFoundError(f"摄像头配置文件不存在：{usb_cam_config}，请检查路径！")
        
    # --------------------------
    # 节点1：USB摄像头节点
    # --------------------------
    usb_cam_node = Node(
        package='usb_cam',          
        executable='usb_cam_node_exe', 
        name='usb_cam_node',
        output='screen',
        parameters=[usb_cam_config],
        remappings=[
            ('image_raw', '/usb_cam/image_raw'),
            ('camera_info', '/usb_cam/camera_info')
        ]
    )

    # --------------------------
    # 节点2：QR-KFS解析节点（核心，解析12个位置状态）
    # --------------------------
    qr_kfs_decoder_node = Node(
        package='vision_opencv',      
        executable='qr_kfs_decoder',
        name='qr_kfs_decoder_node',
        output='screen',
        parameters=[
            {
                'camera_topic': '/usb_cam/image_raw'
            }
        ]
    )

    # 只启动摄像头+QR解析节点
    return LaunchDescription([
        usb_cam_node,
        qr_kfs_decoder_node
    ])
