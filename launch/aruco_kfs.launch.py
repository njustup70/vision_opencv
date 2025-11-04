from launch import LaunchDescription
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    vision_opencv_dir = get_package_share_directory('vision_opencv')
    usb_cam_config = os.path.join(
        vision_opencv_dir,
        'CameraCalibration',
        'yamls',          
        'usb_cam_d435.yaml'
    )
    if not os.path.isfile(usb_cam_config):
        raise FileNotFoundError(f"摄像头配置文件不存在：{usb_cam_config}，请检查路径！")
        
    # --------------------------
    # 节点1：USB摄像头节点（采集图像+加载内参）
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
    # 节点2：Aruco识别节点
    # --------------------------
    aruco_detector_node = Node(
        package='vision_opencv',      
        executable='aruco_detector_node',
        name='aruco_detector_node',
        output='screen',
        parameters=[
            {
                'marker_length': 0.15,  # 15cm码
                'aruco_dict_type': 'DICT_7X7_1000'
            }
        ],
        remappings=[
            ('/usb_cam/image_raw', '/usb_cam/image_raw'),
            ('/usb_cam/camera_info', '/usb_cam/camera_info')
        ]
    )

    # --------------------------
    # 节点3：KFS解析节点（4码映射12位置）
    # --------------------------
    kfs_mapper_node = Node(
        package='vision_opencv', 
        executable='kfs_mapper',
        name='kfs_mapper_node', 
        output='screen',
        remappings=[
            ('/aruco_markers', '/aruco_markers')
        ]
    )

    return LaunchDescription([
        usb_cam_node,
        aruco_detector_node,
        kfs_mapper_node
    ])
