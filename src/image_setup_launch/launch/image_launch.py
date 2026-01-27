from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    
    spear_vision_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('spear_vision'),
                'launch',
                'small_board_pose.launch.py',
            ])
        )
    )

    qr_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('comm_ws'),
                'launch',
                'kfs_qr.launch.py',
            ])
        )
    )

    return LaunchDescription([
        # Depth camera node
        Node(
            package='depth_camera_pkg',
            executable='depth_camera_node',
            name='depth_camera',
            output='screen',
        ),

        spear_vision_launch,
        qr_launch,
    ])