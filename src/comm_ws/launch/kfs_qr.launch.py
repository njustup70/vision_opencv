from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        # R1机器人：二维码显示节点
        Node(
            package='comm_ws',
            executable='qr_detect_node',
            name='qr_display_r1',
            output='screen',
            parameters=[{'node_type': 'R1'}],
            emulate_tty=True,
        ),
        
        # R2机器人：摄像头节点
        Node(
            package='comm_ws',
            executable='camera_node',
            name='camera_r2',
            output='screen',
            parameters=[
                {'camera_index': 11},
                {'fps': 60},
                {'brightness': 10.0},
                {'contrast': 8.0},
                {'exposure': 300.0}
            ],
            emulate_tty=True,
        ),
        
        # R2机器人：二维码识别节点
        Node(
            package='comm_ws',
            executable='qr_detect_node',
            name='qr_detect_r2',
            output='screen',
            parameters=[{'node_type': 'R2'}],
            emulate_tty=True,
        ),
    ])
